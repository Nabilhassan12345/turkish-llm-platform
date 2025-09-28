#!/usr/bin/env python3
"""
Efficient Training Pipeline for Turkish LLM
Full-parameter fine-t uning with QLoRA, 4-bit quantization, and structured pruning
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import yaml
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
import bitsandbytes as bnb
import wandb
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishEfficientDataset(Dataset):
    """Efficient dataset for Turkish training with chunking and tokenization."""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
        stride: int = 512,
        min_length: int = 128,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        self.tokenized_texts = self._tokenize_texts()

    def _tokenize_texts(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize and chunk texts efficiently."""
        tokenized_texts = []

        for text in tqdm(self.texts, desc="Tokenizing texts"):
            # Tokenize the text
            tokens = self.tokenizer.encode(text, add_special_tokens=True)

            # Create overlapping chunks
            for i in range(0, len(tokens) - self.min_length + 1, self.stride):
                chunk = tokens[i : i + self.max_length]

                if len(chunk) >= self.min_length:
                    # Pad if necessary
                    if len(chunk) < self.max_length:
                        chunk = chunk + [self.tokenizer.pad_token_id] * (
                            self.max_length - len(chunk)
                        )

                    tokenized_texts.append(
                        {
                            "input_ids": torch.tensor(chunk),
                            "attention_mask": torch.tensor([1] * len(chunk)),
                        }
                    )

        return tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return self.tokenized_texts[idx]


class StructuredPruning:
    """Structured pruning for model efficiency."""

    def __init__(self, model, pruning_ratio: float = 0.1):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.masks = {}

    def create_pruning_masks(self):
        """Create pruning masks for attention heads and MLP layers."""
        print(
            f"🔧 Creating pruning masks with {self.pruning_ratio*100}% pruning ratio..."
        )

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Prune attention heads and MLP layers
                if (
                    "q_proj" in name
                    or "k_proj" in name
                    or "v_proj" in name
                    or "o_proj" in name
                ):
                    self._prune_attention_heads(module, name)
                elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                    self._prune_mlp_layers(module, name)

    def _prune_attention_heads(self, module, name):
        """Prune attention heads based on importance."""
        # Calculate importance scores (L2 norm of weights)
        importance_scores = torch.norm(module.weight, dim=1)

        # Sort by importance and create mask
        sorted_indices = torch.argsort(importance_scores, descending=True)
        num_to_keep = int(module.weight.shape[0] * (1 - self.pruning_ratio))

        mask = torch.zeros_like(module.weight)
        mask[sorted_indices[:num_to_keep], :] = 1

        self.masks[name] = mask

        # Apply mask
        module.weight.data *= mask

    def _prune_mlp_layers(self, module, name):
        """Prune MLP layers based on importance."""
        # Calculate importance scores
        importance_scores = torch.norm(module.weight, dim=1)

        # Sort by importance and create mask
        sorted_indices = torch.argsort(importance_scores, descending=True)
        num_to_keep = int(module.weight.shape[0] * (1 - self.pruning_ratio))

        mask = torch.zeros_like(module.weight)
        mask[sorted_indices[:num_to_keep], :] = 1

        self.masks[name] = mask

        # Apply mask
        module.weight.data *= mask

    def apply_pruning(self):
        """Apply all pruning masks."""
        print("🔧 Applying structured pruning...")

        for name, module in self.model.named_modules():
            if name in self.masks:
                module.weight.data *= self.masks[name]

        print("✅ Structured pruning applied successfully")

    def get_pruning_stats(self):
        """Get statistics about pruning."""
        total_params = 0
        pruned_params = 0

        for name, mask in self.masks.items():
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()

        pruning_ratio = pruned_params / total_params if total_params > 0 else 0

        return {
            "total_params": total_params,
            "pruned_params": pruned_params,
            "pruning_ratio": pruning_ratio,
            "remaining_params": total_params - pruned_params,
        }


class EfficientTurkishTrainer:
    """Efficient trainer for Turkish LLM with QLoRA and pruning."""

    def __init__(self, config_path: str = "configs/moe_training.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()

        logger.info(f"Using device: {self.device}")
        logger.info(f"Accelerator: {self.accelerator}")

        # Initialize wandb for experiment tracking
        if self.config.get("logging", {}).get("use_wandb", False):
            wandb.init(project="turkish-llm-efficient", config=self.config)

    def _load_config(self) -> Dict:
        """Load training configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                "model": {
                    "name": "meta-llama/Llama-2-7b-hf",
                    "type": "Dense",
                    "experts": 1,
                    "active_experts": 1,
                },
                "training": {
                    "method": "QLoRA",
                    "quantization": "4-bit",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "target_modules": [
                        "q_proj",
                        "v_proj",
                        "k_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                },
                "optimization": {
                    "learning_rate": 2e-4,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "max_grad_norm": 0.3,
                    "warmup_steps": 100,
                    "weight_decay": 0.01,
                    "num_epochs": 3,
                },
                "data": {
                    "max_length": 2048,
                    "truncation": True,
                    "padding": "max_length",
                    "stride": 512,
                    "min_length": 128,
                },
                "pruning": {"enabled": True, "ratio": 0.1, "structured": True},
                "logging": {
                    "use_wandb": True,
                    "log_steps": 10,
                    "eval_steps": 100,
                    "save_steps": 500,
                },
            }

        return config

    def _setup_quantization(self):
        """Setup 4-bit quantization with bitsandbytes."""
        print("🔧 Setting up 4-bit quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        return bnb_config

    def _load_model_and_tokenizer(self, model_name: str):
        """Load model and tokenizer with quantization."""
        print(f"📥 Loading model: {model_name}")

        # Setup quantization
        bnb_config = self._setup_quantization()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        return model, tokenizer

    def _setup_lora(self, model):
        """Setup LoRA adapter."""
        print("🔧 Setting up LoRA adapter...")

        lora_config = LoraConfig(
            r=self.config["training"]["lora_r"],
            lora_alpha=self.config["training"]["lora_alpha"],
            target_modules=self.config["training"]["target_modules"],
            lora_dropout=self.config["training"]["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def _load_training_data(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Load training data from various formats."""
        print(f"📂 Loading training data from: {data_path}")

        data_path = Path(data_path)

        if data_path.is_file():
            if data_path.suffix == ".jsonl":
                texts = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if isinstance(data, dict):
                            # Handle different data formats
                            if "text" in data:
                                texts.append(data["text"])
                            elif "input" in data and "output" in data:
                                texts.append(f"{data['input']} {data['output']}")
                            elif "question" in data and "answer" in data:
                                texts.append(f"{data['question']} {data['answer']}")
                        else:
                            texts.append(str(data))
                return texts, []
            elif data_path.suffix == ".json":
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [str(item) for item in data]
                    else:
                        texts = [str(data)]
                    return texts, []
            elif data_path.suffix == ".txt":
                with open(data_path, "r", encoding="utf-8") as f:
                    texts = [line.strip() for line in f if line.strip()]
                return texts, []
        elif data_path.is_dir():
            # Load from directory
            texts = []
            for file_path in data_path.rglob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.extend([line.strip() for line in f if line.strip()])
            return texts, []

        print("⚠️ No data found, using sample data")
        return self._generate_sample_data(), []

    def _generate_sample_data(self) -> List[str]:
        """Generate sample Turkish training data."""
        sample_texts = [
            "Merhaba, nasılsınız? Ben size nasıl yardımcı olabilirim?",
            "Banka kredisi almak istiyorum. Gerekli belgeler nelerdir?",
            "Hastane randevusu almak istiyorum. Hangi doktoru önerirsiniz?",
            "Üniversite sınavına hazırlanıyorum. Hangi kurslar faydalı olur?",
            "E-ticaret sitesi kurmak istiyorum. Hangi platformu önerirsiniz?",
            "Sigorta poliçesi hakkında bilgi almak istiyorum.",
            "Turizm acentesi açmak istiyorum. Gerekli izinler nelerdir?",
            "Enerji tasarrufu için önerileriniz nelerdir?",
            "Tarım yapmak istiyorum. Hangi ürünler karlı olur?",
            "İnşaat projesi geliştirmek istiyorum. Hangi adımları takip etmeliyim?",
        ]
        return sample_texts

    def _create_dataset(self, texts: List[str], tokenizer) -> TurkishEfficientDataset:
        """Create efficient dataset."""
        print("🔧 Creating efficient dataset...")

        dataset = TurkishEfficientDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config["data"]["max_length"],
            stride=self.config["data"]["stride"],
            min_length=self.config["data"]["min_length"],
        )

        print(f"✅ Dataset created with {len(dataset)} samples")
        return dataset

    def _setup_training_args(self, output_dir: str) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["optimization"]["num_epochs"],
            per_device_train_batch_size=self.config["optimization"]["batch_size"],
            gradient_accumulation_steps=self.config["optimization"][
                "gradient_accumulation_steps"
            ],
            learning_rate=self.config["optimization"]["learning_rate"],
            max_grad_norm=self.config["optimization"]["max_grad_norm"],
            warmup_steps=self.config["optimization"]["warmup_steps"],
            weight_decay=self.config["optimization"]["weight_decay"],
            logging_steps=self.config["logging"]["log_steps"],
            evaluation_strategy=(
                "steps" if self.config["logging"]["eval_steps"] > 0 else "no"
            ),
            eval_steps=self.config["logging"]["eval_steps"],
            save_strategy="steps",
            save_steps=self.config["logging"]["save_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config["logging"]["use_wandb"] else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,
            dataloader_num_workers=4,
        )

    def train(
        self,
        model_name: str,
        data_path: str,
        output_dir: str = "models/turkish_llm_efficient",
    ):
        """Main training function."""
        print("🚀 Starting Efficient Turkish LLM Training")
        print("=" * 60)

        # Load model and tokenizer
        model, tokenizer = self._load_model_and_tokenizer(model_name)

        # Setup LoRA
        model = self._setup_lora(model)

        # Load training data
        texts, _ = self._load_training_data(data_path)

        # Split data
        train_texts, eval_texts = train_test_split(
            texts, test_size=0.1, random_state=42
        )

        # Create datasets
        train_dataset = self._create_dataset(train_texts, tokenizer)
        eval_dataset = (
            self._create_dataset(eval_texts, tokenizer) if eval_texts else None
        )

        # Setup training arguments
        training_args = self._setup_training_args(output_dir)

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Setup trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train the model
        print("🔥 Starting training...")
        trainer.train()

        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Apply pruning if enabled
        if self.config["pruning"]["enabled"]:
            print("🔧 Applying structured pruning...")
            pruner = StructuredPruning(model, self.config["pruning"]["ratio"])
            pruner.create_pruning_masks()
            pruner.apply_pruning()

            # Save pruning stats
            pruning_stats = pruner.get_pruning_stats()
            with open(f"{output_dir}/pruning_stats.json", "w") as f:
                json.dump(pruning_stats, f, indent=2)

            print(
                f"✅ Pruning applied: {pruning_stats['pruning_ratio']*100:.1f}% of parameters pruned"
            )

        # Save configuration
        with open(f"{output_dir}/training_config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        print(f"🎉 Training completed! Model saved to: {output_dir}")

        # Log final metrics
        if self.config["logging"]["use_wandb"]:
            final_metrics = trainer.evaluate()
            wandb.log(final_metrics)
            wandb.finish()

        return model, tokenizer


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Efficient Turkish LLM Training Pipeline"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model name"
    )
    parser.add_argument(
        "--data", type=str, default="data/turkish_training", help="Training data path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/turkish_llm_efficient",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/moe_training.yaml",
        help="Training config path",
    )

    args = parser.parse_args()

    # Create trainer
    trainer = EfficientTurkishTrainer(args.config)

    # Start training
    model, tokenizer = trainer.train(args.model, args.data, args.output)

    print("\n🎯 Training completed successfully!")
    print(f"📁 Model saved to: {args.output}")
    print("🚀 Ready for inference!")


if __name__ == "__main__":
    main()
