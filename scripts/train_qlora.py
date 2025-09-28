#!/usr/bin/env python3
# Minimal QLoRA train wrapper using Transformers + PEFT + bitsandbytes
import argparse, json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bnb_quant_type", default="nf4")
    p.add_argument("--offload_folder", default="./offload")
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading dataset:", args.dataset_path)
    ds = load_dataset("json", data_files=args.dataset_path)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=args.bnb_quant_type,
    )
    print("Loading model (may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=args.offload_folder,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def preprocess(example):
        if isinstance(example, dict):
            if "input" in example and "output" in example:
                text = example["input"] + "\n" + example["output"]
            elif "dialogue" in example:
                text = example["dialogue"]
            else:
                text = json.dumps(example, ensure_ascii=False)
        else:
            text = str(example)
        return tokenizer(text, truncation=True, max_length=args.max_seq_length)

    print("Tokenizing dataset...")
    ds = ds.map(lambda ex: preprocess(ex), batched=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        fp16=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if "train" in ds else ds,
        tokenizer=tokenizer,
    )
    print("Starting training...")
    trainer.train()
    print("Saving model to", args.output_dir)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
