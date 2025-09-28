#!/usr/bin/env python3
"""
Model Selection & MoE Configuration Script
Compares Mixtral-7B vs LLaMA-3-8B for Turkish fine-tuning on RTX 4060
Configures 2-expert MoE routing
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import yaml
import time
import psutil
import GPUtil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare different models for Turkish fine-tuning."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory = self._get_gpu_memory()
        self.system_memory = self._get_system_memory()

        logger.info(f"Device: {self.device}")
        logger.info(f"GPU Memory: {self.gpu_memory:.2f} GB")
        logger.info(f"System Memory: {self.system_memory:.2f} GB")

    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryFree / 1024
            return 0.0
        except:
            return 0.0

    def _get_system_memory(self) -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / (1024**3)

    def compare_models(self):
        """Compare Mixtral-7B vs LLaMA-3-8B."""
        print("🔍 Model Comparison: Mixtral-7B vs LLaMA-3-8B")
        print("=" * 60)

        models = {
            "mixtral_7b": {
                "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "params": "7B",
                "architecture": "MoE (8 experts, 2 active)",
                "context_length": 32768,
                "languages": ["Multilingual", "Good Turkish support"],
                "pros": [
                    "Excellent Turkish language understanding",
                    "MoE architecture for efficient inference",
                    "Strong instruction following",
                    "Good performance on business tasks",
                ],
                "cons": [
                    "Larger memory footprint",
                    "More complex training",
                    "Higher computational requirements",
                ],
            },
            "llama3_8b": {
                "name": "meta-llama/Llama-3-8B-Instruct",
                "params": "8B",
                "architecture": "Dense Transformer",
                "context_length": 8192,
                "languages": ["Multilingual", "Good Turkish support"],
                "pros": [
                    "Simpler architecture",
                    "Easier to fine-tune",
                    "Lower memory requirements",
                    "Good performance on business tasks",
                ],
                "cons": [
                    "Shorter context length",
                    "Less efficient than MoE",
                    "May need more training data",
                ],
            },
        }

        # Print comparison table
        print(f"{'Model':<15} {'Params':<8} {'Architecture':<20} {'Context':<10}")
        print("-" * 60)

        for model_id, model_info in models.items():
            print(
                f"{model_id:<15} {model_info['params']:<8} {model_info['architecture']:<20} {model_info['context_length']:<10}"
            )

        print("\n📊 Detailed Comparison:")
        for model_id, model_info in models.items():
            print(f"\n🔸 {model_id.upper()}:")
            print(f"   Architecture: {model_info['architecture']}")
            print(f"   Context Length: {model_info['context_length']:,} tokens")
            print(f"   Languages: {model_info['languages']}")

            print(f"   ✅ Pros:")
            for pro in model_info["pros"]:
                print(f"      • {pro}")

            print(f"   ❌ Cons:")
            for con in model_info["cons"]:
                print(f"      • {con}")

        # Recommendation for RTX 4060
        print(f"\n🎯 Recommendation for RTX 4060 ({self.gpu_memory:.1f}GB VRAM):")

        if self.gpu_memory >= 6.0:
            print("✅ Mixtral-7B: Can handle with QLoRA + 4-bit quantization")
            print("   - Better Turkish performance")
            print("   - MoE efficiency benefits")
            print("   - Higher quality outputs")
        else:
            print("✅ LLaMA-3-8B: Better fit for limited VRAM")
            print("   - Lower memory requirements")
            print("   - Easier fine-tuning")
            print("   - Good Turkish support")

        return models


class MoERouter:
    """2-expert MoE router for Turkish business sectors."""

    def __init__(self, config_path: str = "configs/sectors.yaml"):
        self.config_path = config_path
        self.sectors = self._load_sectors()
        self.expert_assignments = self._assign_experts_to_sectors()

    def _load_sectors(self) -> Dict:
        """Load sector configuration."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["sectors"]

    def _assign_experts_to_sectors(self) -> Dict[str, List[str]]:
        """Assign sectors to 2 experts based on domain similarity."""

        # Group sectors by domain similarity
        expert1_sectors = [
            "finance_banking",
            "insurance",
            "asset_tracking",
            "ecommerce",
            "tourism_hospitality",
            "logistics",
            "telecommunications",
        ]

        expert2_sectors = [
            "healthcare",
            "education",
            "legal",
            "public_administration",
            "manufacturing",
            "energy",
            "energy_distribution",
            "agriculture",
            "transportation",
            "construction_architecture",
            "smart_cities",
            "mobility",
            "defense_security",
            "emergency_disaster",
        ]

        # Create expert assignments
        expert_assignments = {
            "expert_1": {
                "name": "Business & Commerce Expert",
                "sectors": expert1_sectors,
                "description": "Handles financial, commercial, and service-oriented sectors",
            },
            "expert_2": {
                "name": "Infrastructure & Public Services Expert",
                "sectors": expert2_sectors,
                "description": "Handles infrastructure, public services, and industrial sectors",
            },
        }

        return expert_assignments

    def route_query(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """Route a query to appropriate experts."""

        # Simple keyword-based routing
        text_lower = text.lower()

        expert1_score = 0
        expert2_score = 0

        # Score based on sector keywords
        for sector_id, sector_info in self.sectors.items():
            sector_keywords = sector_info.get("keywords", [])

            # Count keyword matches
            matches = sum(
                1 for keyword in sector_keywords if keyword.lower() in text_lower
            )

            if sector_id in self.expert_assignments["expert_1"]["sectors"]:
                expert1_score += matches
            elif sector_id in self.expert_assignments["expert_2"]["sectors"]:
                expert2_score += matches

        # Normalize scores
        total_score = expert1_score + expert2_score
        if total_score > 0:
            expert1_confidence = expert1_score / total_score
            expert2_confidence = expert2_score / total_score
        else:
            expert1_confidence = 0.5
            expert2_confidence = 0.5

        # Determine routing
        if expert1_confidence > confidence_threshold:
            primary_expert = "expert_1"
            secondary_expert = "expert_2"
            primary_confidence = expert1_confidence
        elif expert2_confidence > confidence_threshold:
            primary_expert = "expert_2"
            secondary_expert = "expert_1"
            primary_confidence = expert2_confidence
        else:
            # Use both experts with equal confidence
            primary_expert = "expert_1"
            secondary_expert = "expert_2"
            primary_confidence = 0.5

        return {
            "primary_expert": primary_expert,
            "secondary_expert": secondary_expert,
            "primary_confidence": primary_confidence,
            "expert_scores": {
                "expert_1": expert1_confidence,
                "expert_2": expert2_confidence,
            },
            "routing_strategy": "2-expert MoE with confidence-based selection",
        }

    def get_expert_info(self) -> Dict:
        """Get information about the experts."""
        return self.expert_assignments

    def test_routing(self):
        """Test the routing system with sample queries."""
        print("\n🧪 Testing 2-Expert MoE Routing")
        print("=" * 50)

        test_queries = [
            "Banka kredisi almak istiyorum, nasıl başvurabilirim?",
            "Hastane randevusu almak istiyorum",
            "Üniversite sınavına hazırlanıyorum, hangi kurslar önerilir?",
            "E-ticaret sitesi kurmak istiyorum",
            "Fabrika üretim süreçlerini optimize etmek istiyorum",
            "Enerji tasarrufu için önerileriniz nelerdir?",
            "Turizm acentesi açmak istiyorum",
            "Sigorta poliçesi hakkında bilgi istiyorum",
        ]

        for query in test_queries:
            routing = self.route_query(query)
            print(f"\n🔍 Query: {query}")
            print(f"   Primary Expert: {routing['primary_expert']}")
            print(f"   Confidence: {routing['primary_confidence']:.2f}")
            print(
                f"   Expert Scores: Expert1={routing['expert_scores']['expert_1']:.2f}, Expert2={routing['expert_scores']['expert_2']:.2f}"
            )


class MoETrainingConfig:
    """Configuration for MoE training."""

    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.config = self._create_training_config()

    def _create_training_config(self) -> Dict:
        """Create training configuration for MoE model."""

        if "mixtral" in self.model_name.lower():
            # Mixtral-specific configuration
            config = {
                "model": {
                    "name": self.model_name,
                    "type": "MoE",
                    "experts": 8,
                    "active_experts": 2,
                    "router_strategy": "top2",
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
                },
                "data": {
                    "max_length": 2048,
                    "truncation": True,
                    "padding": "max_length",
                },
            }
        else:
            # LLaMA-3 configuration
            config = {
                "model": {
                    "name": self.model_name,
                    "type": "Dense",
                    "experts": 1,
                    "active_experts": 1,
                    "router_strategy": "single",
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
                    "batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "max_grad_norm": 0.3,
                    "warmup_steps": 100,
                    "weight_decay": 0.01,
                },
                "data": {
                    "max_length": 2048,
                    "truncation": True,
                    "padding": "max_length",
                },
            }

        return config

    def save_config(self, output_path: str = "configs/moe_training.yaml"):
        """Save training configuration to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ Training configuration saved to: {output_path}")

    def print_config(self):
        """Print training configuration."""
        print("\n⚙️ MoE Training Configuration")
        print("=" * 50)
        print(f"Model: {self.config['model']['name']}")
        print(f"Type: {self.config['model']['type']}")
        print(f"Experts: {self.config['model']['experts']}")
        print(f"Active Experts: {self.config['model']['active_experts']}")
        print(f"Router Strategy: {self.config['model']['router_strategy']}")

        print(f"\nTraining Method: {self.config['training']['method']}")
        print(f"Quantization: {self.config['training']['quantization']}")
        print(f"LoRA r: {self.config['training']['lora_r']}")
        print(f"LoRA alpha: {self.config['training']['lora_alpha']}")

        print(f"\nLearning Rate: {self.config['optimization']['learning_rate']}")
        print(f"Batch Size: {self.config['optimization']['batch_size']}")
        print(
            f"Gradient Accumulation: {self.config['optimization']['gradient_accumulation_steps']}"
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model Selection & MoE Configuration")
    parser.add_argument(
        "--compare", action="store_true", help="Compare Mixtral vs LLaMA-3"
    )
    parser.add_argument("--test-routing", action="store_true", help="Test MoE routing")
    parser.add_argument(
        "--create-config", action="store_true", help="Create training configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name for config",
    )

    args = parser.parse_args()

    print("🚀 Model Selection & MoE Configuration for Turkish LLM")
    print("=" * 70)

    # Model comparison
    if args.compare:
        comparator = ModelComparison()
        models = comparator.compare_models()

    # Test MoE routing
    if args.test_routing:
        router = MoERouter()
        router.test_routing()

        # Show expert assignments
        print("\n📋 Expert Assignments:")
        expert_info = router.get_expert_info()
        for expert_id, expert_data in expert_info.items():
            print(f"\n🔸 {expert_data['name']}:")
            print(f"   Description: {expert_data['description']}")
            print(f"   Sectors: {', '.join(expert_data['sectors'])}")

    # Create training configuration
    if args.create_config:
        config_creator = MoETrainingConfig(args.model)
        config_creator.print_config()
        config_creator.save_config()

    # If no specific action, run all
    if not any([args.compare, args.test_routing, args.create_config]):
        print("Running all components...")

        # Model comparison
        comparator = ModelComparison()
        models = comparator.compare_models()

        # Test MoE routing
        router = MoERouter()
        router.test_routing()

        # Create training configuration
        config_creator = MoETrainingConfig()
        config_creator.print_config()
        config_creator.save_config()

    print("\n🎯 Next Steps:")
    print("1. Choose your preferred model based on the comparison")
    print("2. Test the MoE routing with your specific queries")
    print("3. Review and adjust the training configuration")
    print("4. Proceed to Phase A3: Full-Parameter + Efficiency Pipeline")


if __name__ == "__main__":
    main()
