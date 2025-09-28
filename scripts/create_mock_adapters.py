#!/usr/bin/env python3
"""
Create Mock Adapters for Phase A4 Testing
Creates placeholder adapters for all 22 Turkish business sectors.
"""

import os
import json
import yaml
from pathlib import Path


def create_mock_adapters():
    """Create mock adapters for all sectors."""

    # Load sector configuration
    with open("configs/sectors.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sectors = config["sectors"]

    # Create adapters directory
    adapters_dir = Path("adapters")
    adapters_dir.mkdir(exist_ok=True)

    print(f"Creating mock adapters for {len(sectors)} sectors...")

    for sector_id, sector_info in sectors.items():
        adapter_path = adapters_dir / f"{sector_id}_adapter"
        adapter_path.mkdir(exist_ok=True)

        # Create mock model files that inference service expects
        (adapter_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mock_adapter",
                    "sector_id": sector_id,
                    "sector_name": sector_info["name"],
                    "architectures": ["MockModelForCausalLM"],
                    "vocab_size": 50257,
                    "n_positions": 1024,
                    "n_ctx": 1024,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        # Create mock tokenizer files
        (adapter_path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "model_type": "mock_tokenizer",
                    "sector_id": sector_id,
                    "vocab_size": 50257,
                    "pad_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                },
                indent=2,
            )
        )

        # Create mock vocab file
        (adapter_path / "vocab.json").write_text(
            json.dumps({"<|endoftext|>": 50256, "test": 0}, indent=2)
        )

        # Create mock merges file
        (adapter_path / "merges.txt").write_text("")

        # Create mock model weights file (empty but exists)
        (adapter_path / "pytorch_model.bin").write_bytes(b"")

        # Create sector metadata
        metadata = {
            "sector_id": sector_id,
            "sector_name": sector_info["name"],
            "description": sector_info["description"],
            "keywords": sector_info["keywords"],
            "adapter_type": "mock",
            "model_name": "microsoft/DialoGPT-medium",
            "training_examples": 40,
            "accuracy": 0.85,
        }

        (adapter_path / "sector_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

        print(f"✅ Created mock adapter: {sector_id}")

    # Create training summary
    summary = {
        "total_sectors": len(sectors),
        "successful": len(sectors),
        "failed": 0,
        "adapter_type": "mock",
        "status": "completed",
    }

    (adapters_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n🎉 Successfully created {len(sectors)} mock adapters!")
    print("📁 Adapters saved to: adapters/")

    return len(sectors)


if __name__ == "__main__":
    create_mock_adapters()
