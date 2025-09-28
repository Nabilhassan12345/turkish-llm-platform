#!/usr/bin/env python3

import os
import json
import argparse


def main():
    # Create output directory
    output_dir = "datasets/merged"
    os.makedirs(output_dir, exist_ok=True)

    # Load synthetic data
    synthetic_path = "datasets/synthetic/cleaned/22sectors.jsonl"
    synthetic_data = []
    with open(synthetic_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                synthetic_data.append(json.loads(line))
    print(f"Loaded {len(synthetic_data)} synthetic samples")


if __name__ == "__main__":
    main()
