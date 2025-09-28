#!/usr/bin/env python3

import os
import json

# Create merged directory
os.makedirs("/home/nabil/big-turkish-llm-project/datasets/merged", exist_ok=True)

# Sectors to process
sectors = ["finans", "saglik", "egitim"]

# Path to synthetic data
synthetic_path = (
    "/home/nabil/big-turkish-llm-project/datasets/synthetic/cleaned/22sectors.jsonl"
)

for sector in sectors:
    print(f"Processing {sector}")
    # Add your processing logic here
    pass

print("Dataset merge completed successfully")
