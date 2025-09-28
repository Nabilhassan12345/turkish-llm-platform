#!/usr/bin/env python3

import os
import json

# Create merged directory
os.makedirs("/home/nabil/big-turkish-llm-project/datasets/merged", exist_ok=True)

# Sectors to process
sectors = ["finans", "saglik", "egitim"]

for sector in sectors:
    print(f"Processing {sector}")
    # Add your processing logic here
    pass

print("Merge completed successfully")
