#!/bin/bash

# Create merged directory
mkdir -p /home/nabil/big-turkish-llm-project/datasets/merged

# Process each sector
for sector in finans saglik egitim; do
  echo Processing
