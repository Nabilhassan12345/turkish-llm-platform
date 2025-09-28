#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Install required packages if not already installed
pip install -q transformers accelerate peft bitsandbytes datasets

# Set model and dataset paths
MODEL=" meta-llama/Meta-Llama-3.1-8B\
DATASET=\/home/nabil/big-turkish-llm-project/datasets/pilots/pilot_dataset.json\
OUTPUT_DIR=\/home/nabil/big-turkish-llm-project/output/turkish_llama_qlora\

#
