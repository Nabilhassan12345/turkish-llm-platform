#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install transformers accelerate peft bitsandbytes datasets

# Run training with 4-bit quantization for RTX 4060
python scripts/train_qlora.py   --model_name_or_path 'meta-llama/Meta-Llama-3.1-8B'   --dataset_path '/home/nabil/big-turkish-llm-project/datasets/pilots/pilot_dataset.json'   --output_dir '/home/nabil/big-turkish-llm-project/output/turkish_llama_qlora'   --per_device_train_batch_size 1   --gradient_accumulation_steps 32   --num_train_epochs 1   --max_seq_length 256   --use_4bit   --bnb_quant_type nf4   --offload_folder '/home/nabil/big-turkish-llm-project/offload'
