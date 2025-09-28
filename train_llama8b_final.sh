#!/bin/bash

set -e

echo "=== Turkish LLM Training with QLoRA (4-bit) ==="
echo "Optimized for RTX 4060 GPU"
echo 

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Error: Virtual environment not found in .venv directory"
  echo "Please create a virtual environment first"
  exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dataset exists
DATASET_PATH="/home/nabil/big-turkish-llm-project/datasets/pilots/pilot_finans.jsonl"
if [ ! -f "$DATASET_PATH" ]; then
  echo "Error: Dataset not found at $DATASET_PATH"
  exit 1
fi

# Install required packages
echo "Installing required packages..."
pip install -q transformers accelerate peft bitsandbytes datasets

# Create output directory
OUTPUT_DIR="/home/nabil/big-turkish-llm-project/output/turkish_llama_qlora"
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to $OUTPUT_DIR"

# Create offload directory
OFFLOAD_DIR="/home/nabil/big-turkish-llm-project/offload"
mkdir -p "$OFFLOAD_DIR"

echo "Starting training with 4-bit quantization..."
echo "This process may take some time depending on your hardware."

# Run training with 4-bit quantization for RTX 4060
python scripts/train_qlora.py \
  --model_name_or_path "meta-llama/Meta-Llama-3.1-8B" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 1 \
  --max_seq_length 256 \
  --use_4bit \
  --bnb_quant_type nf4 \
  --offload_folder "$OFFLOAD_DIR"

echo "Training completed successfully!"
echo "Model saved to $OUTPUT_DIR"