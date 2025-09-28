#!/bin/bash

# Enhanced Turkish LLM Training Orchestration Script
# Supports multi-sector training, monitoring, and deployment

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CONFIG_FILE="$SCRIPT_DIR/configs/model_config.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
Turkish LLM Training Orchestration Script

Usage: $0 [OPTIONS]

Options:
    -s, --sector SECTOR     Train specific sector (default: all)
    -m, --model MODEL       Base model to use (default: meta-llama/Llama-2-8b-hf)
    -e, --epochs EPOCHS     Number of training epochs (default: 3)
    -b, --batch-size SIZE   Batch size per device (default: 4)
    -l, --learning-rate LR  Learning rate (default: 2e-4)
    -d, --data-path PATH    Custom dataset path
    -o, --output-dir DIR    Output directory for adapters
    -c, --config CONFIG     Configuration file path
    --dry-run              Show what would be done without executing
    --resume               Resume from checkpoint
    --deploy               Deploy after training
    --monitor              Enable monitoring during training
    -h, --help             Show this help message

Sectors:
    technology, healthcare, education, finance_banking, legal,
    public_administration, manufacturing, insurance, tourism_hospitality,
    ecommerce, telecommunications, real_estate, logistics, agriculture,
    energy, construction, automotive, retail, consulting, entertainment

Examples:
    $0 --sector technology --epochs 5
    $0 --sector healthcare --batch-size 2 --monitor
    $0 --deploy  # Train all sectors and deploy

EOF
}

# Default values
SECTOR="all"
MODEL="meta-llama/Llama-2-8b-hf"
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE="2e-4"
DATA_PATH=""
OUTPUT_DIR="adapters"
DRY_RUN=false
RESUME=false
DEPLOY=false
MONITOR=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--sector)
            SECTOR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "datasets" "models" "cache"

log_info "=== Turkish LLM Training Orchestration Started ==="
log_info "Timestamp: $TIMESTAMP"
log_info "Log file: $LOG_FILE"
log_info "Configuration:"
log_info "  - Sector: $SECTOR"
log_info "  - Model: $MODEL"
log_info "  - Epochs: $EPOCHS"
log_info "  - Batch Size: $BATCH_SIZE"
log_info "  - Learning Rate: $LEARNING_RATE"
log_info "  - Output Dir: $OUTPUT_DIR"
log_info "  - Dry Run: $DRY_RUN"
log_info "  - Resume: $RESUME"
log_info "  - Deploy: $DEPLOY"
log_info "  - Monitor: $MONITOR"

# System checks
log_info "Performing system checks..."

# Check Python and virtual environment
if [[ ! -d ".venv" ]]; then
    log_warn "Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

log_info "Activating virtual environment..."
source .venv/bin/activate

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    error_exit "NVIDIA GPU not found. This training requires CUDA-capable GPU."
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
log_info "GPU Information: $GPU_INFO"

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")
log_info "CUDA Version: $CUDA_VERSION"

# Check available memory
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ $FREE_MEM -lt 8000 ]]; then
    log_warn "Low GPU memory: ${FREE_MEM}MB. Training may fail or be slow."
fi

# Install/update dependencies
log_info "Installing/updating dependencies..."
if [[ "$DRY_RUN" == "false" ]]; then
    pip install --upgrade pip
    pip install -r requirements.txt || error_exit "Failed to install dependencies"
fi

# Verify critical packages
log_info "Verifying critical packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || error_exit "PyTorch verification failed"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || error_exit "Transformers verification failed"
python -c "import peft; print(f'PEFT: {peft.__version__}')" || error_exit "PEFT verification failed"

# Define sectors
ALL_SECTORS=("technology" "healthcare" "education" "finance_banking" "legal" "public_administration" "manufacturing" "insurance" "tourism_hospitality" "ecommerce" "telecommunications" "real_estate" "logistics" "agriculture" "energy" "construction" "automotive" "retail" "consulting" "entertainment")

# Determine sectors to train
if [[ "$SECTOR" == "all" ]]; then
    SECTORS_TO_TRAIN=("${ALL_SECTORS[@]}")
else
    SECTORS_TO_TRAIN=("$SECTOR")
fi

log_info "Sectors to train: ${SECTORS_TO_TRAIN[*]}"

# Training function
train_sector() {
    local sector=$1
    local sector_output_dir="$OUTPUT_DIR/$sector"
    local sector_data_path="$DATA_PATH"
    
    if [[ -z "$sector_data_path" ]]; then
        sector_data_path="datasets/pilots/pilot_${sector}.jsonl"
    fi
    
    log_info "Training sector: $sector"
    log_info "  - Data path: $sector_data_path"
    log_info "  - Output dir: $sector_output_dir"
    
    # Check if dataset exists
    if [[ ! -f "$sector_data_path" ]]; then
        log_warn "Dataset not found: $sector_data_path. Generating synthetic data..."
        if [[ "$DRY_RUN" == "false" ]]; then
            python scripts/generate_training_data.py --sector "$sector" --output "$sector_data_path" --size 1000
        fi
    fi
    
    # Create sector output directory
    mkdir -p "$sector_output_dir"
    
    # Prepare training command
    local train_cmd="accelerate launch --config_file accelerate_config.yaml scripts/train_qlora.py"
    train_cmd+=" --model_name_or_path $MODEL"
    train_cmd+=" --dataset_path $sector_data_path"
    train_cmd+=" --output_dir $sector_output_dir"
    train_cmd+=" --per_device_train_batch_size $BATCH_SIZE"
    train_cmd+=" --gradient_accumulation_steps 4"
    train_cmd+=" --num_train_epochs $EPOCHS"
    train_cmd+=" --learning_rate $LEARNING_RATE"
    train_cmd+=" --max_seq_length 2048"
    train_cmd+=" --use_4bit"
    train_cmd+=" --logging_steps 10"
    train_cmd+=" --save_steps 500"
    train_cmd+=" --eval_steps 500"
    train_cmd+=" --warmup_steps 100"
    train_cmd+=" --lr_scheduler_type cosine"
    train_cmd+=" --optim paged_adamw_32bit"
    train_cmd+=" --bf16"
    train_cmd+=" --remove_unused_columns False"
    train_cmd+=" --run_name turkish_llm_${sector}_${TIMESTAMP}"
    
    if [[ "$RESUME" == "true" ]]; then
        train_cmd+=" --resume_from_checkpoint"
    fi
    
    if [[ "$MONITOR" == "true" ]]; then
        train_cmd+=" --report_to wandb"
    fi
    
    log_info "Training command: $train_cmd"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Start training with timeout and monitoring
        local start_time=$(date +%s)
        
        if timeout 7200 bash -c "$train_cmd" 2>&1 | tee -a "$LOG_FILE"; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_success "Sector $sector training completed in ${duration}s"
            
            # Validate adapter
            if [[ -f "$sector_output_dir/adapter_model.bin" ]]; then
                log_success "Adapter saved successfully: $sector_output_dir/adapter_model.bin"
            else
                log_error "Adapter not found after training: $sector_output_dir/adapter_model.bin"
                return 1
            fi
        else
            log_error "Training failed or timed out for sector: $sector"
            return 1
        fi
    else
        log_info "[DRY RUN] Would execute: $train_cmd"
    fi
}

# Main training loop
log_info "Starting training for ${#SECTORS_TO_TRAIN[@]} sectors..."

failed_sectors=()
successful_sectors=()

for sector in "${SECTORS_TO_TRAIN[@]}"; do
    log_info "Processing sector: $sector ($(date))")
    
    if train_sector "$sector"; then
        successful_sectors+=("$sector")
    else
        failed_sectors+=("$sector")
        log_error "Failed to train sector: $sector"
    fi
    
    # Brief pause between sectors
    sleep 5
done

# Training summary
log_info "=== Training Summary ==="
log_success "Successful sectors (${#successful_sectors[@]}): ${successful_sectors[*]}"
if [[ ${#failed_sectors[@]} -gt 0 ]]; then
    log_error "Failed sectors (${#failed_sectors[@]}): ${failed_sectors[*]}"
fi

# Post-training tasks
if [[ "$DRY_RUN" == "false" && ${#successful_sectors[@]} -gt 0 ]]; then
    log_info "Performing post-training tasks..."
    
    # Create adapter registry
    log_info "Creating adapter registry..."
    python scripts/create_adapter_registry.py --adapters_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/registry.json"
    
    # Run validation tests
    log_info "Running validation tests..."
    python scripts/test_adapters.py --adapters_dir "$OUTPUT_DIR" --config "$CONFIG_FILE"
    
    # Generate training report
    log_info "Generating training report..."
    python scripts/generate_training_report.py --log_file "$LOG_FILE" --output "$LOG_DIR/training_report_$TIMESTAMP.html"
    
    # Deployment
    if [[ "$DEPLOY" == "true" ]]; then
        log_info "Starting deployment..."
        bash scripts/deploy.sh --adapters_dir "$OUTPUT_DIR" --config "$CONFIG_FILE"
    fi
fi

# Final status
if [[ ${#failed_sectors[@]} -eq 0 ]]; then
    log_success "=== All training completed successfully! ==="
    exit 0
else
    log_error "=== Training completed with errors ==="
    exit 1
fi
