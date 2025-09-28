#!/bin/bash

# Enhanced Setup and Training Script for Turkish LLM
# Handles complete environment setup, dependency management, and training

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/setup_train_$TIMESTAMP.log"

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
Turkish LLM Setup and Training Script

This script performs complete environment setup and training:
1. System requirements check
2. Python environment setup
3. CUDA/GPU verification
4. Dependency installation
5. Dataset preparation
6. Model training
7. Validation and deployment

Usage: $0 [OPTIONS]

Options:
    --quick-setup          Skip interactive prompts, use defaults
    --skip-deps           Skip dependency installation
    --skip-data           Skip dataset preparation
    --skip-training       Skip training (setup only)
    --gpu-check           Only perform GPU compatibility check
    --clean-install       Remove existing environment and reinstall
    --sector SECTOR       Train specific sector only
    --batch-size SIZE     Training batch size (default: 4)
    --epochs EPOCHS       Number of training epochs (default: 3)
    --deploy              Deploy after successful training
    --monitor             Enable training monitoring
    -h, --help            Show this help message

Examples:
    $0                           # Interactive setup and training
    $0 --quick-setup --deploy    # Quick setup with deployment
    $0 --gpu-check              # Check GPU compatibility only
    $0 --sector technology       # Setup and train technology sector only

EOF
}

# Default values
QUICK_SETUP=false
SKIP_DEPS=false
SKIP_DATA=false
SKIP_TRAINING=false
GPU_CHECK_ONLY=false
CLEAN_INSTALL=false
SECTOR="all"
BATCH_SIZE=4
EPOCHS=3
DEPLOY=false
MONITOR=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-setup)
            QUICK_SETUP=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --gpu-check)
            GPU_CHECK_ONLY=true
            shift
            ;;
        --clean-install)
            CLEAN_INSTALL=true
            shift
            ;;
        --sector)
            SECTOR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
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
mkdir -p "$LOG_DIR" "adapters" "datasets" "models" "cache" "logs"

log_info "=== Turkish LLM Setup and Training Started ==="
log_info "Timestamp: $TIMESTAMP"
log_info "Log file: $LOG_FILE"
log_info "Configuration:"
log_info "  - Quick Setup: $QUICK_SETUP"
log_info "  - Skip Dependencies: $SKIP_DEPS"
log_info "  - Skip Data Prep: $SKIP_DATA"
log_info "  - Skip Training: $SKIP_TRAINING"
log_info "  - GPU Check Only: $GPU_CHECK_ONLY"
log_info "  - Clean Install: $CLEAN_INSTALL"
log_info "  - Sector: $SECTOR"
log_info "  - Batch Size: $BATCH_SIZE"
log_info "  - Epochs: $EPOCHS"
log_info "  - Deploy: $DEPLOY"
log_info "  - Monitor: $MONITOR"

# System Requirements Check
log_info "=== System Requirements Check ==="

# Check OS
OS_INFO=$(uname -a)
log_info "Operating System: $OS_INFO"

# Check Python version
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3 is required but not installed."
fi

PYTHON_VERSION=$(python3 --version)
log_info "Python Version: $PYTHON_VERSION"

# Check minimum Python version (3.8+)
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    error_exit "Python 3.8+ is required. Current version: $PYTHON_VERSION"
fi

log_success "Python version check passed"

# GPU and CUDA Check
log_info "=== GPU and CUDA Check ==="

if ! command -v nvidia-smi &> /dev/null; then
    error_exit "NVIDIA GPU driver not found. Please install NVIDIA drivers."
fi

# Get GPU information
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader)
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")

log_info "GPU Count: $GPU_COUNT"
log_info "GPU Information:"
echo "$GPU_INFO" | while read line; do
    log_info "  $line"
done
log_info "CUDA Version: $CUDA_VERSION"

# Check GPU memory (minimum 8GB recommended)
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

log_info "GPU Memory: ${FREE_MEM}MB free / ${TOTAL_MEM}MB total"

if [[ $TOTAL_MEM -lt 8000 ]]; then
    log_warn "GPU has less than 8GB memory. Training may be slow or fail."
    if [[ "$QUICK_SETUP" == "false" ]]; then
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

if [[ $FREE_MEM -lt 6000 ]]; then
    log_warn "Low free GPU memory: ${FREE_MEM}MB. Consider closing other GPU applications."
fi

log_success "GPU check completed"

if [[ "$GPU_CHECK_ONLY" == "true" ]]; then
    log_info "GPU check completed. Exiting as requested."
    exit 0
fi

# Disk Space Check
log_info "=== Disk Space Check ==="
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
log_info "Available disk space: ${AVAILABLE_SPACE}GB"

if [[ $AVAILABLE_SPACE -lt 50 ]]; then
    log_warn "Low disk space: ${AVAILABLE_SPACE}GB. Recommend at least 50GB free."
    if [[ "$QUICK_SETUP" == "false" ]]; then
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Python Environment Setup
log_info "=== Python Environment Setup ==="

if [[ "$CLEAN_INSTALL" == "true" && -d ".venv" ]]; then
    log_info "Removing existing virtual environment..."
    rm -rf .venv
fi

if [[ ! -d ".venv" ]]; then
    log_info "Creating Python virtual environment..."
    python3 -m venv .venv
fi

log_info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# Dependency Installation
if [[ "$SKIP_DEPS" == "false" ]]; then
    log_info "=== Dependency Installation ==="
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        log_info "Creating requirements.txt..."
        cat > requirements.txt << EOF
# Core ML libraries
torch>=2.0.0
torchvision
torchaudio
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.39.0
datasets>=2.12.0

# Training and optimization
wandb
tensorboard
scipy
numpy
pandas
scikit-learn

# Web and API
fastapi>=0.100.0
uvicorn[standard]
websockets
cors

# Audio processing
librosa
soundfile
speech-recognition
pydub

# Utilities
tqdm
click
pyyaml
jsonlines
requests
aiofiles
python-multipart

# Development
pytest
black
flake8
mypy
EOF
    fi
    
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Verify critical installations
    log_info "Verifying installations..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, CUDA Version: {torch.version.cuda}')" || error_exit "PyTorch verification failed"
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || error_exit "Transformers verification failed"
    python -c "import peft; print(f'PEFT: {peft.__version__}')" || error_exit "PEFT verification failed"
    python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')" || error_exit "BitsAndBytes verification failed"
    
    log_success "All dependencies installed and verified"
else
    log_info "Skipping dependency installation as requested"
fi

# Accelerate Configuration
log_info "=== Accelerate Configuration ==="

if [[ ! -f "accelerate_config.yaml" ]]; then
    log_info "Creating accelerate configuration..."
    cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# Dataset Preparation
if [[ "$SKIP_DATA" == "false" ]]; then
    log_info "=== Dataset Preparation ==="
    
    # Create datasets directory structure
    mkdir -p datasets/pilots datasets/merged datasets/synthetic
    
    # Check for existing datasets
    DATASET_COUNT=$(find datasets -name "*.jsonl" | wc -l)
    log_info "Found $DATASET_COUNT existing dataset files"
    
    if [[ $DATASET_COUNT -eq 0 ]]; then
        log_info "No datasets found. Generating synthetic training data..."
        
        # Generate sample datasets for each sector
        SECTORS=("technology" "healthcare" "education" "finance_banking" "legal")
        
        for sector in "${SECTORS[@]}"; do
            log_info "Generating synthetic data for sector: $sector"
            python scripts/generate_training_data.py --sector "$sector" --output "datasets/pilots/pilot_${sector}.jsonl" --size 500
        done
        
        log_success "Synthetic datasets generated"
    else
        log_info "Using existing datasets"
    fi
else
    log_info "Skipping dataset preparation as requested"
fi

# Training Execution
if [[ "$SKIP_TRAINING" == "false" ]]; then
    log_info "=== Starting Training Process ==="
    
    # Prepare training arguments
    TRAIN_ARGS="--sector $SECTOR --batch-size $BATCH_SIZE --epochs $EPOCHS"
    
    if [[ "$DEPLOY" == "true" ]]; then
        TRAIN_ARGS+=" --deploy"
    fi
    
    if [[ "$MONITOR" == "true" ]]; then
        TRAIN_ARGS+=" --monitor"
    fi
    
    log_info "Executing training with arguments: $TRAIN_ARGS"
    
    # Execute training script
    if bash run_training.sh $TRAIN_ARGS; then
        log_success "Training completed successfully"
    else
        log_error "Training failed"
        exit 1
    fi
else
    log_info "Skipping training as requested"
fi

# Final Setup Verification
log_info "=== Final Verification ==="

# Check if adapters were created
if [[ -d "adapters" ]]; then
    ADAPTER_COUNT=$(find adapters -name "adapter_model.bin" | wc -l)
    log_info "Found $ADAPTER_COUNT trained adapters"
fi

# Check services
if [[ -f "services/inference_service.py" ]]; then
    log_success "Inference service ready"
fi

if [[ -f "services/voice_orchestrator.py" ]]; then
    log_success "Voice orchestrator ready"
fi

# Generate setup report
log_info "Generating setup report..."
cat > "logs/setup_report_$TIMESTAMP.txt" << EOF
Turkish LLM Setup Report
Generated: $(date)

=== System Information ===
OS: $OS_INFO
Python: $PYTHON_VERSION
GPU Count: $GPU_COUNT
CUDA Version: $CUDA_VERSION
GPU Memory: ${FREE_MEM}MB free / ${TOTAL_MEM}MB total
Disk Space: ${AVAILABLE_SPACE}GB available

=== Configuration ===
Sector: $SECTOR
Batch Size: $BATCH_SIZE
Epochs: $EPOCHS
Deploy: $DEPLOY
Monitor: $MONITOR

=== Results ===
Setup Status: SUCCESS
Training Status: $([ "$SKIP_TRAINING" == "true" ] && echo "SKIPPED" || echo "COMPLETED")
Adapters Created: $ADAPTER_COUNT

=== Next Steps ===
1. Test the inference service: python -m services.inference_service
2. Start the voice orchestrator: python -m services.voice_orchestrator
3. Launch the UI: cd ui && npm run dev
4. Access the application at http://localhost:3000

For more information, see the documentation in README.md
EOF

log_success "=== Setup and Training Completed Successfully! ==="
log_info "Setup report saved to: logs/setup_report_$TIMESTAMP.txt"
log_info "Log file: $LOG_FILE"

if [[ "$SKIP_TRAINING" == "false" ]]; then
    log_info "\nTo start the services:"
    log_info "  1. Inference API: python -m services.inference_service"
    log_info "  2. Voice Service: python -m services.voice_orchestrator"
    log_info "  3. Web UI: cd ui && npm run dev"
fi

log_info "\nSetup completed in $(($(date +%s) - $(date -d "$TIMESTAMP" +%s 2>/dev/null || echo 0)))s"
