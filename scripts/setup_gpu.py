#!/usr/bin/env python3
"""
GPU Infrastructure Setup and Verification Script
For RTX 4060 with CUDA 11.8 and cuDNN 8.9
"""

import subprocess
import sys
import os
import platform
import json
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode


def check_system_info():
    """Check system information."""
    print("🔍 Checking System Information...")

    # OS info
    os_info = platform.platform()
    print(f"OS: {os_info}")

    # Python version
    python_version = sys.version
    print(f"Python: {python_version}")

    # Check if running in WSL
    if "microsoft" in os_info.lower() or "wsl" in os_info.lower():
        print("✅ Running in WSL2 environment")
    else:
        print("⚠️ Not running in WSL2")


def check_nvidia_drivers():
    """Check NVIDIA driver installation."""
    print("\n🔍 Checking NVIDIA Drivers...")

    stdout, stderr, returncode = run_command("nvidia-smi", check=False)

    if returncode == 0:
        print("✅ NVIDIA drivers are installed")
        print("📊 GPU Information:")
        print(stdout)
        return True
    else:
        print("❌ NVIDIA drivers not found")
        print("💡 Install NVIDIA drivers first:")
        print("   sudo apt update")
        print("   sudo apt install nvidia-driver-535")
        return False


def check_cuda_installation():
    """Check CUDA installation."""
    print("\n🔍 Checking CUDA Installation...")

    # Check CUDA version
    stdout, stderr, returncode = run_command("nvcc --version", check=False)

    if returncode == 0:
        print("✅ CUDA is installed")
        print(f"📊 CUDA Version: {stdout}")
        return True
    else:
        print("❌ CUDA not found")
        print("💡 Install CUDA 11.8:")
        print(
            "   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
        )
        print("   sudo sh cuda_11.8.0_520.61.05_linux.run")
        return False


def check_cudnn_installation():
    """Check cuDNN installation."""
    print("\n🔍 Checking cuDNN Installation...")

    # Check cuDNN version
    cudnn_paths = [
        "/usr/local/cuda/include/cudnn.h",
        "/usr/include/cudnn.h",
        "/opt/cuda/include/cudnn.h",
    ]

    cudnn_found = False
    for path in cudnn_paths:
        if os.path.exists(path):
            print(f"✅ cuDNN found at: {path}")
            cudnn_found = True
            break

    if not cudnn_found:
        print("❌ cuDNN not found")
        print("💡 Install cuDNN 8.9:")
        print("   Download from NVIDIA Developer Portal")
        print("   sudo cp cuda/include/cudnn*.h /usr/local/cuda/include")
        print("   sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64")
        print(
            "   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*"
        )

    return cudnn_found


def check_pytorch_gpu():
    """Check PyTorch GPU support."""
    print("\n🔍 Checking PyTorch GPU Support...")

    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"📊 CUDA version: {torch.version.cuda}")
            print(f"📊 cuDNN version: {torch.backends.cudnn.version()}")
            print(f"📊 GPU count: {torch.cuda.device_count()}")

            # Check RTX 4060 specifically
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"📊 GPU {i}: {gpu_name}")

                if "4060" in gpu_name.lower():
                    print("🎯 RTX 4060 detected!")

                    # Test GPU memory
                    torch.cuda.set_device(i)
                    test_tensor = torch.randn(1000, 1000).cuda()
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    print(f"📊 GPU Memory Test: {memory_allocated:.2f} GB allocated")

                    return True
            else:
                print("⚠️ RTX 4060 not found, but CUDA is working")
                return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False

    except ImportError:
        print("❌ PyTorch not installed")
        print("💡 Install PyTorch with CUDA support:")
        print(
            "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n🔧 Installing Dependencies...")

    # Install PyTorch with CUDA 11.8 support
    print("📦 Installing PyTorch with CUDA 11.8 support...")
    stdout, stderr, returncode = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    )

    if returncode == 0:
        print("✅ PyTorch installed successfully")
    else:
        print("❌ Failed to install PyTorch")
        print(f"Error: {stderr}")

    # Install other required packages
    packages = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "datasets",
        "evaluate",
    ]

    for package in packages:
        print(f"📦 Installing {package}...")
        stdout, stderr, returncode = run_command(f"pip install {package}")

        if returncode == 0:
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            print(f"Error: {stderr}")


def create_environment_script():
    """Create environment setup script."""
    print("\n📝 Creating Environment Setup Script...")

    script_content = """#!/bin/bash
# GPU Environment Setup Script for RTX 4060

echo "🔧 Setting up GPU environment for RTX 4060..."

# Add CUDA to PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_ROOT=/usr/local/cuda-11.8

# Verify setup
echo "📊 CUDA Version:"
nvcc --version

echo "📊 GPU Status:"
nvidia-smi

echo "📊 PyTorch CUDA Support:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "✅ Environment setup complete!"
"""

    with open("setup_gpu_env.sh", "w") as f:
        f.write(script_content)

    # Make executable
    run_command("chmod +x setup_gpu_env.sh")
    print("✅ Created setup_gpu_env.sh")


def main():
    """Main function."""
    print("🚀 GPU Infrastructure Setup and Verification")
    print("=" * 50)

    # Check system info
    check_system_info()

    # Check NVIDIA drivers
    drivers_ok = check_nvidia_drivers()

    # Check CUDA
    cuda_ok = check_cuda_installation()

    # Check cuDNN
    cudnn_ok = check_cudnn_installation()

    # Check PyTorch GPU support
    pytorch_ok = check_pytorch_gpu()

    # Summary
    print("\n📊 Setup Summary:")
    print(f"NVIDIA Drivers: {'✅' if drivers_ok else '❌'}")
    print(f"CUDA 11.8: {'✅' if cuda_ok else '❌'}")
    print(f"cuDNN 8.9: {'✅' if cudnn_ok else '❌'}")
    print(f"PyTorch GPU: {'✅' if pytorch_ok else '❌'}")

    if all([drivers_ok, cuda_ok, cudnn_ok, pytorch_ok]):
        print("\n🎉 All GPU components are working correctly!")
        print("Your RTX 4060 is ready for Turkish LLM training!")
    else:
        print("\n⚠️ Some components need to be installed/fixed")
        print("Run the setup commands shown above")

        # Offer to install dependencies
        response = input(
            "\n🤔 Would you like to install PyTorch and other dependencies? (y/n): "
        )
        if response.lower() == "y":
            install_dependencies()

    # Create environment script
    create_environment_script()

    print("\n📚 Next Steps:")
    print("1. Source the environment: source setup_gpu_env.sh")
    print("2. Run this script again to verify everything is working")
    print("3. Proceed to Phase A2: Model Selection & MoE Configuration")


if __name__ == "__main__":
    main()
