# Turkish LLM: Enterprise-Grade AI for Turkish Business

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Turkish LLM Project

[![CI Build Status](https://img.shields.io/github/workflow/status/turkish-ai/turkish-llm/Turkish%20LLM%20Demo%20CI/main)](https://github.com/turkish-ai/turkish-llm/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-available-brightgreen.svg)](https://turkish-ai.org/demo)

*A state-of-the-art Turkish language model with sector-specific adapters*

## 🌟 **Features**

- **Sector-Specific Models**: Specialized for healthcare, finance, education, and more
- **Voice Integration**: Seamless speech-to-text and text-to-speech capabilities
- **Efficient Architecture**: Adapter-based approach for minimal resource usage
- **Comprehensive Benchmarks**: Rigorous performance and accuracy testing
- **Production-Ready**: Containerized deployment with monitoring and scaling

## 🚀 **Quick Start**

```bash
# Clone the repository
git clone https://github.com/turkish-ai/turkish-llm.git
cd turkish-llm

# Run the demo
./scripts/docker_build_demo.sh
./scripts/docker_run_demo.sh

# Access the API at http://localhost:8000
```

## 🏗️ **Architecture**

The Turkish LLM project implements a modular architecture with:

- **Base Model**: Gemma-2B foundation model
- **Adapter Layers**: Sector-specific fine-tuning with LoRA
- **Inference Service**: FastAPI-based REST API
- **Voice Service**: WebSocket-based speech processing
- **Router**: Intelligent request routing to appropriate models
- **Monitoring**: Prometheus and Grafana integration
