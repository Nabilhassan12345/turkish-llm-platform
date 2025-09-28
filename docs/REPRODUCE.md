# Turkish LLM Reproduction Guide

This document provides instructions for reproducing the Turkish LLM demo environment.

## Quick Start with Docker

The easiest way to run the Turkish LLM demo is using Docker:

```bash
# Clone the repository
git clone https://github.com/turkish-ai/turkish-llm.git
cd turkish-llm

# Build the Docker image
./scripts/docker_build_demo.sh
# or on Windows:
# scripts\docker_build_demo.sh

# Run the Docker container
./scripts/docker_run_demo.sh
# or on Windows:
# scripts\docker_run_demo.sh
```

The demo API will be available at http://localhost:8000

## Alternative: Using docker-compose

You can also use docker-compose to run the demo:

```bash
# Build and run with docker-compose
docker-compose -f docker-compose.demo.yml up -d

# Check logs
docker-compose -f docker-compose.demo.yml logs -f
```

## Testing the API

Once the server is running, you can test it with:

```bash
# Health check
curl http://localhost:8000/health

# Inference request
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"text": "Aile hekimimi nasıl değiştirebilirim?", "sector": "healthcare"}'
```

## Running Without Docker

If you prefer to run without Docker:

```bash
# Install dependencies
pip install -r requirements_demo.txt

# Run the demo API
uvicorn services.demo_service:app --host 0.0.0.0 --port 8000

# Or run the demo script
python real_demo.py
```

## Using Real Model Weights

By default, the demo uses a placeholder for model weights. To use real weights:

1. Download the model from Hugging Face:
   ```
   https://huggingface.co/turkish-ai/healthcare-small
   ```

2. Extract the files to `demo_models/healthcare-small/`, replacing the placeholder.

## Troubleshooting

If you encounter issues:

1. Check that all dependencies are installed
2. Ensure ports are not already in use
3. Check Docker logs for detailed error messages