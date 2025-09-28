# Turkish LLM Demo Ready

The Turkish LLM project is now ready for demonstration with a polished demo and CI integration.

## Demo Build and Run Commands

```bash
# Build the Docker demo image
./scripts/docker_build_demo.sh
# or on Windows:
# scripts\docker_build_demo.sh

# Run the Docker demo container
./scripts/docker_run_demo.sh
# or on Windows:
# scripts\docker_run_demo.sh

# Alternative: Using docker-compose
docker-compose -f docker-compose.demo.yml up -d
```

The demo API will be available at http://localhost:8000

## Next Steps for Repository Owner

1. **Upload full weights to Hugging Face**:
   - Create a repository on Hugging Face Hub: `turkish-ai/healthcare-small`
   - Upload the model weights (keep under 100MB for the demo version)
   - Paste the URL in `ARTIFACTS_MANIFEST.json`

2. **Apply changes and commit**:
   ```bash
   ./apply_changes_and_commit.sh --confirm
   ```

3. **Run pre-push checks and push**:
   ```bash
   ./prepush_check.sh --confirm
   git push
   ```

## Demo Features

- **FastAPI Demo Service**: Simple API for demonstrating the model
- **Docker Integration**: Easy deployment with Docker
- **CI Pipeline**: Automated testing and building
- **Documentation**: Clear reproduction instructions
- **Safety Checks**: Pre-push validation for large files and secrets

## Testing the Demo

Once the server is running, you can test it with:

```bash
# Health check
curl http://localhost:8000/health

# Inference request
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"text": "Aile hekimimi nasıl değiştirebilirim?", "sector": "healthcare"}'
```

## Created Files

- `Dockerfile.demo`: CPU-based Docker image for the demo
- `docker-compose.demo.yml`: Docker Compose configuration
- `requirements_demo.txt`: Demo dependencies
- `real_demo.py`: Demo script with smoke test support
- `services/demo_service.py`: FastAPI demo service
- `scripts/docker_build_demo.sh`: Script to build the Docker image
- `scripts/docker_run_demo.sh`: Script to run the Docker container
- `DEMO_RUN_OUTPUT.txt`: Sample output from the demo
- `.github/workflows/ci.yml`: CI workflow configuration
- `docs/REPRODUCE.md`: Reproduction instructions
- `prepush_check.sh`: Pre-push safety checks
- `apply_changes_and_commit.sh`: Script to apply changes and commit
- `demo_models/healthcare-small/DEMO_PLACEHOLDER_README.txt`: Placeholder for demo model