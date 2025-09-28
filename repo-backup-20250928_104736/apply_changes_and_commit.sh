#!/bin/bash

# Apply changes and commit script for Turkish LLM
# Stages only generated files and commits them

CONFIRM_FLAG="--confirm"

# Check if confirm flag is provided
if [[ "$1" != "$CONFIRM_FLAG" ]]; then
    echo "❌ Error: You must run this script with the --confirm flag to acknowledge the changes"
    echo "Usage: ./apply_changes_and_commit.sh --confirm"
    exit 1
fi

echo "📝 Applying changes and committing..."

# List of generated files to stage
GENERATED_FILES=(
    "Dockerfile.demo"
    "docker-compose.demo.yml"
    "requirements_demo.txt"
    "real_demo.py"
    "services/demo_service.py"
    "scripts/docker_build_demo.sh"
    "scripts/docker_run_demo.sh"
    "DEMO_RUN_OUTPUT.txt"
    ".github/workflows/ci.yml"
    "docs/REPRODUCE.md"
    "prepush_check.sh"
    "apply_changes_and_commit.sh"
    "FINAL_DEMO_READY.md"
    "demo_models/healthcare-small/DEMO_PLACEHOLDER_README.txt"
)

# Stage each file if it exists
for file in "${GENERATED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        git add "$file"
        echo "✅ Staged: $file"
    else
        echo "⚠️ Warning: $file not found, skipping"
    fi
done

# Create commit
echo "Creating commit..."
git commit -m "Add polished demo and CI integration

- Added Dockerfile.demo for CPU-based demo
- Added docker-compose.demo.yml for easy deployment
- Created real_demo.py with smoke test support
- Added CI workflow with linting and smoke tests
- Added documentation for reproduction
- Added push safety tooling"

echo "✅ Changes applied and committed successfully!"
echo ""
echo "Next steps:"
echo "1) Upload full weights to HF and paste URLs in ARTIFACTS_MANIFEST.json"
echo "2) Run prepush_check.sh --confirm"
echo "3) Push the changes"