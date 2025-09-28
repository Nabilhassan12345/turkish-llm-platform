#!/bin/bash

# Pre-push check script for Turkish LLM
# Scans for large files and secrets before pushing

CONFIRM_FLAG="--confirm"

# Check if confirm flag is provided
if [[ "$1" != "$CONFIRM_FLAG" ]]; then
    echo "❌ Error: You must run this script with the --confirm flag to acknowledge the checks"
    echo "Usage: ./prepush_check.sh --confirm"
    exit 1
fi

echo "🔍 Running pre-push checks..."

# Check for large files (>100MB)
echo "Checking for large files (>100MB)..."
LARGE_FILES=$(find . -type f -size +100M | grep -v "\.git")

if [[ -n "$LARGE_FILES" ]]; then
    echo "❌ Error: Found files larger than 100MB:"
    echo "$LARGE_FILES"
    echo "Please remove these files before pushing."
    exit 1
fi

# Check for common secrets
echo "Checking for potential secrets..."
SECRET_PATTERNS=(
    "api_key"
    "apikey"
    "secret"
    "password"
    "token"
    "credential"
    "auth"
    "BEGIN PRIVATE KEY"
    "BEGIN RSA PRIVATE KEY"
    "BEGIN DSA PRIVATE KEY"
    "BEGIN EC PRIVATE KEY"
)

FOUND_SECRETS=false
for pattern in "${SECRET_PATTERNS[@]}"; do
    MATCHES=$(grep -r --include="*.py" --include="*.sh" --include="*.yml" --include="*.yaml" --include="*.json" --include="*.md" -i "$pattern" . | grep -v "prepush_check.sh" || true)
    if [[ -n "$MATCHES" ]]; then
        echo "⚠️ Potential secret found: $pattern"
        echo "$MATCHES"
        FOUND_SECRETS=true
    fi
done

if [[ "$FOUND_SECRETS" == true ]]; then
    echo "❌ Error: Potential secrets found in the codebase."
    echo "Please review and remove any secrets before pushing."
    exit 1
fi

# Check for demo model placeholder
if [[ ! -d "demo_models/healthcare-small" ]]; then
    echo "⚠️ Warning: demo_models/healthcare-small directory not found."
    echo "Please create the directory and add the DEMO_PLACEHOLDER_README.txt file."
fi

# All checks passed
echo "✅ All pre-push checks passed!"
exit 0