#!/usr/bin/env python3
"""
Turkish AI Agent - GitHub Upload Helper
This script helps you upload your project to GitHub correctly.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_git_status():
    """Check if git is initialized and files are ready."""
    print("🔍 Checking Git status...")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("📝 Initializing Git repository...")
        if not run_command("git init", "Git initialization"):
            return False
    
    # Check git status
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("📋 Files ready to commit:")
        print(result.stdout)
        return True
    else:
        print("⚠️  No changes to commit. Make sure you have files to upload.")
        return False

def create_gitignore():
    """Create a comprehensive .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Model cache
.cache/
cache/

# Experiment tracking
wandb/
mlruns/

# Temporary files
temp/
tmp/
*.tmp

# Large model files (if any)
*.bin
*.safetensors
*.pt
*.ckpt
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore file")

def main():
    """Main upload process."""
    print("🚀 Turkish AI Agent - GitHub Upload Helper")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('services') or not os.path.exists('rag_system.py'):
        print("❌ Please run this script from your Turkish AI Agent project directory")
        return
    
    print("📁 Current directory:", os.getcwd())
    print("📋 Project components found:")
    
    # Check for key components
    components = [
        ('services/', 'Core services'),
        ('rag_system.py', 'RAG system'),
        ('ui/', 'React frontend'),
        ('scripts/', 'Training scripts'),
        ('docker-compose.yml', 'Docker configuration'),
        ('README.md', 'Documentation')
    ]
    
    for component, description in components:
        if os.path.exists(component):
            print(f"  ✅ {component} - {description}")
        else:
            print(f"  ⚠️  {component} - {description} (not found)")
    
    print("\n" + "=" * 50)
    
    # Create .gitignore
    create_gitignore()
    
    # Check git status
    if not check_git_status():
        print("❌ Git setup failed. Please check the errors above.")
        return
    
    print("\n🎯 Next Steps:")
    print("1. Add files to git:")
    print("   git add .")
    print("2. Create initial commit:")
    print("   git commit -m \"Initial commit: Turkish AI Agent with RAG, voice interaction, and 22 sector models\"")
    print("3. Create GitHub repository at: https://github.com/new")
    print("   - Name: turkish-ai-agent")
    print("   - Description: Enterprise Turkish AI platform with RAG, voice interaction, and 22 sector-specific models")
    print("   - Make it Public")
    print("   - Don't initialize with README")
    print("4. Connect and push:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/turkish-ai-agent.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n🏆 Your project includes:")
    print("✅ 7 Production Models (Healthcare, Education, Finance)")
    print("✅ RAG System with FAISS vector search")
    print("✅ 22 Sector Adapters for Turkish business")
    print("✅ Real-time Voice with WebSocket + STT/TTS")
    print("✅ Modern React Frontend with TypeScript")
    print("✅ Production Infrastructure with Docker + monitoring")
    print("✅ Advanced Testing with load testing and benchmarking")
    
    print("\n🎉 This will be an impressive portfolio project!")

if __name__ == "__main__":
    main()
