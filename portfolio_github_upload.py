#!/usr/bin/env python3
"""
Turkish-LLM Portfolio GitHub Upload Script
-----------------------------------------
This script automates the process of preparing and uploading your Turkish-LLM
project to GitHub for portfolio presentation.
"""

import os
import subprocess
import sys
import shutil
import json
from pathlib import Path

# Configuration
GITHUB_USERNAME = ""  # Fill in your GitHub username
REPO_NAME = "turkish-ai-agent"
REPO_DESCRIPTION = "Enterprise Turkish AI platform with trained models, RAG system, and 22 sector-specific configurations"

# Files and directories to exclude from GitHub upload
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "env/",
    "venv/",
    ".venv/",
    "ENV/",
    "build/",
    "develop-eggs/",
    "dist/",
    "downloads/",
    "eggs/",
    ".eggs/",
    "lib/",
    "lib64/",
    "parts/",
    "sdist/",
    "var/",
    "*.egg-info/",
    ".installed.cfg",
    "*.egg",
    ".env",
    ".venv",
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    ".DS_Store",
    "Thumbs.db",
    "wandb/",
    "mlruns/",
    "logs/",
    "temp/",
    "tmp/",
    "cache/",
    ".cache/"
]

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def run_command(command, error_message=None):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        if error_message:
            print(f"ERROR: {error_message}")
        print(f"Command failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_gitignore():
    """Create a comprehensive .gitignore file."""
    print_header("Creating .gitignore file")
    
    gitignore_content = "\n".join(EXCLUDE_PATTERNS)
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore file created successfully")

def clean_repository():
    """Clean the repository of unnecessary files."""
    print_header("Cleaning repository")
    
    # Directories to remove if they exist
    dirs_to_remove = [
        ".venv",
        "wandb",
        "__pycache__",
        "logs",
        "temp",
        "tmp",
        "cache",
        ".cache"
    ]
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✅ Removed {dir_name}/")
            except Exception as e:
                print(f"❌ Failed to remove {dir_name}/: {e}")
    
    print("✅ Repository cleaned successfully")

def enhance_readme():
    """Enhance README.md for portfolio presentation."""
    print_header("Enhancing README.md for portfolio")
    
    try:
        with open("README.md", "r") as f:
            readme_content = f.read()
        
        # Add portfolio section if not already present
        if "## 🏆 Portfolio Highlights" not in readme_content:
            portfolio_section = """
## 🏆 Portfolio Highlights

This project demonstrates:
- **Advanced AI Engineering**: QLoRA fine-tuning, RAG implementation
- **Full-Stack Development**: React frontend + FastAPI backend
- **Production Systems**: Docker, monitoring, CI/CD
- **Turkish Language Expertise**: Rare and valuable skill
- **Enterprise Architecture**: Scalable, maintainable design

## 💼 Professional Skills Demonstrated

- **Machine Learning Engineering**: Model training, evaluation, and deployment
- **Software Architecture**: Microservices, API design
- **DevOps**: Containerization, monitoring, CI/CD
- **Frontend Development**: Modern React with TypeScript
- **Natural Language Processing**: Specialized language model fine-tuning
"""
            
            # Find the position to insert the portfolio section (before the contact section)
            if "## 📧 Contact" in readme_content:
                readme_content = readme_content.replace("## 📧 Contact", f"{portfolio_section}\n\n## 📧 Contact")
            else:
                readme_content += f"\n{portfolio_section}\n"
            
            with open("README.md", "w") as f:
                f.write(readme_content)
            
            print("✅ README.md enhanced with portfolio highlights")
        else:
            print("ℹ️ Portfolio section already exists in README.md")
    
    except Exception as e:
        print(f"❌ Failed to enhance README.md: {e}")

def initialize_git():
    """Initialize Git repository."""
    print_header("Initializing Git repository")
    
    # Check if .git directory already exists
    if os.path.exists(".git"):
        print("ℹ️ Git repository already initialized")
        return True
    
    # Initialize git repository
    if run_command("git init", "Failed to initialize git repository"):
        print("✅ Git repository initialized")
        return True
    return False

def setup_github_repository():
    """Set up GitHub repository using GitHub CLI if available."""
    print_header("Setting up GitHub repository")
    
    if not GITHUB_USERNAME:
        print("❌ Please set your GitHub username in the script configuration")
        return False
    
    # Check if gh CLI is installed
    if shutil.which("gh"):
        print("ℹ️ GitHub CLI detected, using it to create repository")
        
        # Check if already logged in
        login_status = run_command("gh auth status", "Failed to check GitHub login status")
        if "Logged in to github.com" not in login_status:
            print("ℹ️ Please login to GitHub CLI")
            run_command("gh auth login", "Failed to login to GitHub")
        
        # Create repository
        create_cmd = f'gh repo create {REPO_NAME} --public --description "{REPO_DESCRIPTION}" --source=. --remote=origin'
        if run_command(create_cmd, "Failed to create GitHub repository"):
            print(f"✅ GitHub repository '{REPO_NAME}' created successfully")
            return True
    else:
        print("ℹ️ GitHub CLI not found. Please create repository manually:")
        print(f"1. Go to https://github.com/new")
        print(f"2. Repository name: {REPO_NAME}")
        print(f"3. Description: {REPO_DESCRIPTION}")
        print(f"4. Make it Public")
        print(f"5. Don't initialize with README")
        print(f"6. Click 'Create repository'")
        
        # Provide git commands for manual setup
        remote_url = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
        print("\nAfter creating the repository, run these commands:")
        print(f"git remote add origin {remote_url}")
        print("git branch -M main")
        print("git push -u origin main")
        
        user_input = input("\nHave you created the repository? (y/n): ")
        if user_input.lower() == 'y':
            # Set up remote
            remote_cmd = f"git remote add origin {remote_url}"
            if run_command(remote_cmd, f"Failed to add remote {remote_url}"):
                print(f"✅ Remote '{remote_url}' added successfully")
                return True
    
    return False

def commit_and_push():
    """Commit changes and push to GitHub."""
    print_header("Committing and pushing to GitHub")
    
    # Add all files
    if run_command("git add .", "Failed to add files to git"):
        print("✅ Files added to git")
    
    # Commit
    commit_cmd = 'git commit -m "Initial commit: Turkish AI Agent - Enterprise platform"'
    if run_command(commit_cmd, "Failed to commit changes"):
        print("✅ Changes committed")
    
    # Set branch to main
    if run_command("git branch -M main", "Failed to rename branch to main"):
        print("✅ Branch renamed to main")
    
    # Push to GitHub
    if run_command("git push -u origin main", "Failed to push to GitHub"):
        print("✅ Code pushed to GitHub successfully")
        return True
    
    return False

def main():
    """Main function to execute the GitHub upload process."""
    print_header("Turkish-LLM Portfolio GitHub Upload")
    
    # Check if GitHub username is set
    if not GITHUB_USERNAME:
        username = input("Enter your GitHub username: ")
        globals()["GITHUB_USERNAME"] = username
    
    # Execute steps
    clean_repository()
    create_gitignore()
    enhance_readme()
    
    if initialize_git():
        if setup_github_repository():
            if commit_and_push():
                print_header("SUCCESS: Your Turkish-LLM project is now on GitHub!")
                print(f"Repository URL: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
                print("\nNext steps for your portfolio:")
                print("1. Add relevant topics to your repository (turkish, ai, nlp, etc.)")
                print("2. Enable GitHub Pages to showcase the project")
                print("3. Link this project in your resume and LinkedIn profile")
                return 0
    
    print("❌ Some steps failed. Please check the output above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())