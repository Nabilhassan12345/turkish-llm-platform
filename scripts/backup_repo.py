#!/usr/bin/env python3
"""
Backup script for Turkish-llm repository
Creates a timestamped backup before making any changes
"""

import os
import sys
import shutil
import datetime
import argparse
from pathlib import Path

def create_backup(repo_path=None, backup_dir=None):
    """Create a timestamped backup of the repository"""
    if repo_path is None:
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if backup_dir is None:
        parent_dir = os.path.dirname(repo_path)
        backup_dir = os.path.join(parent_dir, f"Turkish-llm_backup_{timestamp}")
    
    print(f"Creating backup of repository at: {backup_dir}")
    
    # Exclude large files and unnecessary directories
    exclude = [
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        ".env",
        "venv",
        "env",
        ".venv",
        ".pytest_cache",
        ".coverage",
        "*.pt",
        "*.bin",
        "*.ckpt",
        "*.safetensors"
    ]
    
    def ignore_patterns(path, names):
        ignored = set()
        for pattern in exclude:
            if pattern.startswith("*."):
                ext = pattern[1:]  # Get the extension with the *
                for name in names:
                    if name.endswith(ext):
                        ignored.add(name)
            else:
                if pattern in names:
                    ignored.add(pattern)
        return ignored
    
    try:
        shutil.copytree(repo_path, backup_dir, ignore=ignore_patterns)
        print(f"✅ Backup created successfully at: {backup_dir}")
        return backup_dir
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a backup of the Turkish-llm repository")
    parser.add_argument("--repo-path", help="Path to the repository (default: parent directory of this script)")
    parser.add_argument("--backup-dir", help="Directory to store the backup (default: parent directory of repo)")
    
    args = parser.parse_args()
    
    backup_path = create_backup(args.repo_path, args.backup_dir)
    
    if backup_path:
        print(f"Repository backed up to: {backup_path}")
        sys.exit(0)
    else:
        print("Backup failed")
        sys.exit(1)