#!/usr/bin/env python3
"""
Cleanup analyzer for Turkish-llm repository
Analyzes repository for cleanup opportunities and generates reports
"""

import os
import sys
import json
import hashlib
import shutil
import datetime
import argparse
from pathlib import Path
from collections import defaultdict
import tarfile

# Import the backup script
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from backup_repo import create_backup

def get_file_size(file_path):
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except (FileNotFoundError, PermissionError):
        return 0

def get_dir_size(dir_path):
    """Get directory size in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total_size += get_file_size(file_path)
    return total_size

def bytes_to_gb(bytes_size):
    """Convert bytes to GB with 2 decimal places"""
    return round(bytes_size / (1024 * 1024 * 1024), 2)

def get_file_hash(file_path):
    """Get SHA256 hash of a file"""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except (FileNotFoundError, PermissionError):
        return None

def is_model_weight_file(file_path):
    """Check if file is a model weight file"""
    extensions = ['.bin', '.pt', '.safetensors', '.ckpt', '.pth']
    return any(file_path.endswith(ext) for ext in extensions)

def is_cache_or_venv(path):
    """Check if path is a cache or virtual environment"""
    cache_patterns = [
        'venv', '.venv', 'env', '__pycache__', '.pytest_cache', 
        '.cache', 'wandb', '.git', 'node_modules'
    ]
    path_parts = Path(path).parts
    return any(pattern in path_parts for pattern in cache_patterns)

def is_backup(path):
    """Check if path is a backup"""
    backup_patterns = ['-backup-', '_backup_', '.bak', '.backup']
    return any(pattern in str(path) for pattern in backup_patterns)

def analyze_repository(repo_path):
    """Analyze repository for cleanup opportunities"""
    print(f"Analyzing repository at: {repo_path}")
    
    # Initialize results
    results = {
        "top_directories_by_size": [],
        "top_files_by_size": [],
        "large_files": [],
        "duplicate_files": {},
        "cache_venv_backups": [],
        "model_weight_files": [],
        "total_reclaimable_bytes": 0
    }
    
    # Get all directories
    all_dirs = []
    for dirpath, dirnames, _ in os.walk(repo_path):
        for dirname in dirnames:
            dir_path = os.path.join(dirpath, dirname)
            dir_size = get_dir_size(dir_path)
            all_dirs.append((dir_path, dir_size))
    
    # Sort directories by size
    all_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 20 directories by size
    for dir_path, dir_size in all_dirs[:20]:
        results["top_directories_by_size"].append({
            "path": dir_path,
            "size_bytes": dir_size,
            "size_gb": bytes_to_gb(dir_size),
            "reason": "cache" if is_cache_or_venv(dir_path) else "backup" if is_backup(dir_path) else "data",
            "safe_to_delete": "true" if is_cache_or_venv(dir_path) else "needs-confirm",
            "recommended_action": f"rm -rf {dir_path}" if is_cache_or_venv(dir_path) else f"# Needs review: rm -rf {dir_path}",
            "estimated_reclaim_bytes": dir_size if is_cache_or_venv(dir_path) else 0
        })
        
        # Add to cache/venv/backup list if applicable
        if is_cache_or_venv(dir_path) or is_backup(dir_path):
            results["cache_venv_backups"].append({
                "path": dir_path,
                "size_bytes": dir_size,
                "reason": "cache" if is_cache_or_venv(dir_path) else "backup",
                "safe_to_delete": "true" if is_cache_or_venv(dir_path) else "needs-confirm",
                "recommended_action": f"rm -rf {dir_path}" if is_cache_or_venv(dir_path) else f"# Needs review: rm -rf {dir_path}",
                "estimated_reclaim_bytes": dir_size if is_cache_or_venv(dir_path) else 0
            })
    
    # Get all files
    all_files = []
    file_hashes = defaultdict(list)
    
    for dirpath, _, filenames in os.walk(repo_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                file_size = get_file_size(file_path)
                all_files.append((file_path, file_size))
                
                # Check for large files
                if file_size > 100 * 1024 * 1024:  # > 100MB
                    is_model = is_model_weight_file(file_path)
                    results["large_files"].append({
                        "path": file_path,
                        "size_bytes": file_size,
                        "size_gb": bytes_to_gb(file_size),
                        "reason": "model_weights" if is_model else "large_file",
                        "safe_to_delete": "needs-confirm",
                        "recommended_action": f"# Needs review: rm {file_path}",
                        "estimated_reclaim_bytes": 0  # Don't count as reclaimable
                    })
                    
                    # Add to model weights list if applicable
                    if is_model:
                        results["model_weight_files"].append({
                            "path": file_path,
                            "size_bytes": file_size,
                            "size_gb": bytes_to_gb(file_size)
                        })
                
                # Calculate hash for duplicate detection (skip very large files)
                if file_size < 100 * 1024 * 1024:  # < 100MB
                    file_hash = get_file_hash(file_path)
                    if file_hash:
                        file_hashes[file_hash].append((file_path, file_size))
    
    # Sort files by size
    all_files.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 50 files by size
    for file_path, file_size in all_files[:50]:
        is_model = is_model_weight_file(file_path)
        results["top_files_by_size"].append({
            "path": file_path,
            "size_bytes": file_size,
            "size_gb": bytes_to_gb(file_size),
            "reason": "model_weights" if is_model else "large_file",
            "safe_to_delete": "needs-confirm",
            "recommended_action": f"# Needs review: rm {file_path}",
            "estimated_reclaim_bytes": 0  # Don't count as reclaimable
        })
    
    # Find duplicate files
    duplicate_groups = []
    total_duplicate_bytes = 0
    
    for file_hash, files in file_hashes.items():
        if len(files) > 1:
            # Sort duplicates by path to keep the first one
            files.sort(key=lambda x: x[0])
            
            # Calculate reclaimable bytes (all but the first file)
            reclaimable_bytes = sum(file_size for _, file_size in files[1:])
            total_duplicate_bytes += reclaimable_bytes
            
            duplicate_group = {
                "hash": file_hash,
                "files": [],
                "reclaimable_bytes": reclaimable_bytes
            }
            
            # First file is kept
            duplicate_group["files"].append({
                "path": files[0][0],
                "size_bytes": files[0][1],
                "action": "keep"
            })
            
            # Rest are duplicates
            for file_path, file_size in files[1:]:
                is_model = is_model_weight_file(file_path)
                duplicate_group["files"].append({
                    "path": file_path,
                    "size_bytes": file_size,
                    "reason": "duplicate" + ("_model" if is_model else ""),
                    "safe_to_delete": "false" if is_model else "true",
                    "recommended_action": f"{'# Needs review: ' if is_model else ''}rm {file_path}",
                    "estimated_reclaim_bytes": 0 if is_model else file_size
                })
            
            duplicate_groups.append(duplicate_group)
    
    # Sort duplicate groups by reclaimable bytes
    duplicate_groups.sort(key=lambda x: x["reclaimable_bytes"], reverse=True)
    results["duplicate_files"] = duplicate_groups
    
    # Calculate total reclaimable bytes
    cache_venv_bytes = sum(item["estimated_reclaim_bytes"] for item in results["cache_venv_backups"])
    duplicate_bytes = sum(group["reclaimable_bytes"] for group in duplicate_groups)
    results["total_reclaimable_bytes"] = cache_venv_bytes + duplicate_bytes
    
    return results

def check_for_secrets(repo_path):
    """Check for potential secrets in the repository"""
    secret_patterns = [
        "api_key", "apikey", "secret", "password", "token", "credential", "auth"
    ]
    
    secrets_found = []
    
    for dirpath, _, filenames in os.walk(repo_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                try:
                    # Skip binary files and very large files
                    if get_file_size(file_path) > 10 * 1024 * 1024:  # > 10MB
                        continue
                        
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        
                        for pattern in secret_patterns:
                            if pattern.lower() in content.lower():
                                # Check if it's in a comment or test file
                                if "test" in file_path.lower() or "example" in file_path.lower():
                                    continue
                                    
                                # Check if it's a false positive (common in CI files)
                                if "ci.yml" in file_path.lower() and "check for secrets" in content.lower():
                                    continue
                                    
                                secrets_found.append({
                                    "file": file_path,
                                    "pattern": pattern,
                                    "line_number": None  # Would need more complex parsing
                                })
                except (UnicodeDecodeError, PermissionError):
                    continue
    
    return secrets_found

def generate_windows_cleanup_script(results, repo_path):
    """Generate Windows batch cleanup script"""
    script_content = """@echo off
:: Cleanup script for Turkish-llm repository
:: Usage:
::   cleanup_proposal.bat --dry-run   # Print actions without executing
::   cleanup_proposal.bat --confirm   # Execute actions marked as safe_to_delete=true

:: Check arguments
if "%1" NEQ "--dry-run" if "%1" NEQ "--confirm" (
    echo Usage: cleanup_proposal.bat [--dry-run^|--confirm]
    exit /b 1
)

set DRY_RUN=true
if "%1" == "--confirm" (
    set DRY_RUN=false
    echo WARNING: This will delete files marked as safe_to_delete=true
    echo Press Ctrl+C to cancel or Enter to continue...
    pause > nul
)

echo Starting cleanup process...
echo.

"""

    # Add cache/venv/backup cleanup
    script_content += ":: Cache, venv, and backup directories\n"
    for item in results["cache_venv_backups"]:
        if item["safe_to_delete"] == "true":
            script_content += f"""if "%DRY_RUN%" == "true" (
    echo Would delete: {item['path']} ({bytes_to_gb(item['size_bytes'])} GB)
) else (
    echo Deleting: {item['path']} ({bytes_to_gb(item['size_bytes'])} GB)
    if exist "{item['path']}" rmdir /s /q "{item['path']}"
)
"""
        else:
            script_content += f"echo Needs confirmation: {item['path']} ({bytes_to_gb(item['size_bytes'])} GB) - {item['reason']}\n"
    
    script_content += "\n:: Duplicate files\n"
    for group in results["duplicate_files"]:
        script_content += f":: Duplicate group (hash: {group['hash'][:8]}...)\n"
        for file in group["files"]:
            if "action" in file and file["action"] == "keep":
                script_content += f":: Keeping: {file['path']}\n"
            elif "safe_to_delete" in file and file["safe_to_delete"] == "true":
                script_content += f"""if "%DRY_RUN%" == "true" (
    echo Would delete duplicate: {file['path']} ({file['size_bytes'] / 1024 / 1024:.2f} MB)
) else (
    echo Deleting duplicate: {file['path']} ({file['size_bytes'] / 1024 / 1024:.2f} MB)
    if exist "{file['path']}" del /f /q "{file['path']}"
)
"""
            else:
                script_content += f"echo Needs confirmation: {file['path']} ({file['size_bytes'] / 1024 / 1024:.2f} MB) - duplicate file\n"
    
    script_content += """
echo.
if "%DRY_RUN%" == "true" (
    echo Dry run completed. No files were deleted.
    echo To execute the cleanup, run: cleanup_proposal.bat --confirm
) else (
    echo Cleanup completed.
)
"""
    
    # Write script to file
    script_path = os.path.join(repo_path, "cleanup_proposal.bat")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return script_path

def main():
    """Main function to run the cleanup analyzer"""
    parser = argparse.ArgumentParser(description="Analyze repository for cleanup opportunities")
    parser.add_argument("--repo-path", help="Path to the repository (default: current directory)")
    parser.add_argument("--windows", action="store_true", help="Generate Windows batch script instead of bash")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup")
    
    args = parser.parse_args()
    
    # Set repository path
    repo_path = args.repo_path if args.repo_path else os.getcwd()
    
    # Create backup
    backup_path = None
    if not args.no_backup:
        print("Creating backup before analysis...")
        backup_path = create_backup(repo_path)
        if not backup_path:
            print("⚠️ Warning: Backup failed. Continuing without backup.")
    
    # Analyze repository
    results = analyze_repository(repo_path)
    
    # Check for secrets
    secrets = check_for_secrets(repo_path)
    
    # Generate cleanup script
    if args.windows:
        script_path = generate_windows_cleanup_script(results, repo_path)
        print(f"Windows cleanup script generated: {script_path}")
    else:
        script_path = generate_cleanup_script(results, repo_path)
        print(f"Bash cleanup script generated: {script_path}")
    
    # Generate summary
    summary_path = os.path.join(repo_path, "CLEANUP_SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write(generate_cleanup_summary(results, backup_path, repo_path))
    
    # Save results as JSON
    results_path = os.path.join(repo_path, "CLEANUP_REPORT.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Cleanup summary generated: {summary_path}")
    print(f"Cleanup report generated: {results_path}")
    
    # Print results
    total_reclaimable_gb = bytes_to_gb(results["total_reclaimable_bytes"])
    print(f"\n🧹 Cleanup Analysis Results:")
    print(f"- Total reclaimable space: {total_reclaimable_gb} GB")
    print(f"- Cache/venv directories: {len(results['cache_venv_backups'])}")
    print(f"- Duplicate file groups: {len(results['duplicate_files'])}")
    print(f"- Large files: {len(results['large_files'])}")
    
    if args.windows:
        print("\nTo run the cleanup script:")
        print("  cleanup_proposal.bat --dry-run   # Preview changes")
        print("  cleanup_proposal.bat --confirm   # Execute cleanup")
    else:
        print("\nTo run the cleanup script:")
        print("  ./cleanup_proposal.sh --dry-run   # Preview changes")
        print("  ./cleanup_proposal.sh --confirm   # Execute cleanup")

if __name__ == "__main__":
    main()