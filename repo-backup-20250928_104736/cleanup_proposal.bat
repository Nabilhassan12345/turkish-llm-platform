@echo off
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

:: Cache, venv, and backup directories
if "%DRY_RUN%" == "true" (
    echo Would delete: C:\Users\nh483\Turkish-llm\.git (0.0 GB)
) else (
    echo Deleting: C:\Users\nh483\Turkish-llm\.git (0.0 GB)
    if exist "C:\Users\nh483\Turkish-llm\.git" rmdir /s /q "C:\Users\nh483\Turkish-llm\.git"
)
if "%DRY_RUN%" == "true" (
    echo Would delete: C:\Users\nh483\Turkish-llm\.git\objects (0.0 GB)
) else (
    echo Deleting: C:\Users\nh483\Turkish-llm\.git\objects (0.0 GB)
    if exist "C:\Users\nh483\Turkish-llm\.git\objects" rmdir /s /q "C:\Users\nh483\Turkish-llm\.git\objects"
)
if "%DRY_RUN%" == "true" (
    echo Would delete: C:\Users\nh483\Turkish-llm\.git\objects\pack (0.0 GB)
) else (
    echo Deleting: C:\Users\nh483\Turkish-llm\.git\objects\pack (0.0 GB)
    if exist "C:\Users\nh483\Turkish-llm\.git\objects\pack" rmdir /s /q "C:\Users\nh483\Turkish-llm\.git\objects\pack"
)
if "%DRY_RUN%" == "true" (
    echo Would delete: C:\Users\nh483\Turkish-llm\scripts\__pycache__ (0.0 GB)
) else (
    echo Deleting: C:\Users\nh483\Turkish-llm\scripts\__pycache__ (0.0 GB)
    if exist "C:\Users\nh483\Turkish-llm\scripts\__pycache__" rmdir /s /q "C:\Users\nh483\Turkish-llm\scripts\__pycache__"
)
if "%DRY_RUN%" == "true" (
    echo Would delete: C:\Users\nh483\Turkish-llm\.git\hooks (0.0 GB)
) else (
    echo Deleting: C:\Users\nh483\Turkish-llm\.git\hooks (0.0 GB)
    if exist "C:\Users\nh483\Turkish-llm\.git\hooks" rmdir /s /q "C:\Users\nh483\Turkish-llm\.git\hooks"
)
if "%DRY_RUN%" == "true" (
    echo Would delete: C:\Users\nh483\Turkish-llm\services\__pycache__ (0.0 GB)
) else (
    echo Deleting: C:\Users\nh483\Turkish-llm\services\__pycache__ (0.0 GB)
    if exist "C:\Users\nh483\Turkish-llm\services\__pycache__" rmdir /s /q "C:\Users\nh483\Turkish-llm\services\__pycache__"
)

:: Duplicate files
:: Duplicate group (hash: db4ea1d0...)
:: Keeping: C:\Users\nh483\Turkish-llm\.git\logs\HEAD
if "%DRY_RUN%" == "true" (
    echo Would delete duplicate: C:\Users\nh483\Turkish-llm\.git\logs\refs\heads\main (0.00 MB)
) else (
    echo Deleting duplicate: C:\Users\nh483\Turkish-llm\.git\logs\refs\heads\main (0.00 MB)
    if exist "C:\Users\nh483\Turkish-llm\.git\logs\refs\heads\main" del /f /q "C:\Users\nh483\Turkish-llm\.git\logs\refs\heads\main"
)
if "%DRY_RUN%" == "true" (
    echo Would delete duplicate: C:\Users\nh483\Turkish-llm\.git\logs\refs\remotes\origin\HEAD (0.00 MB)
) else (
    echo Deleting duplicate: C:\Users\nh483\Turkish-llm\.git\logs\refs\remotes\origin\HEAD (0.00 MB)
    if exist "C:\Users\nh483\Turkish-llm\.git\logs\refs\remotes\origin\HEAD" del /f /q "C:\Users\nh483\Turkish-llm\.git\logs\refs\remotes\origin\HEAD"
)
:: Duplicate group (hash: e3b0c442...)
:: Keeping: C:\Users\nh483\Turkish-llm\datasets\merged\finans_merged.jsonl
if "%DRY_RUN%" == "true" (
    echo Would delete duplicate: C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_egitim.jsonl (0.00 MB)
) else (
    echo Deleting duplicate: C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_egitim.jsonl (0.00 MB)
    if exist "C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_egitim.jsonl" del /f /q "C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_egitim.jsonl"
)
if "%DRY_RUN%" == "true" (
    echo Would delete duplicate: C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_saglik.jsonl (0.00 MB)
) else (
    echo Deleting duplicate: C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_saglik.jsonl (0.00 MB)
    if exist "C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_saglik.jsonl" del /f /q "C:\Users\nh483\Turkish-llm\datasets\pilots\pilot_saglik.jsonl"
)
if "%DRY_RUN%" == "true" (
    echo Would delete duplicate: C:\Users\nh483\Turkish-llm\rag_system.py (0.00 MB)
) else (
    echo Deleting duplicate: C:\Users\nh483\Turkish-llm\rag_system.py (0.00 MB)
    if exist "C:\Users\nh483\Turkish-llm\rag_system.py" del /f /q "C:\Users\nh483\Turkish-llm\rag_system.py"
)

echo.
if "%DRY_RUN%" == "true" (
    echo Dry run completed. No files were deleted.
    echo To execute the cleanup, run: cleanup_proposal.bat --confirm
) else (
    echo Cleanup completed.
)
