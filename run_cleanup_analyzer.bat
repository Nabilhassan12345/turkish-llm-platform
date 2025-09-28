@echo off
echo Running cleanup analyzer to generate Windows-compatible cleanup script...
python cleanup_analyzer.py --windows
echo.
echo If the analysis completed successfully, you can run:
echo   cleanup_proposal.bat --dry-run   (to preview changes)
echo   cleanup_proposal.bat --confirm   (to execute cleanup)
pause