@echo off
REM Batch script to run GUI tests with conda environment activated
REM This script activates the 'redaction' conda environment and runs the GUI tests

echo Activating conda environment 'redaction'...
call conda activate redaction

if %errorlevel% neq 0 (
    echo Failed to activate conda environment 'redaction'
    echo Please ensure conda is installed and the 'redaction' environment exists
    pause
    exit /b 1
)

echo Running GUI tests...
python test_gui_only.py

if %errorlevel% neq 0 (
    echo GUI tests failed
    pause
    exit /b 1
) else (
    echo GUI tests passed successfully
)

pause
