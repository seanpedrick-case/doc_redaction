# PowerShell script to run GUI tests with conda environment activated
# This script activates the 'redaction' conda environment and runs the GUI tests

Write-Host "Activating conda environment 'redaction'..." -ForegroundColor Green

try {
    # Try to activate the conda environment
    conda activate redaction
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to activate conda environment 'redaction'" -ForegroundColor Red
        Write-Host "Please ensure conda is installed and the 'redaction' environment exists" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host "Running GUI tests..." -ForegroundColor Green
    python test_gui_only.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GUI tests failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    } else {
        Write-Host "GUI tests passed successfully" -ForegroundColor Green
    }
    
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit"
