# GUI Testing for Document Redaction App

This directory contains tests specifically for verifying that the GUI application (`app.py`) loads correctly.

## Test Files

### `test_gui_only.py`
A standalone script that tests only the GUI functionality. This is useful for:
- Quick verification that the Gradio interface loads without errors
- CI/CD pipelines where you want to test GUI separately from CLI functionality
- Development testing when you only want to check GUI components

**Usage:**

Option 1 - Manual activation:
```bash
conda activate redaction
cd test
python test_gui_only.py
```

Option 2 - Using helper scripts (Windows):
```bash
cd test
# For Command Prompt:
run_gui_test.bat

# For PowerShell:
.\run_gui_test.ps1
```

### `test.py` (Updated)
The main test suite now includes both CLI and GUI tests. The GUI tests are in the `TestGUIApp` class.

**Usage:**

Option 1 - Manual activation:
```bash
conda activate redaction
cd test
python test.py
```

Option 2 - Using helper scripts (Windows):
```bash
cd test
# For Command Prompt:
run_gui_test.bat

# For PowerShell:
.\run_gui_test.ps1
```

## What the GUI Tests Check

1. **App Import and Initialization** (`test_app_import_and_initialization`)
   - Verifies that `app.py` can be imported without errors
   - Checks that the Gradio `app` object is created successfully
   - Ensures the app is a proper Gradio Blocks instance

2. **App Launch in Headless Mode** (`test_app_launch_headless`)
   - Tests that the app can be launched without opening a browser
   - Verifies the Gradio server starts successfully
   - Uses threading to prevent blocking the test execution

3. **Configuration Loading** (`test_app_configuration_loading`)
   - Verifies that configuration variables are loaded correctly
   - Checks key settings like server port, file size limits, language settings
   - Ensures the app has access to all required configuration

## Test Requirements

- **Conda environment 'redaction' must be activated** before running tests
- Python environment with all dependencies installed
- Access to the `tools.config` module
- Gradio and related GUI dependencies (including `gradio_image_annotation`)
- The `app.py` file in the parent directory

### Prerequisites

Before running the GUI tests, ensure you have activated the conda environment:

```bash
conda activate redaction
```

The `gradio_image_annotation` package is already installed in the 'redaction' environment.

## Expected Behavior

- All tests should pass if the GUI loads correctly
- Tests will fail if there are import errors, missing dependencies, or configuration issues
- The headless launch test may take up to 10 seconds to complete

## Troubleshooting

If tests fail:
1. Check that all dependencies are installed (`pip install -r requirements.txt`)
2. Verify that `app.py` exists in the parent directory
3. Ensure configuration files are properly set up
4. Check for any missing environment variables or configuration issues

## Integration with CI/CD

These tests are designed to run in headless environments and are suitable for:
- GitHub Actions
- Jenkins pipelines
- Docker containers
- Any automated testing environment

The tests do not require a display or browser to be available.
