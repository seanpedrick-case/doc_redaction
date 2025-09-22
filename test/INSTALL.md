# Test Suite Installation Guide

This guide explains how to install the dependencies needed to run the CLI redaction test suite.

## Quick Start

### Option 1: Install test dependencies with pip
```bash
# Install main application dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r test/requirements.txt
```

### Option 2: Install with pyproject.toml
```bash
# Install with test dependencies
pip install -e ".[test]"
```

### Option 3: Install everything at once
```bash
# Install main dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest pytest-cov pytest-html pytest-xdist
```

## Detailed Requirements

### Core Dependencies (Already in your requirements.txt)
The test suite uses your existing application dependencies:
- All the packages in your main `requirements.txt`
- Standard Python libraries (unittest, tempfile, shutil, os, subprocess)

### Additional Test Dependencies

#### Required for Testing:
- **pytest** (>=7.0.0): Modern test framework with better discovery and reporting
- **pytest-cov** (>=4.0.0): Coverage reporting for tests

#### Optional for Enhanced Testing:
- **pytest-html** (>=3.1.0): Generate HTML test reports
- **pytest-xdist** (>=3.0.0): Run tests in parallel for faster execution

## Installation Commands

### Minimal Installation (Required)
```bash
pip install pytest pytest-cov
```

### Full Installation (Recommended)
```bash
pip install pytest pytest-cov pytest-html pytest-xdist
```

### Development Installation
```bash
# Install in development mode with test dependencies
pip install -e ".[test]"
```

## Verification

After installation, verify everything works:

```bash
# Check pytest is installed
pytest --version

# Run a simple test to verify the test suite works
cd test
python test.py
```

## Running Tests

### Method 1: Using the test script (Recommended)
```bash
cd test
python test.py
```

### Method 2: Using pytest
```bash
# Run all tests
pytest test/test.py -v

# Run with coverage
pytest test/test.py --cov=. --cov-report=html

# Run in parallel (faster)
pytest test/test.py -n auto
```

### Method 3: Using unittest directly
```bash
cd test
python -m unittest test.test.TestCLIRedactExamples -v
```

## Troubleshooting

### Common Issues:

1. **Missing example data files**
   - Ensure you have the example data in `example_data/` directory
   - Tests will skip gracefully if files are missing

2. **AWS credentials not configured**
   - AWS-related tests may fail but this is expected
   - Tests are designed to handle missing credentials gracefully

3. **Import errors**
   - Make sure you're in the correct directory
   - Ensure all main application dependencies are installed first

4. **Permission errors**
   - Ensure you have write permissions for temporary directories
   - The test suite creates and cleans up temporary files automatically

### Getting Help:

If you encounter issues:
1. Check that all main application dependencies are installed
2. Verify you're running from the correct directory
3. Ensure example data files are present
4. Check the test output for specific error messages

## Notes

- The test suite is designed to be robust and will skip tests if required files are missing
- All temporary files are automatically cleaned up
- Tests have a 10-minute timeout to prevent hanging
- AWS tests are expected to fail if credentials aren't configured
