# GitHub Actions CI/CD Setup

This directory contains GitHub Actions workflows for automated testing of the CLI redaction application.

## Workflows Overview

### 1. **Simple Test Run** (`.github/workflows/simple-test.yml`)
- **Purpose**: Basic test execution
- **Triggers**: Push to main/dev, Pull requests
- **OS**: Ubuntu Latest
- **Python**: 3.11
- **Features**: 
  - Installs system dependencies
  - Sets up test data
  - Runs CLI tests
  - Runs pytest

### 2. **Comprehensive CI/CD** (`.github/workflows/ci.yml`)
- **Purpose**: Full CI/CD pipeline
- **Features**:
  - Linting (Ruff, Black)
  - Unit tests (Python 3.10, 3.11, 3.12)
  - Integration tests
  - Security scanning (Safety, Bandit)
  - Coverage reporting
  - Package building (on main branch)

### 3. **Multi-OS Testing** (`.github/workflows/multi-os-test.yml`)
- **Purpose**: Cross-platform testing
- **OS**: Ubuntu, macOS (Windows not included currently but may be reintroduced)
- **Python**: 3.10, 3.11, 3.12
- **Features**: Tests compatibility across different operating systems

### 4. **Basic Test Suite** (`.github/workflows/test.yml`)
- **Purpose**: Original test workflow
- **Features**: 
  - Multiple Python versions
  - System dependency installation
  - Test data creation
  - Coverage reporting

## Setup Scripts

### Test Data Setup (`.github/scripts/setup_test_data.py`)
Creates dummy test files when example data is not available:
- PDF documents
- CSV files
- Word documents
- Images
- Allow/deny lists
- OCR output files

## Usage

### Running Tests Locally

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Setup test data
python .github/scripts/setup_test_data.py

# Run tests
cd test
python test.py
```

### GitHub Actions Triggers

1. **Push to main/dev**: Runs all tests
2. **Pull Request**: Runs tests and linting
3. **Daily Schedule**: Runs tests at 2 AM UTC
4. **Manual Trigger**: Can be triggered manually from GitHub

## Configuration

### Environment Variables
- `PYTHON_VERSION`: Default Python version (3.11)
- `PYTHONPATH`: Set automatically for test discovery

### Caching
- Pip dependencies are cached for faster builds
- Cache key based on requirements.txt hash

### Artifacts
- Test results (JUnit XML)
- Coverage reports (HTML, XML)
- Security reports
- Build artifacts (on main branch)

## Test Data

The workflows automatically create test data when example files are missing:

### Required Files Created:
- `example_data/example_of_emails_sent_to_a_professor_before_applying.pdf`
- `example_data/combined_case_notes.csv`
- `example_data/Bold minimalist professional cover letter.docx`
- `example_data/example_complaint_letter.jpg`
- `example_data/test_allow_list_*.csv`
- `example_data/partnership_toolkit_redact_*.csv`
- `example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv`

### Dependencies Installed:
- **System**: tesseract-ocr, poppler-utils, OpenGL libraries
- **Python**: All requirements.txt packages + pytest, reportlab, pillow

## Workflow Status

### Success Criteria:
- ✅ All tests pass
- ✅ No linting errors
- ✅ Security checks pass
- ✅ Coverage meets threshold (if configured)

### Failure Handling:
- Tests are designed to skip gracefully if files are missing
- AWS tests are expected to fail without credentials
- System dependency failures are handled with fallbacks

## Customization

### Adding New Tests:
1. Add test methods to `test/test.py`
2. Update test data in `setup_test_data.py` if needed
3. Tests will automatically run in all workflows

### Modifying Workflows:
1. Edit the appropriate `.yml` file
2. Test locally first
3. Push to trigger the workflow

### Environment-Specific Settings:
- **Ubuntu**: Full system dependencies
- **Windows**: Python packages only
- **macOS**: Homebrew dependencies

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**:
   - Check system dependency installation
   - Verify Python package versions

2. **Test Failures**:
   - Check test data creation
   - Verify file paths
   - Review test output logs

3. **AWS Test Failures**:
   - Expected without credentials
   - Tests are designed to handle this gracefully

4. **System Dependency Issues**:
   - Different OS have different requirements
   - Check the specific OS section in workflows

### Debug Mode:
Add `--verbose` or `-v` flags to pytest commands for more detailed output.

## Security

- Dependencies are scanned with Safety
- Code is scanned with Bandit
- No secrets are exposed in logs
- Test data is temporary and cleaned up

## Performance

- Tests run in parallel where possible
- Dependencies are cached
- Only necessary system packages are installed
- Test data is created efficiently

## Monitoring

- Workflow status is visible in GitHub Actions tab
- Coverage reports are uploaded to Codecov
- Test results are available as artifacts
- Security reports are generated and stored
