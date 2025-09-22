# GitHub Actions Integration Guide

This guide explains how to use your test suite with GitHub Actions for automated CI/CD.

## 🚀 Quick Start

### 1. **Choose Your Workflow**

I've created multiple workflow options for you:

#### **Option A: Simple Test Run** (Recommended for beginners)
```yaml
# File: .github/workflows/simple-test.yml
# - Basic test execution
# - Ubuntu Latest
# - Python 3.11
# - Minimal setup
```

#### **Option B: Comprehensive CI/CD** (Recommended for production)
```yaml
# File: .github/workflows/ci.yml
# - Full pipeline with linting, security, coverage
# - Multiple Python versions
# - Integration tests
# - Package building
```

#### **Option C: Multi-OS Testing** (For cross-platform compatibility)
```yaml
# File: .github/workflows/multi-os-test.yml
# - Tests on Ubuntu, Windows, macOS
# - Multiple Python versions
# - Cross-platform compatibility
```

### 2. **Enable GitHub Actions**

1. **Push your code to GitHub**
2. **Go to your repository → Actions tab**
3. **Select a workflow and click "Run workflow"**
4. **Watch the tests run automatically!**

## 📋 What Each Workflow Does

### **Simple Test Run** (`.github/workflows/simple-test.yml`)
```yaml
✅ Installs system dependencies (tesseract, poppler, OpenGL)
✅ Installs Python dependencies from requirements.txt
✅ Downloads spaCy model
✅ Creates dummy test data automatically
✅ Runs your CLI tests
✅ Runs pytest with coverage
```

### **Comprehensive CI/CD** (`.github/workflows/ci.yml`)
```yaml
✅ Linting (Ruff, Black)
✅ Unit tests (Python 3.10, 3.11, 3.12)
✅ Integration tests
✅ Security scanning (Safety, Bandit)
✅ Coverage reporting
✅ Package building (on main branch)
✅ Artifact uploads
```

### **Multi-OS Testing** (`.github/workflows/multi-os-test.yml`)
```yaml
✅ Tests on Ubuntu, Windows, macOS
✅ Python 3.10, 3.11, 3.12
✅ Cross-platform compatibility
✅ OS-specific dependency handling
```

## 🔧 How It Works

### **Automatic Test Data Creation**
The workflows automatically create dummy test files when your example data is missing:

```python
# .github/scripts/setup_test_data.py creates:
- example_data/example_of_emails_sent_to_a_professor_before_applying.pdf
- example_data/combined_case_notes.csv
- example_data/Bold minimalist professional cover letter.docx
- example_data/example_complaint_letter.jpg
- example_data/test_allow_list_*.csv
- example_data/partnership_toolkit_redact_*.csv
- example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv
```

### **System Dependencies**
Each OS gets the right dependencies:

**Ubuntu:**
```bash
sudo apt-get install tesseract-ocr poppler-utils libgl1-mesa-glx
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
```bash
# Handled by Python packages
```

### **Python Dependencies**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov reportlab pillow
```

## 🎯 Triggers

### **When Tests Run:**
- ✅ **Push to main/dev branches**
- ✅ **Pull requests to main/dev**
- ✅ **Daily at 2 AM UTC** (scheduled)
- ✅ **Manual trigger** from GitHub UI

### **What Happens:**
1. **Checkout code**
2. **Install dependencies**
3. **Create test data**
4. **Run tests**
5. **Generate reports**
6. **Upload artifacts**

## 📊 Test Results

### **Success Criteria:**
- ✅ All tests pass
- ✅ No linting errors
- ✅ Security checks pass
- ✅ Coverage reports generated

### **Failure Handling:**
- ✅ Tests skip gracefully if files missing
- ✅ AWS tests expected to fail without credentials
- ✅ System dependency failures handled with fallbacks

## 🔍 Monitoring

### **GitHub Actions Tab:**
- View workflow runs
- See test results
- Download artifacts
- View logs

### **Artifacts Generated:**
- `test-results.xml` - JUnit test results
- `coverage.xml` - Coverage data
- `htmlcov/` - HTML coverage report
- `bandit-report.json` - Security scan results

### **Coverage Reports:**
- Uploaded to Codecov automatically
- Available in GitHub Actions artifacts
- HTML reports for detailed analysis

## 🛠️ Customization

### **Adding New Tests:**
1. Add test methods to `test/test.py`
2. Update `setup_test_data.py` if needed
3. Tests run automatically in all workflows

### **Modifying Workflows:**
1. Edit the `.yml` file
2. Test locally first
3. Push to trigger workflow

### **Environment Variables:**
```yaml
env:
  PYTHON_VERSION: "3.11"
  # Add your custom variables here
```

## 🚨 Troubleshooting

### **Common Issues:**

1. **"Example file not found"**
   - ✅ **Solution**: Test data is created automatically
   - ✅ **Check**: `setup_test_data.py` runs in workflow

2. **"AWS credentials not configured"**
   - ✅ **Expected**: AWS tests fail without credentials
   - ✅ **Solution**: Tests are designed to handle this

3. **"System dependency error"**
   - ✅ **Check**: OS-specific installation commands
   - ✅ **Solution**: Dependencies are installed automatically

4. **"Test timeout"**
   - ✅ **Default**: 10-minute timeout per test
   - ✅ **Solution**: Tests are designed to be fast

### **Debug Mode:**
Add `--verbose` to pytest commands for detailed output:
```yaml
pytest test/test.py -v --tb=short
```

## 📈 Performance

### **Optimizations:**
- ✅ **Parallel execution** where possible
- ✅ **Dependency caching** for faster builds
- ✅ **Minimal system packages** installed
- ✅ **Efficient test data creation**

### **Build Times:**
- **Simple Test**: ~5-10 minutes
- **Comprehensive CI**: ~15-20 minutes
- **Multi-OS**: ~20-30 minutes

## 🔒 Security

### **Security Features:**
- ✅ **Dependency scanning** with Safety
- ✅ **Code scanning** with Bandit
- ✅ **No secrets exposed** in logs
- ✅ **Temporary test data** cleaned up

### **Secrets Management:**
- Use GitHub Secrets for sensitive data
- Never hardcode credentials in workflows
- Test data is dummy data only

## 🎉 Success!

Once set up, your GitHub Actions will:

1. **Automatically test** every push and PR
2. **Generate reports** and coverage data
3. **Catch issues** before they reach production
4. **Ensure compatibility** across platforms
5. **Provide confidence** in your code quality

## 📚 Next Steps

1. **Choose a workflow** that fits your needs
2. **Push to GitHub** to trigger the first run
3. **Monitor the Actions tab** for results
4. **Customize** as needed for your project
5. **Enjoy** automated testing! 🎉

---

**Need help?** Check the workflow logs in the GitHub Actions tab for detailed error messages and troubleshooting information.
