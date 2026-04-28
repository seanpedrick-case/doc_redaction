---
title: Document redaction
emoji: 📝
colorFrom: blue
colorTo: yellow
sdk: docker
app_file: app.py
pinned: true
license: agpl-3.0
short_description: OCR / redact PDF documents and tabular data
---
# Document redaction (doc_redaction)

<a href="https://pypi.org/project/doc-redaction/" target="_blank"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/doc-redaction"></a> 

Redact personally identifiable information (PII) from documents (PDF, PNG, JPG), Word files (DOCX), or tabular data (XLSX/CSV/Parquet). Please see the [User Guide](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html) for a full walkthrough of all the features in the app.

---

## 🚀 Quick Start - Installation and first run

Follow these instructions to get the document redaction application running on your local machine.

### 1. Package installation

#### Option 1 - Recommended: Install from source repo

Clone the repository and install in editable mode:

```bash
git clone https://github.com/seanpedrick-case/doc_redaction.git
cd doc_redaction
pip install -e .
```

##### Install extras (Paddle or Transformers/Torch VLM)

To install with PaddleOCR:

```bash
pip install -e ".[paddle]"
```

Note that the versions of both PaddleOCR and Torch installed by default are the CPU-only versions. If you want to install the equivalent GPU versions, you will need to run the following commands:
```bash
pip install paddlepaddle-gpu==3.2.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

If you want to run VLMs / LLMs with the transformers package:

```bash
pip install -e ".[vlm]"
```


**Note:** It is difficult to get paddlepaddle gpu working in an environment alongside torch. You may well need to reinstall the cpu version to ensure compatibility, and run paddlepaddle-gpu in a separate environment without torch installed. If you get errors related to .dll files following paddle gpu install, you may need to install the latest c++ redistributables. For Windows, you can find them [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install torchvision --index-url https://download.pytorch.org/whl/cu129
```

#### Option 2 - Install from PyPI

Create a virtual environment (recommended) and install **doc_redaction**.

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

The package is published on PyPI as **`doc-redaction`** (import name **`doc_redaction`**):

```bash
pip install doc_redaction
```

Optional extras (same as in `pyproject.toml`). For installing paddleOCR:

```bash
pip install "doc_redaction[paddle]"
```

For running VLMs / LLMs with the transformers package:

```bash
pip install "doc_redaction[vlm]"
```

For programmatic use (CLI-first API matching Gradio `api_name` routes), see **[Python Package usage (Python)](https://seanpedrick-case.github.io/doc_redaction/src/python_package_usage.html)**. The console script **`cli_redact`** is available after install.

**Web UI from a PyPI install:** You *can* start the Gradio UI after `pip install doc_redaction` by running (note that the prerequisites tesseract and poppler will need to be correctly installed following step 2 below):

```bash
python -m app
```

**Important: your working directory matters.** When you run `python -m app`, the app treats your *current folder* as the “app folder”:

- It will look for configuration at `config/app_config.env` *relative to the folder you run it from* (and `python -m doc_redaction.install_deps` will also write `config/app_config.env` there).
- It may create new folders in that location (for example `config/`, `output/`, `input/`, `logs/`, `usage/`, `feedback/`, and temporary/cache folders depending on your settings).
- The UI example files and bundled assets are packaged with the PyPI install (they live inside the installed `doc_redaction` package). If you run from a “random” directory after a PyPI install, the app can still locate its packaged examples; your working directory mainly affects where `config/`, `input/`, `output/`, logs, and temp folders are created.

In practice, the **smoothest UI experience** (examples, bundled assets, docs links, predictable relative paths) is still usually via a **repository checkout** or **Docker**, but PyPI install is sufficient to launch the UI as long as you run it from a suitable working folder and have the system dependencies available (or run `python -m doc_redaction.install_deps` first).

#### Option 3 - Docker installation

The doc_redaction Redaction app can be installed by using the [Dockerfile](https://github.com/seanpedrick-case/doc_redaction/blob/main/Dockerfile) or Docker compose files ([llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://docs.vllm.ai/en/stable/)) provided in the repo.

##### With Llama.cpp / vLLM inference server

The project now has Docker and Docker compose files available to pair running the Redaction app with local inference servers powered by [llama.cpp](https://github.com/ggml-org/llama.cpp), or [vLLM](https://docs.vllm.ai/en/stable/). Llama.cpp is more flexible than vLLM for low VRAM systems, as Llama.cpp will offload to cpu/system RAM automatically rather than failing as vLLM tends to do.

For Llama.cpp, you can use the [docker-compose_llama.yml](https://github.com/seanpedrick-case/doc_redaction/blob/main/docker-compose_llama.yml) file, and for vLLM, you can use the [docker-compose_vllm.yml](https://github.com/seanpedrick-case/doc_redaction/blob/main/docker-compose_vllm.yml) file. To run, Docker / Docker Desktop should be installed, and then you can run the commands suggested in the top of the files to run the servers.

You will need ~40 GB of disk space to run everything depending on the model chosen from the compose file. For the vLLM server, you will need 24 GB VRAM. For the Llama.cpp server, 24 GB VRAM is needed to run at full speed, but the n-gpu-layers and n-cpu-moe parameters in the Docker compose file can be adjusted to fit into your system. I would suggest that 8 GB VRAM is needed as a bare minimum for decent inference speed. See the [Unsloth guide](https://unsloth.ai/docs/models/qwen3.5) for more details on working with GGUF files for Qwen 3.5.

##### Without Llama.cpp / vLLM inference server

If you want a working Docker installation without GPU support, you can install from the [Dockerfile](https://github.com/seanpedrick-case/doc_redaction/blob/main/Dockerfile) in the repo. A working example of this, with the CPU version of PaddleOCR, can be found on [Hugging Face](https://huggingface.co/spaces/seanpedrickcase/document_redaction). You can adjust the INSTALL_PADDLEOCR, PADDLE_GPU_ENABLED, INSTALL_VLM, and TORCH_GPU_ENABLED config variables to adjust for PaddleOCR and Transformers packages for local VLM support. Note that GPU-enabled PaddleOCR, and GPU-enabled Transformers/Torch often don't work well together, which is one reason why a Llama.cpp/vLLM inference server Docker installation option is provided below.

### 2. Install prerequisites: Tesseract and Poppler

This application relies on two external tools for OCR (Tesseract) and PDF processing (Poppler). Please install them on your system before proceeding. To run the Document Redaction app successfully, these tools need to be installed and either 1. added to PATH, or 2. be in a folder that is directly referenced in the config/app_config.env file with the variables TESSERACT_FOLDER and POPPLER_FOLDER (defined [here](https://github.com/seanpedrick-case/doc_redaction/blob/main/tools/config.py) if you want to see the code). The instructions below will guide you through diffferent ways to install these dependencies.

---

#### Automated dependency setup (recommended)

If you **don’t have admin rights** (or you just want the simplest setup), you can have the project download and configure **Tesseract** and **Poppler** into a local `redaction_deps/` folder inside the doc_redaction folder.

You need the installer script available first, which means either:

- **Repository checkout**: `git clone ...` and run the command from the repo root (recommended for the web UI), or
- **PyPI install**: `pip install doc_redaction` and run from a writable folder where you want `redaction_deps/` and `config/app_config.env` to be created/updated.

From the repository root (or your chosen working folder) after creating/activating your venv and installing Python requirements:

```bash
python -m doc_redaction.install_deps
```

This writes `TESSERACT_FOLDER` / `POPPLER_FOLDER` into `config/app_config.env` so the app can find the binaries without you editing your system PATH.

To just check whether your machine can already see the tools:

```bash
python -m doc_redaction.install_deps --verify-only
```

#### **On Windows**

If you don’t use the automated setup above, you can install the dependencies manually by downloading installers and adding the programs to your system's PATH.

1.  **Install Tesseract OCR:**
    *   Download the installer from the official Tesseract at [UB Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki) (e.g., `tesseract-ocr-w64-setup-v5.X.X...exe`).
    *   Run the installer.
    *   **IMPORTANT:** During installation, ensure you select the option to "Add Tesseract to system PATH for all users" or a similar option. This is crucial for the application to find the Tesseract executable.


2.  **Install Poppler:**
    *   Download the latest Poppler binary for Windows. A common source is the [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows) GitHub releases page. Download the `.zip` file (e.g., `poppler-25.07.0-win.zip`).
    *   Extract the contents of the zip file to a permanent location on your computer, for example, `C:\Program Files\poppler\`.
    *   You must add the `bin` folder from your Poppler installation to your system's PATH environment variable.
        *   Search for "Edit the system environment variables" in the Windows Start Menu and open it.
        *   Click the "Environment Variables..." button.
        *   In the "System variables" section, find and select the `Path` variable, then click "Edit...".
        *   Click "New" and add the full path to the `bin` directory inside your Poppler folder (e.g., `C:\Program Files\poppler\poppler-24.02.0\bin`).
        *   Click OK on all windows to save the changes.

    To verify, open a new Command Prompt and run `tesseract --version` and `pdftoppm -v`. If they both return version information, you have successfully installed the prerequisites.
---

#### **On Linux (Debian/Ubuntu)**

Open your terminal and run the following command to install Tesseract and Poppler:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
```

#### **On Linux (Fedora/CentOS/RHEL)**

Open your terminal and use the `dnf` or `yum` package manager:

```bash
sudo dnf install -y tesseract poppler-utils
```
---

### 3. Run the Application

With all dependencies installed, you can now start the Gradio application GUI. For a guide on how to use this, please go [here](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html).

```bash
python app.py
```

After running the command, the application will start, and you will see a local URL in your terminal (usually `http://127.0.0.1:7860`).

Open this URL in your web browser to use the document redaction tool

#### Command line interface

For example CLI commands, please refer to [this guide](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html#command-line-interface-cli) or the examples in [cli_redact.py](https://github.com/seanpedrick-case/doc_redaction/blob/main/cli_redact.py#L321)

If you installed from **PyPI**, use the installed console script:

```bash
cli_redact --help
```

From a **repository checkout**, you can also run:

```bash
python cli_redact.py --help
```

#### Python package commands

For Python examples in using the Python package, please see [Python Package usage (Python)](https://seanpedrick-case.github.io/doc_redaction/src/python_package_usage.html).

---


### 4. ⚙️ Configuration (Optional)

You can customise the application's behavior by creating a configuration file. This allows you to change settings without modifying the source code, such as enabling AWS features, changing logging behavior, or pointing to local Tesseract/Poppler installations. A full overview of all the potential settings you can modify in the app_config.env file can be seen in tools/config.py, with explanation on the documentation website for [the github repo](https://seanpedrick-case.github.io/doc_redaction/)

To get started:
1.  Locate the `example_config.env` file in the root of the project.
2.  Create a new file named `app_config.env` inside the `config/` directory (i.e., `config/app_config.env`).
3.  Copy the contents from `example_config.env` into your new `config/app_config.env` file.
4.  Modify the values in `config/app_config.env` to suit your needs. The application will automatically load these settings on startup.

If you do not create this file, the application will run with default settings.

#### Configuration Breakdown

Here is an overview of the most important settings, separated by whether they are for local use or require AWS.

---

#### **Local & General Settings (No AWS Required)**

These settings are useful for all users, regardless of whether you are using AWS.

*   `TESSERACT_FOLDER` / `POPPLER_FOLDER`
    *   Use these if you installed Tesseract or Poppler to a custom location on **Windows** and did not add them to the system PATH.
    *   Provide the path to the respective installation folders (for Poppler, point to the `bin` sub-directory).
    *   **Examples:** `POPPLER_FOLDER=C:/Program Files/poppler-24.02.0/bin/` `TESSERACT_FOLDER=tesseract/`

*   `TESSERACT_DATA_FOLDER`
    *   If Tesseract runs but you see an error like `Error opening data file ./eng.traineddata` or `Tesseract couldn't load any languages`, this is usually because it can't find the `tessdata/` language files.
    *   Set this to the folder that contains `eng.traineddata` (typically a `tessdata` directory).
    *   **Examples (Windows):** `TESSERACT_DATA_FOLDER=C:/Program Files/Tesseract-OCR/tessdata`

*   `SHOW_LANGUAGE_SELECTION=True`
    *   Set to `True` to display a language selection dropdown in the UI for OCR processing.

*   `DEFAULT_LOCAL_OCR_MODEL=tesseract`"
    *   Choose the backend for local OCR. Options are `tesseract`, `paddle`, or `hybrid`. "Tesseract" is the default, and is recommended. "hybrid-paddle" is a combination of the two - first pass through the redactions will be done with Tesseract, and then a second pass will be done with PaddleOCR on words with low confidence. "paddle" will only return whole line text extraction, and so will only work for OCR, not redaction. 

*   `SESSION_OUTPUT_FOLDER=False`
    *   If `True`, redacted files will be saved in unique subfolders within the `output/` directory for each session.

*   `DISPLAY_FILE_NAMES_IN_LOGS=False`
    *   For privacy, file names are not recorded in usage logs by default. Set to `True` to include them.

---

#### **AWS-Specific Settings**

These settings are only relevant if you intend to use AWS services like Textract for OCR and Comprehend for PII detection.

*   `RUN_AWS_FUNCTIONS=True`
    *   **This is the master switch.** You must set this to `True` to enable any AWS functionality. If it is `False`, all other AWS settings will be ignored.

*   **UI Options:**
    *   `SHOW_AWS_TEXT_EXTRACTION_OPTIONS=True`: Adds "AWS Textract" as an option in the text extraction dropdown.
    *   `SHOW_AWS_PII_DETECTION_OPTIONS=True`: Adds "AWS Comprehend" as an option in the PII detection dropdown.

*   **Core AWS Configuration:**
    *   `AWS_REGION=example-region`: Set your AWS region (e.g., `us-east-1`).
    *   `DOCUMENT_REDACTION_BUCKET=example-bucket`: The name of the S3 bucket the application will use for temporary file storage and processing.

*   **AWS Logging:**
    *   `SAVE_LOGS_TO_DYNAMODB=True`: If enabled, usage and feedback logs will be saved to DynamoDB tables.
    *   `ACCESS_LOG_DYNAMODB_TABLE_NAME`, `USAGE_LOG_DYNAMODB_TABLE_NAME`, etc.: Specify the names of your DynamoDB tables for logging.

*   **Advanced AWS Textract Features:**
    *   `SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS=True`: Enables UI components for large-scale, asynchronous document processing via Textract.
    *   `TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET=example-bucket-output`: A separate S3 bucket for the final output of asynchronous Textract jobs.
    *   `LOAD_PREVIOUS_TEXTRACT_JOBS_S3=True`: If enabled, the app will try to load the status of previously submitted asynchronous jobs from S3.

*   **Cost Tracking (for internal accounting):**
    *   `SHOW_COSTS=True`: Displays an estimated cost for AWS operations. Can be enabled even if AWS functions are off.
    *   `GET_COST_CODES=True`: Enables a dropdown for users to select a cost code before running a job.
    *   `COST_CODES_PATH=config/cost_codes.csv`: The local path to a CSV file containing your cost codes.
    *   `ENFORCE_COST_CODES=True`: Makes selecting a cost code mandatory before starting a redaction.

Now you have the app installed, please refer to the [User Guide](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html) for more information on how to use it for basic and advanced redaction.

## For agents (API quickstart)

If you are an LLM/agent interacting with this app over HTTP (e.g. Hugging Face Spaces), **do not guess inputs** from the UI. Use the Gradio schema as the source of truth:

- **Discover schema**: `GET /gradio_api/info`
- **Upload files**: `POST /gradio_api/upload` (multipart field `files`) → returns server-internal paths like `/tmp/gradio_tmp/...`
- **Call**: `POST /gradio_api/call/{api_name}` with body `{"data":[...]}` (argument order must match `/gradio_api/info`)
- **Poll**: `GET /gradio_api/call/{api_name}/{event_id}` until complete
- **Download outputs**: `GET /gradio_api/file={path}` (note: some deployments return 403 without session cookies)

### Choose the correct route (prefer short `gr.api` endpoints)

Fetch `/gradio_api/info` and then prefer the simplest route that exists:

- **Apply edited review CSV to a PDF**: `/review_apply`
- **Redact a PDF/image document**: `/doc_redact`
- **Summarise a PDF**: `/pdf_summarise`
- **Redact tabular files (CSV/XLSX/Parquet/DOCX)**: `/tabular_redact`

If those endpoints are not present in your deployment, fall back to the long UI-chained routes (`/apply_review_redactions`, `/redact_data`, etc.) and build `data[]` strictly from `/gradio_api/info`.

### Common gotchas

- **Arity errors** (`needed: N, got: M`) mean you called a session-heavy UI handler with the wrong `data[]`. Prefer the short endpoints above.
- **`handle_file()` gotcha** (for `gradio_client` users): do **not** wrap server-internal upload paths (e.g. `/tmp/gradio_tmp/...`) with `handle_file()`. Pass them as plain strings.
- **Container-only outputs**: outputs may be written to container paths (e.g. `/home/user/app/output/`). Plan to download via `file=...` or use a mounted output directory in Docker.

### Optional: MCP server

If you want external agents to call this app reliably without re-implementing Gradio upload/call/poll/download details, consider an **MCP server** that wraps the main tasks (`redact_document`, `apply_review_redactions`, `redact_tabular`, `summarise_document`) behind a small tool interface. See the [relevant documentation](https://github.com/seanpedrick-case/doc_redaction/blob/main/mcp_doc_redaction/README.md).

**Use as a library:** After installing from [PyPI](https://pypi.org/project/doc-redaction/) (`pip install doc_redaction`), you can call the same workflows as the Gradio `api_name` routes from Python. See the documentation: [Python Package usage (Python)](https://seanpedrick-case.github.io/doc_redaction/src/python_package_usage.html).
    
To extract text from documents, the 'Local' options are PikePDF for PDFs with selectable text, and OCR with Tesseract. Use AWS Textract to extract more complex elements e.g. handwriting, signatures, or unclear text. PaddleOCR and VLM support is also provided (see the installation instructions below). 

For PII identification, 'Local' (based on spaCy) gives good results if you are looking for common names or terms, or a custom list of terms to redact (see Redaction settings).  AWS Comprehend gives better results at a small cost.

Additional options on the 'Redaction settings' include, the type of information to redact (e.g. people, places), custom terms to include/ exclude from redaction, fuzzy matching, language settings, and whole page redaction. After redaction is complete, you can view and modify suggested redactions on the 'Review redactions' tab to quickly create a final redacted document.

NOTE: The app is not 100% accurate, and it will miss some personal information. It is essential that all outputs are reviewed **by a human** before using the final outputs.