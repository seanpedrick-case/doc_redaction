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
# Document redaction

version: 1.6.3

Redact personally identifiable information (PII) from documents (pdf, png, jpg), Word files (docx), or tabular data (xlsx/csv/parquet). Please see the [User Guide](#user-guide) for a full walkthrough of all the features in the app.
    
To extract text from documents, the 'Local' options are PikePDF for PDFs with selectable text, and OCR with Tesseract. Use AWS Textract to extract more complex elements e.g. handwriting, signatures, or unclear text. PaddleOCR and VLM support is also provided (see the installation instructions below). 

For PII identification, 'Local' (based on spaCy) gives good results if you are looking for common names or terms, or a custom list of terms to redact (see Redaction settings).  AWS Comprehend gives better results at a small cost.

Additional options on the 'Redaction settings' include, the type of information to redact (e.g. people, places), custom terms to include/ exclude from redaction, fuzzy matching, language settings, and whole page redaction. After redaction is complete, you can view and modify suggested redactions on the 'Review redactions' tab to quickly create a final redacted document.

NOTE: The app is not 100% accurate, and it will miss some personal information. It is essential that all outputs are reviewed **by a human** before using the final outputs.

---

## 🚀 Quick Start - Installation and first run

Follow these instructions to get the document redaction application running on your local machine.

### 1. Prerequisites: System Dependencies

This application relies on two external tools for OCR (Tesseract) and PDF processing (Poppler). Please install them on your system before proceeding.

---


#### **On Windows**

Installation on Windows requires downloading installers and adding the programs to your system's PATH.

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


### 2. Installation: Code and Python Packages

Once the system prerequisites are installed, you can set up the Python environment.

#### Step 1: Clone the Repository

Open your terminal or Git Bash and clone this repository:
```bash
git clone https://github.com/seanpedrick-case/doc_redaction.git
cd doc_redaction
```

#### Step 2: Create and Activate a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to isolate project dependencies and avoid conflicts with other Python projects.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Python Dependencies

##### Lightweight version (without PaddleOCR and VLM support)

This project uses `pyproject.toml` to manage dependencies. You can install everything with a single pip command. This process will also download the required Spacy models and other packages directly from their URLs.

```bash
pip install .
```

Alternatively, you can install from the `requirements_lightweight.txt` file:
```bash
pip install -r requirements_lightweight.txt
```

##### Full version (with Paddle and VLM support)

Run the following command to install the additional dependencies:

```bash
pip install .[paddle,vlm]
```

Alternatively, you can use the full `requirements.txt` file, that contains references to the PaddleOCR and related Torch/transformers dependencies (for cuda 12.9):
```bash
pip install -r requirements.txt
```

Note that the versions of both PaddleOCR and Torch installed by default are the CPU-only versions. If you want to install the equivalent GPU versions, you will need to run the following commands:
```bash
pip install paddlepaddle-gpu==3.2.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

**Note:** It is difficult to get paddlepaddle gpu working in an environment alongside torch. You may well need to reinstall the cpu version to ensure compatibility, and run paddlepaddle-gpu in a separate environment without torch installed. If you get errors related to .dll files following paddle gpu install, you may need to install the latest c++ redistributables. For Windows, you can find them [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install torchvision --index-url https://download.pytorch.org/whl/cu129
```

### 3. Run the Application

With all dependencies installed, you can now start the Gradio application.

```bash
python app.py
```

After running the command, the application will start, and you will see a local URL in your terminal (usually `http://127.0.0.1:7860`).

Open this URL in your web browser to use the document redaction tool

#### Command line interface

If instead you want to run redactions or other app functions in CLI mode, run the following for instructions:

```bash
python cli_redact.py --help
```

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

*   `SHOW_LANGUAGE_SELECTION=True`
    *   Set to `True` to display a language selection dropdown in the UI for OCR processing.

*   `CHOSEN_LOCAL_OCR_MODEL=tesseract`"
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

Now you have the app installed, what follows is a guide on how to use it for basic and advanced redaction.

# User guide

## Table of contents

### Getting Started
- [Built-in example data](#built-in-example-data)
- [Basic redaction](#basic-redaction)
- [Customising redaction options](#customising-redaction-options)
    - [Custom allow, deny, and page redaction lists](#custom-allow-deny-and-page-redaction-lists)
        - [Allow list example](#allow-list-example)
        - [Deny list example](#deny-list-example)
        - [Full page redaction list example](#full-page-redaction-list-example)
    - [Redacting additional types of personal information](#redacting-additional-types-of-personal-information)
    - [Redacting only specific pages](#redacting-only-specific-pages)
    - [Handwriting and signature redaction](#handwriting-and-signature-redaction)
- [Reviewing and modifying suggested redactions](#reviewing-and-modifying-suggested-redactions)
- [Redacting Word, tabular data files (CSV/XLSX) or copy and pasted text](#redacting-word-tabular-data-files-xlsxcsv-or-copy-and-pasted-text)
- [Identifying and redacting duplicate pages](#identifying-and-redacting-duplicate-pages)

### Advanced user guide
- [Fuzzy search and redaction](#fuzzy-search-and-redaction)
- [Export redactions to and import from Adobe Acrobat](#export-to-and-import-from-adobe)
    - [Using _for_review.pdf files with Adobe Acrobat](#using-_for_reviewpdf-files-with-adobe-acrobat)
    - [Exporting to Adobe Acrobat](#exporting-to-adobe-acrobat)
    - [Importing from Adobe Acrobat](#importing-from-adobe-acrobat)
- [Using the AWS Textract document API](#using-the-aws-textract-document-api)
- [Using AWS Textract and Comprehend when not running in an AWS environment](#using-aws-textract-and-comprehend-when-not-running-in-an-aws-environment)
- [Modifying existing redaction review files](#modifying-existing-redaction-review-files)
- [Merging redaction review files](#merging-redaction-review-files)

### Features for expert users/system administrators
- [Advanced OCR options (Hybrid OCR)](#advanced-ocr-options-hybrid-ocr)
- [Command Line Interface (CLI)](#command-line-interface-cli)

## Built-in example data

The app now includes built-in example files that you can use to quickly test different features. These examples are automatically loaded and can be accessed directly from the interface without needing to download files separately.

### Using built-in examples

**For PDF/image redaction:** On the 'Redact PDFs/images' tab, you'll see a section titled "Try an example - Click on an example below and then the 'Extract text and redact document' button". Simply click on any of the available examples to load them with pre-configured settings:

- **PDF with selectable text redaction** - Uses local text extraction with standard PII detection
- **Image redaction with local OCR** - Processes an image file using OCR
- **PDF redaction with custom entities** - Demonstrates custom entity selection (Titles, Person, Dates)
- **PDF redaction with AWS services and signature detection** - Shows AWS Textract with signature extraction (if AWS is enabled)
- **PDF redaction with custom deny list and whole page redaction** - Demonstrates advanced redaction features

Once you have clicked on an example, you can click the 'Extract text and redact document' button to load the example into the app and redact it.

**For tabular data:** On the 'Word or Excel/csv files' tab, you'll find examples for both redaction and duplicate detection:

- **CSV file redaction** - Shows how to redact specific columns in tabular data
- **Word document redaction** - Demonstrates Word document processing
- **Excel file duplicate detection** - Shows how to find duplicate rows in spreadsheet data

Once you have clicked on an example, you can click the 'Redact text/data files' button to load the example into the app and redact it. For the duplicate detection example, you can click the 'Find duplicate cells/rows' button to load the example into the app and find duplicates.

**For duplicate page detection:** On the 'Identify duplicate pages' tab, you'll find examples for finding duplicate content in documents:

- **Find duplicate pages of text in document OCR outputs** - Uses page-level analysis with a similarity threshold of 0.95 and minimum word count of 10
- **Find duplicate text lines in document OCR outputs** - Uses line-level analysis with a similarity threshold of 0.95 and minimum word count of 3

Once you have clicked on an example, you can click the 'Identify duplicate pages/subdocuments' button to load the example into the app and find duplicate content.

### External example files (optional)

If you prefer to use your own example files or want to follow along with specific tutorials, you can still download these external example files:

- [Example of files sent to a professor before applying](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/example_of_emails_sent_to_a_professor_before_applying.pdf)
- [Example complaint letter (jpg)](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/example_complaint_letter.jpg)
- [Partnership Agreement Toolkit (for signatures and more advanced usage)](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/Partnership-Agreement-Toolkit_0_0.pdf)
- [Dummy case note data](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/combined_case_notes.csv)

## Basic redaction

The document redaction app can detect personally-identifiable information (PII) in documents. Documents can be redacted directly, or suggested redactions can be reviewed and modified using a grapical user interface. Basic document redaction can be performed quickly using the default options.

Download the example PDFs above to your computer. Open up the redaction app with the link provided by email.

![Upload files](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/file_upload_highlight.PNG)

### Upload files to the app

The 'Redact PDFs/images tab' currently accepts PDFs and image files (JPG, PNG) for redaction. Click on the 'Drop files here or Click to Upload' area of the screen, and select one of the three different [example files](#example-data-files) (they should all be stored in the same folder if you want them to be redacted at the same time).

### Text extraction

You can modify default text extraction methods by clicking on the 'Change default text extraction method...' box'.

Here you can select one of the three text extraction options:
- **'Local model - selectable text'** - This will read text directly from PDFs that have selectable text to redact (using PikePDF). This is fine for most PDFs, but will find nothing if the PDF does not have selectable text, and it is not good for handwriting or signatures. If it encounters an image file, it will send it onto the second option below.
- **'Local OCR model - PDFs without selectable text'** - This option will use a simple Optical Character Recognition (OCR) model (Tesseract) to pull out text from a PDF/image that it 'sees'. This can handle most typed text in PDFs/images without selectable text, but struggles with handwriting/signatures. If you are interested in the latter, then you should use the third option if available.
- **'AWS Textract service - all PDF types'** - Only available for instances of the app running on AWS. AWS Textract is a service that performs OCR on documents within their secure service. This is a more advanced version of OCR compared to the local option, and carries a (relatively small) cost. Textract excels in complex documents based on images, or documents that contain a lot of handwriting and signatures.

### Enable AWS Textract signature extraction
If you chose the AWS Textract service above, you can choose if you want handwriting and/or signatures redacted by default. Choosing signatures here will have a cost implication, as identifying signatures will cost ~£2.66 ($3.50) per 1,000 pages vs ~£1.14 ($1.50) per 1,000 pages without signature detection. 

![AWS Textract handwriting and signature options](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/textract_handwriting_signatures.PNG)

**NOTE:** it is also possible to enable form extraction, layout extraction, and table extraction with AWS Textract. This is not enabled by default, but it is possible for your system admin to enable this feature in the config file.

### PII redaction method

If you are running with the AWS service enabled, here you will also have a choice for PII redaction method:
- **'Only extract text - (no redaction)'** - If you are only interested in getting the text out of the document for further processing (e.g. to find duplicate pages, or to review text on the Review redactions page)
- **'Local'** - This uses the spacy package to rapidly detect PII in extracted text. This method is often sufficient if you are just interested in redacting specific terms defined in a custom list. 
- **'AWS Comprehend'** - This method calls an AWS service to provide more accurate identification of PII in extracted text.

### Optional - costs and time estimation
If the option is enabled (by your system admin, in the config file), you will see a cost and time estimate for the redaction process. 'Existing Textract output file found' will be checked automatically if previous Textract text extraction files exist in the output folder, or have been [previously uploaded by the user](#aws-textract-outputs) (saving time and money for redaction). 

![Cost and time estimation](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/costs_and_time.PNG)

### Optional - cost code selection
If the option is enabled (by your system admin, in the config file), you may be prompted to select a cost code before continuing with the redaction task.

![Cost code selection](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/cost_code_selection.PNG)

The relevant cost code can be found either by: 1. Using the search bar above the data table to find relevant cost codes, then clicking on the relevant row, or 2. typing it directly into the dropdown to the right, where it should filter as you type.

### Optional - Submit whole documents to Textract API
If this option is enabled (by your system admin, in the config file), you will have the option to submit whole documents in quick succession to the AWS Textract service to get extracted text outputs quickly (faster than using the 'Redact document' process described here). This feature is described in more detail in the [advanced user guide](#using-the-aws-textract-document-api).

![Textract document API](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/textract_document_api.PNG)

### Redact the document

Click 'Redact document'. After loading in the document, the app should be able to process about 30 pages per minute (depending on redaction methods chose above). When ready, you should see a message saying that processing is complete, with output files appearing in the bottom right.

### Redaction outputs

![Redaction outputs](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/redaction_outputs.PNG)

- **'...redacted.pdf'** files contain the original pdf with suggested redacted text deleted and replaced by a black box on top of the document.
- **'...redactions_for_review.pdf'** files contain the original PDF with redaction boxes overlaid but the original text still visible underneath. This file is designed for use in Adobe Acrobat and other PDF viewers where you can see the suggested redactions without the text being permanently removed. This is particularly useful for reviewing redactions before finalising them.
- **'...ocr_results.csv'** files contain the line-by-line text outputs from the entire document. This file can be useful for later searching through for any terms of interest in the document (e.g. using Excel or a similar program).
- **'...review_file.csv'** files are the review files that contain details and locations of all of the suggested redactions in the document. This file is key to the [review process](#reviewing-and-modifying-suggested-redactions), and should be downloaded to use later for this.

### Additional AWS Textract / local OCR outputs

If you have used the AWS Textract option for extracting text, you may also see a '..._textract.json' file. This file contains all the relevant extracted text information that comes from the AWS Textract service. You can keep this file and upload it at a later date alongside your input document, which will enable you to skip calling AWS Textract every single time you want to do a redaction task, as follows:

![Document upload alongside Textract](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/document_upload_with_textract.PNG)

#### Additional outputs in the log file outputs

On the Redaction settings tab, near the bottom of the pagethere is a section called 'Log file outputs'. This section contains the following files:

You may see a '..._ocr_results_with_words... .json' file. This file works in the same way as the AWS Textract .json results described above, and can be uploaded alongside an input document to save time on text extraction in future in the same way.

Also you will see a 'decision_process_table.csv' file. This file contains a table of the decisions made by the app for each page of the document. This can be useful for debugging and understanding the decisions made by the app.

Additionally, if the option is enabled by your system administrator, on this tab you may see an image of the output from the OCR model used to extract the text from the document, an image ending with page number and '_visualisations.jpg'. A separate image will be created for each page of the document like the one below. This can be useful for seeing at a glance whether the text extraction process for a page was successful, and whether word-level bounding boxes are correctly positioned.

![Text analysis output](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/example_complaint_letter_1_textract_visualisations.jpg)

### Downloading output files from previous redaction tasks

If you are logged in via AWS Cognito and you lose your app page for some reason (e.g. from a crash, reloading), it is possible recover your previous output files, provided the server has not been shut down since you redacted the document. If enabled, this feature can be found at the bottom of the front tab, called 'View and download all output files from this session'. If you open this and click on 'Refresh files in output folder' you should see a file directory of all files. If you click on the box next to a given file, it should appear below for you to download.

![View all output files](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/view_all_output_files.PNG)

### Basic redaction summary

We have covered redacting documents with the default redaction options. The '...redacted.pdf' file output may be enough for your purposes. But it is very likely that you will need to customise your redaction options, which we will cover below. 

## Customising redaction options

On the 'Redaction settings' page, there are a number of options that you can tweak to better match your use case and needs.

### Custom allow, deny, and page redaction lists

The app allows you to specify terms that should never be redacted (an allow list), terms that should always be redacted (a deny list), and also to provide a list of page numbers for pages that should be fully redacted.

![Custom allow, deny, and page redaction lists](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/allow_list/allow_deny_full_page_list.PNG)

#### Allow list example

It may be the case that specific terms that are frequently redacted are not interesting to 

In the redacted outputs of the 'Example of files sent to a professor before applying' PDF, you can see that it is frequently redacting references to Dr Hyde's lab in the main body of the text. Let's say that references to Dr Hyde were not considered personal information in this context. You can exclude this term from redaction (and others) by providing an 'allow list' file. This is simply a csv that contains the case sensitive terms to exclude in the first column, in our example, 'Hyde' and 'Muller glia'. The example file is provided [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/allow_list/allow_list.csv). 

To import this to use with your redaction tasks, go to the 'Redaction settings' tab, click on the 'Import allow list file' button halfway down, and select the csv file you have created. It should be loaded for next time you hit the redact button. Go back to the first tab and do this.

#### Deny list example

Say you wanted to remove specific terms from a document. In this app you can do this by providing a custom deny list as a csv. Like for the allow list described above, this should be a one-column csv without a column header. The app will suggest each individual term in the list with exact spelling as whole words. So it won't select text from within words. To enable this feature, the 'CUSTOM' tag needs to be chosen as a redaction entity [(the process for adding/removing entity types to redact is described below)](#redacting-additional-types-of-personal-information).

Here is an example using the [Partnership Agreement Toolkit file](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/Partnership-Agreement-Toolkit_0_0.pdf). This is an [example of a custom deny list file](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/allow_list/partnership_toolkit_redact_custom_deny_list.csv). 'Sister', 'Sister City'
'Sister Cities', 'Friendship City' have been listed as specific terms to redact. You can see the outputs of this redaction process on the review page:

![Deny list redaction Partnership file](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/allow_list/deny_list_partnership_example.PNG). 

You can see that the app has highlighted all instances of these terms on the page shown. You can then consider each of these terms for modification or removal on the review page [explained here](#reviewing-and-modifying-suggested-redactions).

#### Full page redaction list example

There may be full pages in a document that you want to redact. The app also provides the capability of redacting pages completely based on a list of input page numbers in a csv. The format of the input file is the same as that for the allow and deny lists described above - a one-column csv without a column header. An [example of this is here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/allow_list/partnership_toolkit_redact_some_pages.csv). You can see an example of the redacted page on the review page:

![Whole page partnership redaction](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/allow_list/whole_page_partnership_example.PNG).

Using the above approaches to allow, deny, and full page redaction lists will give you an output [like this](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/allow_list/Partnership-Agreement-Toolkit_0_0_redacted.pdf).

#### Adding to the loaded allow, deny, and whole page lists in-app

If you open the accordion below the allow list options called 'Manually modify custom allow...', you should be able to see a few tables with options to add new rows:

![Manually modify allow or deny list](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/allow_list/manually_modify.PNG)

If the table is empty, you can add a new entry, you can add a new row by clicking on the '+' item below each table header. If there is existing data, you may need to click on the three dots to the right and select 'Add row below'. Type the item you wish to keep/remove in the cell, and then (important) press enter to add this new item to the allow/deny/whole page list. Your output tables should look something like below.

![Manually modify allow or deny list filled](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/allow_list/manually_modify_filled.PNG)

### Redacting additional types of personal information

You may want to redact additional types of information beyond the defaults, or you may not be interested in default suggested entity types. There are dates in the example complaint letter. Say we wanted to redact those dates also?

Under the 'Redaction settings' tab, go to 'Entities to redact (click close to down arrow for full list)'. Different dropdowns are provided according to whether you are using the Local service to redact PII, or the AWS Comprehend service. Click within the empty box close to the dropdown arrow and you should see a list of possible 'entities' to redact. Select 'DATE_TIME' and it should appear in the main list. To remove items, click on the 'x' next to their name.

![Redacting additional types of information dropdown](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/additional_entities/additional_entities_select.PNG)

Now, go back to the main screen and click 'Redact Document' again. You should now get a redacted version of 'Example complaint letter' that has the dates and times removed.

If you want to redact different files, I suggest you refresh your browser page to start a new session and unload all previous data.

## Redacting only specific pages

Say also we are only interested in redacting page 1 of the loaded documents. On the Redaction settings tab, select 'Lowest page to redact' as 1, and 'Highest page to redact' also as 1. When you next redact your documents, only the first page will be modified. The output files should now have a suffix similar to '..._1_1.pdf', indicating the lowest and highest page numbers that were redacted.

![Selecting specific pages to redact](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/allow_list/select_pages.PNG)

## Handwriting and signature redaction

The file [Partnership Agreement Toolkit (for signatures and more advanced usage)](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/Partnership-Agreement-Toolkit_0_0.pdf) is provided as an example document to test AWS Textract + redaction with a document that has signatures in. If you have access to AWS Textract in the app, try removing all entity types from redaction on the Redaction settings and clicking the big X to the right of 'Entities to redact'. 

To ensure that handwriting and signatures are enabled (enabled by default), on the front screen go the 'AWS Textract signature detection' to enable/disable the following options :

![Handwriting and signatures](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/textract_handwriting_signatures.PNG)

The outputs should show handwriting/signatures redacted (see pages 5 - 7), which you can inspect and modify on the 'Review redactions' tab.

![Handwriting and signatures redacted example](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/refs/heads/main/review_redactions/Signatures%20and%20handwriting%20found.PNG)

## Reviewing and modifying suggested redactions

Sometimes the app will suggest redactions that are incorrect, or will miss personal information entirely. The app allows you to review and modify suggested redactions to compensate for this. You can do this on the 'Review redactions' tab.

We will go through ways to review suggested redactions with an example.On the first tab 'PDFs/images' upload the ['Example of files sent to a professor before applying.pdf'](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/example_of_emails_sent_to_a_professor_before_applying.pdf) file. Let's stick with the 'Local model - selectable text' option, and click 'Redact document'. Once the outputs are created, go to the 'Review redactions' tab.

On the 'Review redactions' tab you have a visual interface that allows you to inspect and modify redactions suggested by the app. There are quite a few options to look at, so we'll go from top to bottom.

![Review redactions](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/review_redactions.PNG)

### Uploading documents for review

The top area has a file upload area where you can upload files for review . In the left box, upload the original PDF file. Click '1. Upload original PDF'. In the right box, you can upload the '..._review_file.csv' that is produced by the redaction process.

Optionally, you can upload a '..._ocr_result_with_words' file here, that will allow you to search through the text and easily [add new redactions based on word search](#searching-and-adding-custom-redactions). You can also upload one of the '..._ocr_output.csv' file here that comes out of a redaction task, so that you can navigate the extracted text from the document. Click the button '2. Upload Review or OCR csv files' load in these files.

Now you can review and modify the suggested redactions using the interface described below.

![Search extracted text](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/search_extracted_text.PNG)

You can upload the three review files in the box (unredacted document, '..._review_file.csv' and '..._ocr_output.csv' file) before clicking '**Review redactions based on original PDF...**', as in the image below:

![Upload three files for review](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/upload_three_files.PNG)

**NOTE:** ensure you upload the ***unredacted*** document here and not the redacted version, otherwise you will be checking over a document that already has redaction boxes applied!

### Page navigation

You can change the page viewed either by clicking 'Previous page' or 'Next page', or by typing a specific page number in the 'Current page' box and pressing Enter on your keyboard. Each time you switch page, it will save redactions you have made on the page you are moving from, so you will not lose changes you have made.

You can also navigate to different pages by clicking on rows in the tables under 'Search suggested redactions' to the right, or 'search all extracted text' (if enabled) beneath that.

### The document viewer pane

On the selected page, each redaction is highlighted with a box next to its suggested redaction label (e.g. person, email).

![Document view pane](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/document_viewer_pane.PNG)

There are a number of different options to add and modify redaction boxes and page on the document viewer pane. To zoom in and out of the page, use your mouse wheel. To move around the page while zoomed, you need to be in modify mode. Scroll to the bottom of the document viewer to see the relevant controls. You should see a box icon, a hand icon, and two arrows pointing counter-clockwise and clockwise.

![Change redaction mode](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/change_review_mode.PNG)

Click on the hand icon to go into modify mode. When you click and hold on the document viewer, This will allow you to move around the page when zoomed in. To rotate the page, you can click on either of the round arrow buttons to turn in that direction. 

**NOTE:** When you switch page, the viewer will stay in your selected orientation, so if it looks strange, just rotate the page again and hopefully it will look correct!

#### Modify existing redactions (hand icon)

After clicking on the hand icon, the interface allows you to modify existing redaction boxes. When in this mode, you can click and hold on an existing box to move it. 

![Modify existing redaction box](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/modify_existing_redaction_box.PNG)

Click on one of the small boxes at the edges to change the size of the box. To delete a box, click on it to highlight it, then press delete on your keyboard. Alternatively, double click on a box and click 'Remove' on the box that appears.

![Remove existing redaction box](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/existing_redaction_box_remove.PNG)

#### Add new redaction boxes (box icon)

To change to 'add redaction boxes' mode, scroll to the bottom of the page. Click on the box icon, and your cursor will change into a crosshair. Now you can add new redaction boxes where you wish. A popup will appear when you create a new box so you can select a label and colour for the new box.

#### 'Locking in' new redaction box format

It is possible to lock in a chosen format for new redaction boxes so that you don't have the popup appearing each time. When you make a new box, select the options for your 'locked' format, and then click on the lock icon on the left side of the popup, which should turn blue.

![Lock redaction box format](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/new_redaction_box_lock_mode.PNG)

You can now add new redaction boxes without a popup appearing. If you want to change or 'unlock' the your chosen box format, you can click on the new icon that has appeared at the bottom of the document viewer pane that looks a little like a gift tag. You can then change the defaults, or click on the lock icon again to 'unlock' the new box format - then popups will appear again each time you create a new box.

![Change or unlock redaction box format](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/change_review_mode_with_lock.PNG)

### Apply redactions to PDF and Save changes on current page

Once you have reviewed all the redactions in your document and you are happy with the outputs, you can click 'Apply revised redactions to PDF' to create a new '_redacted.pdf' output alongside a new '_review_file.csv' output.

If you are working on a page and haven't saved for a while, you can click 'Save changes on current page to file' to ensure that they are saved to an updated 'review_file.csv' output.

![Review modified outputs](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/review_mod_outputs.PNG)

### Selecting and removing redaction boxes using the 'Search suggested redactions' table

The table shows a list of all the suggested redactions in the document alongside the page, label, and text (if available).

![Search suggested redaction area](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/list_find_labels.PNG)

If you click on one of the rows in this table, you will be taken to the page of the redaction. Clicking on a redaction row on the same page will change the colour of redaction box to blue to help you locate it in the document viewer (just when using the app, not in redacted output PDFs).

![Search suggested redaction area](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/review_row_highlight.PNG)

You can choose a specific entity type to see which pages the entity is present on. If you want to go to the page specified in the table, you can click on a cell in the table and the review page will be changed to that page.

To filter the 'Search suggested redactions' table you can:
1. Click on one of the dropdowns (Redaction category, Page, Text), and select an option, or
2. Write text in the 'Filter' box just above the table. Click the blue box to apply the filter to the table.

Once you have filtered the table, or selected a row from the table, you have a few options underneath on what you can do with the filtered rows:

- Click the **Exclude all redactions in table** button to remove all redactions visible in the table from the document. **Important:** ensure that you have clicked the blue tick icon next to the search box before doing this, or you will remove all redactions from the document. If you do end up doing this, click the 'Undo last element removal' button below to restore the redactions.
- Click the **Exclude specific redaction row** button to remove only the redaction from the last row you clicked on from the document. The currently selected row is visible below.
- Click the **Exclude all redactions with the same text as selected row** button to remove all redactions from the document that are exactly the same as the selected row text.

**NOTE**: After excluding redactions using any of the above options, click the 'Reset filters' button below to ensure that the dropdowns and table return to seeing all remaining redactions in the document.

If you made a mistake, click the 'Undo last element removal' button to restore the Search suggested redactions table to its previous state (can only undo the last action).

### Searching and Adding Custom Redactions

After a document has been processed, you may need to redact specific terms, names, or phrases that the automatic PII (Personally Identifiable Information) detection might have missed. The **"Search text and redact"** tab gives you the power to find and redact any text within your document manually.

#### How to Use the Search and Redact Feature

The workflow is designed to be simple: **Search → Select → Redact**.

---

#### **Step 1: Search for Text**

1.  Navigate to the **"Search text and redact"** tab.
2.  The main table will initially be populated with all the text extracted from the document for a page, broken down by word.
3.  To narrow this down, use the **"Multi-word text search"** box to type the word or phrase you want to find (this will search the whole document). If you want to do a regex-based search, tick the 'Enable regex pattern matching' box under 'Search options' below (Note this will only be able to search for patterns in text within each cell).
4.  Click the **"Search"** button or press Enter.
5.  The table below will update to show only the rows containing text that matches your search query.

> **Tip:** You can also filter the results by page number using the **"Page"** dropdown. To clear all filters and see the full text again, click the **"Reset table to original state"** button.

---

#### **Step 2: Select and Review a Match**

When you click on any row in the search results table:

*   The document preview on the left will automatically jump to that page, allowing you to see the word in its original context.
*   The details of your selection will appear in the smaller **"Selected row"** table for confirmation.

---

#### **Step 3: Choose Your Redaction Method**

You have several powerful options for redacting the text you've found:

*   **Redact a Single, Specific Instance:**
    *   Click on the exact row in the table you want to redact.
    *   Click the **`Redact specific text row`** button.
    *   Only that single instance will be redacted.

*   **Redact All Instances of a Word/Phrase:**
    *   Let's say you want to redact the project name "Project Alpha" everywhere it appears.
    *   Find and select one instance of "Project Alpha" in the table.
    *   Click the **`Redact all words with same text as selected row`** button.
    *   The application will find and redact every single occurrence of "Project Alpha" throughout the entire document.

*   **Redact All Current Search Results:**
    *   Perform a search (e.g., for a specific person's name).
    *   If you are confident that every result shown in the filtered table should be redacted, click the **`Redact all text in table`** button.
    *   This will apply a redaction to all currently visible items in the table in one go.

---

#### **Customising Your New Redactions**

Before you click one of the redact buttons, you can customize the appearance and label of the new redactions under the **"Search options"** accordion:

*   **Label for new redactions:** Change the text that appears on the redaction box (default is "Redaction"). You could change this to "CONFIDENTIAL" or "CUSTOM".
*   **Colour for labels:** Set a custom color for the redaction box by providing an RGB value. The format must be three numbers (0-255) in parentheses, for example:
    *   ` (255, 0, 0) ` for Red
    *   ` (0, 0, 0) ` for Black
    *   ` (255, 255, 0) ` for Yellow

#### **Undoing a Mistake**

If you make a mistake, you can reverse the last redaction action you performed on this tab.

*   Click the **`Undo latest redaction`** button. This will revert the last set of redactions you added (whether it was a single row, all of a certain text, or all search results).

> **Important:** This undo button only works for the *most recent* action. It maintains a single backup state, so it cannot undo actions that are two or more steps in the past.

### Navigating through the document using the 'Search all extracted text' 

The 'search all extracted text' table will contain text if you have just redacted a document, or if you have uploaded a '..._ocr_output.csv' file alongside a document file and review file on the Review redactions tab as [described above](#uploading-documents-for-review). 

You can navigate through the document using this table. When you click on a row, the Document viewer pane to the left will change to the selected page. 

![Search suggested redaction area](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/select_extracted_text.PNG)

You can search through the extracted text by using the search bar just above the table, which should filter as you type. To apply the filter and 'cut' the table, click on the blue tick inside the box next to your search term. To return the table to its original content, click the button below the table 'Reset OCR output table filter'.

![Search suggested redaction area](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/review_redactions/search_extracted_text.PNG)

## Redacting Word, tabular data files (XLSX/CSV) or copy and pasted text

### Word or tabular data files (XLSX/CSV)

The app can be used to redact Word (.docx), or tabular data files such as xlsx or csv files. For this to work properly, your data file needs to be in a simple table format, with a single table starting from the first cell (A1), and no other information in the sheet. Similarly for .xlsx files, each sheet in the file that you want to redact should be in this simple format.

To demonstrate this, we can use [the example csv file 'combined_case_notes.csv'](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/combined_case_notes.csv), which is a small dataset of dummy social care case notes. Go to the 'Open text or Excel/csv files' tab. Drop the file into the upload area. After the file is loaded, you should see the suggested columns for redaction in the box underneath. You can select and deselect columns to redact as you wish from this list.

![csv upload](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/tabular_files/file_upload_csv_columns.PNG)

If you were instead to upload an xlsx file, you would see also a list of all the sheets in the xlsx file that can be redacted. The 'Select columns' area underneath will suggest a list of all columns in the file across all sheets.

![xlsx upload](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/tabular_files/file_upload_xlsx_columns.PNG)

Once you have chosen your input file and sheets/columns to redact, you can choose the redaction method. 'Local' will use the same local model as used for documents on the first tab. 'AWS Comprehend' will give better results, at a slight cost.

When you click Redact text/data files, you will see the progress of the redaction task by file and sheet, and you will receive a csv output with the redacted data.

### Choosing output anonymisation format
You can also choose the anonymisation format of your output results.  Open the tab 'Anonymisation output format' to see the options. By default, any detected PII will be replaced with the word 'REDACTED' in the cell. You can choose one of the following options as the form of replacement for the redacted text:
- replace with 'REDACTED': Replaced by the word 'REDACTED' (default)
- replace with <ENTITY_NAME>: Replaced by e.g. 'PERSON' for people, 'EMAIL_ADDRESS' for emails etc.
- redact completely: Text is removed completely and replaced by nothing.
- hash: Replaced by a unique long ID code that is consistent with entity text. I.e. a particular name will always have the same ID code.
- mask: Replace with stars '*'.

### Redacting copy and pasted text
You can also write open text into an input box and redact that using the same methods as described above. To do this, write or paste text into the 'Enter open text' box that appears when you open the 'Redact open text' tab. Then select a redaction method, and an anonymisation output format as described above. The redacted text will be printed in the output textbox, and will also be saved to a simple csv file in the output file box.  

![Text analysis output](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/tabular_files/text_anonymisation_outputs.PNG)

### Redaction log outputs
A list of the suggested redaction outputs from the tabular data / open text data redaction is available on the Redaction settings page under 'Log file outputs'.


## Identifying and redacting duplicate pages

The files for this section are stored [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/duplicate_page_find_in_app/).

Some redaction tasks involve removing duplicate pages of text that may exist across multiple documents. This feature helps you find and remove duplicate content that may exist in single or multiple documents.  It can identify everything from single identical pages to multi-page sections (subdocuments). The process involves three main steps: configuring the analysis, reviewing the results in the interactive interface, and then using the generated files to perform the redactions.

### Duplicate page detection in documents

This section covers finding duplicate pages across PDF documents using OCR output files.

![Example duplicate page inputs](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/duplicate_page_find_in_app/img/duplicate_page_input_interface_new.PNG)

**Step 1: Upload and Configure the Analysis**
First, navigate to the "Identify duplicate pages" tab. Upload all the ocr_output.csv files you wish to compare into the file area. These files are generated every time you run a redaction task and contain the text for each page of a document.

For our example, you can upload the four 'ocr_output.csv' files provided in the example folder into the file area. Click 'Identify duplicate pages' and you will see a number of files returned. In case you want to see the original PDFs, they are available [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/duplicate_page_find_in_app/input_pdfs/).

The default options will search for matching subdocuments of any length. Before running the analysis, you can configure these matching parameters to tell the tool what you're looking for:

![Duplicate matching parameters](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/duplicate_page_find_in_app/img/duplicate_matching_parameters.PNG)

*Matching Parameters*
- **Similarity Threshold:** A score from 0 to 1. Pages or sequences of pages with a calculated text similarity above this value will be considered a match. The default of 0.9 (90%) is a good starting point for finding near-identical pages.
- **Min Word Count:** Pages with fewer words than this value will be completely ignored during the comparison. This is extremely useful for filtering out blank pages, title pages, or boilerplate pages that might otherwise create noise in the results. The default is 10.
- **Choosing a Matching Strategy:** You have three main options to find duplicate content.
    - *'Subdocument' matching (default):* Use this to find the longest possible sequence of matching pages. The tool will find an initial match and then automatically expand it forward page-by-page until the consecutive match breaks. This is the best method for identifying complete copied chapters or sections of unknown length. This is enabled by default by ticking the "Enable 'subdocument' matching" box. This setting overrides the described below.
    - *Minimum length subdocument matching:* Use this to find sequences of consecutively matching pages with a minimum page lenght. For example, setting the slider to 3 will only return sections that are at least 3 pages long. How to enable: Untick the "Enable 'subdocument' matching" box and set the "Minimum consecutive pages" slider to a value greater than 1.
    - *Single Page Matching:* Use this to find all individual page pairs that are similar to each other. Leave the "Enable 'subdocument' matching" box unchecked and keep the "Minimum consecutive pages" slider at 1.

Once your parameters are set, click the "Identify duplicate pages/subdocuments" button.

**Step 2: Review Results in the Interface**
After the analysis is complete, the results will be displayed directly in the interface.

*Analysis Summary:* A table will appear showing a summary of all the matches found. The columns will change depending on the matching strategy you chose. For subdocument matches, it will show the start and end pages of the matched sequence.

*Interactive Preview:* This is the most important part of the review process. Click on any row in the summary table. The full text of the matching page(s) will appear side-by-side in the "Full Text Preview" section below, allowing you to instantly verify the accuracy of the match.

![Duplicate review interface](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/duplicate_page_find_in_app/img/duplicate_page_output_review_overview.PNG)

**Step 3: Download and Use the Output Files**
The analysis also generates a set of downloadable files for your records and for performing redactions.


- page_similarity_results.csv: This is a detailed report of the analysis you just ran. It shows a breakdown of the pages from each file that are most similar to each other above the similarity threshold. You can compare the text in the two columns 'Page_1_Text' and 'Page_2_Text'. For single-page matches, it will list each pair of matching pages. For subdocument matches, it will list the start and end pages of each matched sequence, along with the total length of the match.

![Page similarity file example](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/duplicate_page_find_in_app/img/page_similarity_example.PNG)

- [Original_Filename]_pages_to_redact.csv: For each input document that was found to contain duplicate content, a separate redaction list is created. This is a simple, one-column CSV file containing a list of all page numbers that should be removed. To use these files, you can either upload the original document (i.e. the PDF) on the 'Review redactions' tab, and then click on the 'Apply relevant duplicate page output to document currently under review' button. You should see the whole pages suggested for redaction on the 'Review redactions' tab. Alternatively, you can reupload the file into the whole page redaction section as described in the ['Full page redaction list example' section](#full-page-redaction-list-example).

![Example duplicate page redaction list](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/duplicate_page_find_in_app/img/duplicate_page_output_interface_new.PNG)

If you want to combine the results from this redaction process with previous redaction tasks for the same PDF, you could merge review file outputs following the steps described in [Merging existing redaction review files](#merging-existing-redaction-review-files) above.

### Duplicate detection in tabular data

The app also includes functionality to find duplicate cells or rows in CSV, Excel, or Parquet files. This is particularly useful for cleaning datasets where you need to identify and remove duplicate entries.

**Step 1: Upload files and configure analysis**

Navigate to the 'Word or Excel/csv files' tab and scroll down to the "Find duplicate cells in tabular data" section. Upload your tabular files (CSV, Excel, or Parquet) and configure the analysis parameters:

- **Similarity threshold**: Score (0-1) to consider cells a match. 1 = perfect match
- **Minimum word count**: Cells with fewer words than this value are ignored
- **Do initial clean of text**: Remove URLs, HTML tags, and non-ASCII characters
- **Remove duplicate rows**: Automatically remove duplicate rows from deduplicated files
- **Select Excel sheet names**: Choose which sheets to analyze (for Excel files)
- **Select text columns**: Choose which columns contain text to analyze

**Step 2: Review results**

After clicking "Find duplicate cells/rows", the results will be displayed in a table showing:
- File1, Row1, File2, Row2
- Similarity_Score
- Text1, Text2 (the actual text content being compared)

Click on any row to see more details about the duplicate match in the preview boxes below.

**Step 3: Remove duplicates**

Select a file from the dropdown and click "Remove duplicate rows from selected file" to create a cleaned version with duplicates removed. The cleaned file will be available for download.

# Advanced user guide

This advanced user guide covers features that require system administration access or command-line usage. These features are typically used by system administrators or advanced users who need more control over the redaction process.

## Fuzzy search and redaction

The files for this section are stored [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/fuzzy_search/).

Sometimes you may be searching for terms that are slightly mispelled throughout a document, for example names. The document redaction app gives the option for searching for long phrases that may contain spelling mistakes, a method called 'fuzzy matching'.

To do this, go to the Redaction Settings, and the 'Select entity types to redact' area. In the box below relevant to your chosen redaction method (local or AWS Comprehend), select 'CUSTOM_FUZZY' from the list. Next, we can select the maximum number of spelling mistakes allowed in the search (up to nine). Here, you can either type in a number or use the small arrows to the right of the box. Change this option to 3. This will allow for a maximum of three 'changes' in text needed to match to the desired search terms.

The other option we can leave as is (should fuzzy search match on entire phrases in deny list) - this option would allow you to fuzzy search on each individual word in the search phrase (apart from stop words).

Next, we can upload a deny list on the same page to do the fuzzy search. A relevant deny list file can be found [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/fuzzy_search/Partnership-Agreement-Toolkit_test_deny_list_para_single_spell.csv) - you can upload it following [these steps](#deny-list-example). You will notice that the suggested deny list has spelling mistakes compared to phrases found in the example document.

![Deny list example with spelling mistakes](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/fuzzy_search/img/fuzzy_deny_list_example.PNG)

Upload the [Partnership-Agreement-Toolkit file](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/Partnership-Agreement-Toolkit_0_0.pdf) into the 'Redact document' area on the first tab. Now, press the 'Redact document' button.

Using these deny list with spelling mistakes, the app fuzzy match these terms to the correct text in the document. After redaction is complete, go to the Review Redactions tab to check the first tabs. You should see that the phrases in the deny list have been successfully matched.

![Fuzzy match review outputs](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/fuzzy_search/img/fuzzy_search_review.PNG)

## Export to and import from Adobe

Files for this section are stored [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/export_to_adobe/).

The Document Redaction app has enhanced features for working with Adobe Acrobat. You can now export suggested redactions to Adobe, import Adobe comment files into the app, and use the new `_for_review.pdf` files directly in Adobe Acrobat.

### Using _for_review.pdf files with Adobe Acrobat

The app now generates `...redactions_for_review.pdf` files that contain the original PDF with redaction boxes overlaid but the original text still visible underneath. These files are specifically designed for use in Adobe Acrobat and other PDF viewers where you can:

- See the suggested redactions without the text being permanently removed
- Review redactions before finalising them
- Use Adobe Acrobat's built-in redaction tools to modify or apply the redactions
- Export the final redacted version directly from Adobe

Simply open the `...redactions_for_review.pdf` file in Adobe Acrobat to begin reviewing and modifying the suggested redactions.

### Exporting to Adobe Acrobat

To convert suggested redactions to Adobe format, you need to have the original PDF and a review file csv in the input box at the top of the Review redactions page.

![Input area for files for Adobe export](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/export_to_adobe/img/adobe_export_input_area.PNG)

Then, you can find the export to Adobe option at the bottom of the Review redactions tab. Adobe comment files will be output here.

![Adobe export/import options](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/export_to_adobe/img/export_to_adobe_interface.PNG)

Once the input files are ready, you can click on the 'Convert review file to Adobe comment format'. You should see a file appear in the output box with a '.xfdf' file type. To use this in Adobe, after download to your computer, you should be able to double click on it, and a pop-up box will appear asking you to find the PDF file associated with it. Find the original PDF file used for your redaction task. The file should be opened up in Adobe Acrobat with the suggested redactions.

![Suggested redactions in Adobe Acrobat](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/export_to_adobe/img/adobe_redact_example.PNG)

### Importing from Adobe Acrobat

The app also allows you to import .xfdf files from Adobe Acrobat. To do this, go to the same Adobe import/export area as described above at the bottom of the Review Redactions tab. In this box, you need to upload a .xfdf Adobe comment file, along with the relevant original PDF for redaction.

![Adobe import interface](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/export_to_adobe/img/import_from_adobe_interface.PNG)

When you click the 'convert .xfdf comment file to review_file.csv' button, the app should take you up to the top of the screen where the new review file has been created and can be downloaded.

![Outputs from Adobe import](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/export_to_adobe/img/import_from_adobe_interface_outputs.PNG)

## Using the AWS Textract document API

This option can be enabled by your system admin, in the config file ('SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS' environment variable, and subsequent variables). Using this, you will have the option to submit whole documents in quick succession to the AWS Textract service to get extracted text outputs quickly (faster than using the 'Redact document' process described here).

### Starting a new Textract API job

To use this feature, first upload a document file in the file input box [in the usual way](#upload-files-to-the-app) on the first tab of the app. Under AWS Textract signature detection you can select whether or not you would like to analyse signatures or not (with a [cost implication](#optional---select-signature-extraction)).

Then, open the section under the heading 'Submit whole document to AWS Textract API...'.

![Textract document API menu](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/quick_start/textract_document_api.PNG)

Click 'Analyse document with AWS Textract API call'. After a few seconds, the job should be submitted to the AWS Textract service. The box 'Job ID to check status' should now have an ID filled in. If it is not already filled with previous jobs (up to seven days old), the table should have a row added with details of the new API job.

Click the button underneath, 'Check status of Textract job and download', to see progress on the job. Processing will continue in the background until the job is ready, so it is worth periodically clicking this button to see if the outputs are ready. In testing, and as a rough estimate, it seems like this process takes about five seconds per page. However, this has not been tested with very large documents. Once ready, the '_textract.json' output should appear below.

### Textract API job outputs

The '_textract.json' output can be used to speed up further redaction tasks as [described previously](#optional---costs-and-time-estimation), the 'Existing Textract output file found' flag should now be ticked.

![Textract document API initial ouputs](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/textract_api/textract_api_initial_outputs.PNG)

You can now easily get the '..._ocr_output.csv' redaction output based on this '_textract.json' (described in [Redaction outputs](#redaction-outputs)) by clicking on the button 'Convert Textract job outputs to OCR results'. You can now use this file e.g. for [identifying duplicate pages](#identifying-and-redacting-duplicate-pages), or for redaction review.



## Modifying existing redaction review files
You can find the folder containing the files discussed in this section [here](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/merge_review_files/).

As well as serving as inputs to the document redaction app's review function, the 'review_file.csv' output can be modified insider or outside of the app. This gives you the flexibility to change redaction details outside of the app.

### Inside the app
You can now modify redaction review files directly in the app on the 'Review redactions' tab. Open the accordion 'View and edit review data' under the file input area. You can edit review file data cells here - press Enter to apply changes. You should see the effect on the current page if you click the 'Save changes on current page to file' button to the right.

### Outside the app
If you open up a 'review_file' csv output using a spreadsheet software program such as Microsoft Excel you can easily modify redaction properties. Open the file '[Partnership-Agreement-Toolkit_0_0_redacted.pdf_review_file_local.csv](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/merge_review_files/Partnership-Agreement-Toolkit_0_0.pdf_review_file_local.csv)', and you should see a spreadshet with just four suggested redactions (see below). The following instructions are for using Excel.

![Review file before](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/merge_review_files/img/review_file_before.PNG)

The first thing we can do is remove the first row - 'et' is suggested as a person, but is obviously not a genuine instance of personal information. Right click on the row number and select delete on this menu. Next, let's imagine that what the app identified as a 'phone number' was in fact another type of number and so we wanted to change the label. Simply click on the relevant label cells, let's change it to 'SECURITY_NUMBER'. You could also use 'Find & Select' -> 'Replace' from the top ribbon menu if you wanted to change a number of labels simultaneously.

How about we wanted to change the colour of the 'email address' entry on the redaction review tab of the redaction app? The colours in a review file are based on an RGB scale with three numbers ranging from 0-255. [You can find suitable colours here](https://rgbcolorpicker.com). Using this scale, if I wanted my review box to be pure blue, I can change the cell value to (0,0,255).

Imagine that a redaction box was slightly too small, and I didn't want to use the in-app options to change the size. In the review file csv, we can modify e.g. the ymin and ymax values for any box to increase the extent of the redaction box. For the 'email address' entry, let's decrease ymin by 5, and increase ymax by 5.

I have saved an output file following the above steps as '[Partnership-Agreement-Toolkit_0_0_redacted.pdf_review_file_local_mod.csv](https://github.com/seanpedrick-case/document_redaction_examples/blob/main/merge_review_files/outputs/Partnership-Agreement-Toolkit_0_0.pdf_review_file_local_mod.csv)' in the same folder that the original was found. Let's upload this file to the app along with the original pdf to see how the redactions look now.

![Review file after modification](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/merge_review_files/img/partnership_redactions_after.PNG)

We can see from the above that we have successfully removed a redaction box, changed labels, colours, and redaction box sizes.

## Merging redaction review files

Say you have run multiple redaction tasks on the same document, and you want to merge all of these redactions together. You could do this in your spreadsheet editor, but this could be fiddly especially if dealing with multiple review files or large numbers of redactions. The app has a feature to combine multiple review files together to create a 'merged' review file.

![Merging review files in the user interface](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/merge_review_files/img/merge_review_files_interface.PNG)

You can find this option at the bottom of the 'Redaction Settings' tab. Upload multiple review files here to get a single output 'merged' review_file. In the examples file, merging the 'review_file_custom.csv' and 'review_file_local.csv' files give you an output containing redaction boxes from both. This combined review file can then be uploaded into the review tab following the usual procedure.

![Merging review files outputs in spreadsheet](https://raw.githubusercontent.com/seanpedrick-case/document_redaction_examples/main/merge_review_files/img/merged_review_file_outputs_csv.PNG)

# Features for expert users/system administrators
This advanced user guide covers features that require system administration access or command-line usage. These options are not enabled by default but can be configured by your system administrator, and are not available to users who are just using the graphical user interface. These features are typically used by system administrators or advanced users who need more control over the redaction process.

## Using AWS Textract and Comprehend when not running in an AWS environment

AWS Textract and Comprehend give much better results for text extraction and document redaction than the local model options in the app. The most secure way to access them in the Redaction app is to run the app in a secure AWS environment with relevant permissions. Alternatively, you could run the app on your own system while logged in to AWS SSO with relevant permissions.

However, it is possible to access these services directly via API from outside an AWS environment by creating IAM users and access keys with relevant permissions to access AWS Textract and Comprehend services. Please check with your IT and data security teams that this approach is acceptable for your data before trying the following approaches.

To do the following, in your AWS environment you will need to create a new user with permissions for "textract:AnalyzeDocument", "textract:DetectDocumentText", and "comprehend:DetectPiiEntities". Under security credentials, create new access keys - note down the access key and secret key.

### Direct access by passing AWS access keys through app
The Redaction Settings tab now has boxes for entering the AWS access key and secret key. If you paste the relevant keys into these boxes before performing redaction, you should be able to use these services in the app.

### Picking up AWS access keys through an .env file
The app also has the capability of picking up AWS access key details through a .env file located in a '/config/aws_config.env' file (default), or alternative .env file location specified by the environment variable AWS_CONFIG_PATH. The env file should look like the following with just two lines:

AWS_ACCESS_KEY= your-access-key
AWS_SECRET_KEY= your-secret-key

The app should then pick up these keys when trying to access the AWS Textract and Comprehend services during redaction.

## Advanced OCR options

The app supports advanced OCR options that combine multiple OCR engines for improved accuracy. These options are not enabled by default but can be configured by changing the app_config.env file in your '/config' folder, or system environment variables in your system.

### Available OCR models

- **Tesseract** (default): The standard OCR engine that works well for most documents. Provides good word-level bounding box accuracy.
- **PaddleOCR**: More accurate for whole line text extraction, but word-level bounding boxes may be less precise. Best for documents with clear, well-formatted text.
- **Hybrid-paddle**: Combines Tesseract and PaddleOCR - uses Tesseract for initial extraction, then PaddleOCR for re-extraction of low-confidence text regions.
- **Hybrid-vlm**: Combines Tesseract with Vision Language Models (VLM) - uses Tesseract for initial extraction, then a VLM model (default: Dots.OCR) for re-extraction of low-confidence text.
- **Hybrid-paddle-vlm**: Combines PaddleOCR with Vision Language Models - uses PaddleOCR first, then a VLM model for low-confidence regions.

### Enabling advanced OCR options

To enable these options, you need to modify the app_config.env file in your '/config' folder and set the following environment variables:

**Basic OCR model selection:**
```
SHOW_LOCAL_OCR_MODEL_OPTIONS = "True"
```

**To enable PaddleOCR options (paddle, hybrid-paddle):**
```
SHOW_PADDLE_MODEL_OPTIONS = "True"
```

**To enable Vision Language Model options (hybrid-vlm, hybrid-paddle-vlm):**
```
SHOW_VLM_MODEL_OPTIONS = "True"
```

Once enabled, users will see a "Change default local OCR model" section in the redaction settings where they can choose between the available models based on what has been enabled.

### OCR configuration parameters

The following parameters can be configured by your system administrator to fine-tune OCR behavior:

#### Hybrid OCR settings

- **SHOW_HYBRID_MODELS** (default: False): If enabled, hybrid OCR options will be shown in the UI.
- **HYBRID_OCR_CONFIDENCE_THRESHOLD** (default: 80): Tesseract confidence score below which the secondary OCR engine (PaddleOCR or VLM) will be used for re-extraction. Lower values mean more text will be re-extracted.
- **HYBRID_OCR_PADDING** (default: 1): Padding (in pixels) added to word bounding boxes before re-extraction with the secondary engine.
- **SAVE_EXAMPLE_HYBRID_IMAGES** (default: False): If enabled, saves comparison images showing Tesseract vs. secondary engine results when using hybrid modes.
- **SAVE_PAGE_OCR_VISUALISATIONS** (default: False): If enabled, saves images with detected bounding boxes overlaid for debugging purposes.

#### Tesseract settings

- **TESSERACT_SEGMENTATION_LEVEL** (default: 11): Tesseract PSM (Page Segmentation Mode) level. Valid values are 0-13. Higher values provide more detailed segmentation but may be slower.

#### PaddleOCR settings

- **SHOW_PADDLE_MODEL_OPTIONS** (default: False): If enabled, PaddleOCR options will be shown in the UI.
- **PADDLE_USE_TEXTLINE_ORIENTATION** (default: False): If enabled, PaddleOCR will detect and correct text line orientation.
- **PADDLE_DET_DB_UNCLIP_RATIO** (default: 1.2): Controls the expansion ratio of detected text regions. Higher values expand the detection area more.
- **CONVERT_LINE_TO_WORD_LEVEL** (default: False): If enabled, converts PaddleOCR line-level results to word-level for better precision in bounding boxes (not perfect, but pretty good).
- **LOAD_PADDLE_AT_STARTUP** (default: False): If enabled, loads the PaddleOCR model when the application starts, reducing latency for first use but increasing startup time.

#### Image preprocessing

- **PREPROCESS_LOCAL_OCR_IMAGES** (default: True): If enabled, images are preprocessed before OCR. This can improve accuracy but may slow down processing.
- **SAVE_PREPROCESS_IMAGES** (default: False): If enabled, saves the preprocessed images for debugging purposes.

#### Vision Language Model (VLM) settings

When VLM options are enabled, the following settings are available:

- **SHOW_VLM_MODEL_OPTIONS** (default: False): If enabled, VLM options will be shown in the UI.
- **SELECTED_MODEL** (default: "Dots.OCR"): The VLM model to use. Options include: "Nanonets-OCR2-3B", "Dots.OCR", "Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct", "PaddleOCR-VL". Generally, the Qwen3-VL-8B-Instruct model is the most accurate, and vlm/inference server inference is based on using this model, but is also the slowest. Qwen3-VL-4B-Instruct can also work quite well on easier documents.
- **MAX_SPACES_GPU_RUN_TIME** (default: 60): Maximum seconds to run GPU operations on Hugging Face Spaces.
- **MAX_NEW_TOKENS** (default: 30): Maximum number of tokens to generate for VLM responses.
- **MAX_INPUT_TOKEN_LENGTH** (default: 4096): Maximum number of tokens that can be input to the VLM.
- **VLM_MAX_IMAGE_SIZE** (default: 1000000): Maximum total pixels (width × height) for images. Larger images are resized while maintaining aspect ratio.
- **VLM_MAX_DPI** (default: 300.0): Maximum DPI for images. Higher DPI images are resized accordingly.
- **USE_FLASH_ATTENTION** (default: False): If enabled, uses flash attention for improved VLM performance.
- **SAVE_VLM_INPUT_IMAGES** (default: False): If enabled, saves input images sent to VLM for debugging.

#### General settings

- **MODEL_CACHE_PATH** (default: "./model_cache"): Directory where OCR models are cached.
- **OVERWRITE_EXISTING_OCR_RESULTS** (default: False): If enabled, always creates new OCR results instead of loading from existing JSON files.

### Using an alternative OCR model

If the SHOW_LOCAL_OCR_MODEL_OPTIONS, SHOW_PADDLE_MODEL_OPTIONS, and SHOW_INFERENCE_SERVER_OPTIONS are set to 'True' in your app_config.env file, you should see the following options available under 'Change default redaction settings...' on the front tab. The different OCR options can be used in different contexts.

- **Tesseract (option 'tesseract')**: Best for documents with clear, well-formatted text, providing a good balance of speed and accuracy with precise word-level bounding boxes. But struggles a lot with handwriting or 'noisy' documents (e.g. scanned documents).
- **PaddleOCR (option 'paddle')**: More powerful than Tesseract, but slower. Does a decent job with unclear typed text on scanned documents. Also, bounding boxes may not all be accurate as they will be calculated from the line-level bounding boxes produced by Paddle after analysis.
- **VLM (option 'vlm')**: Recommended for use with the Qwen-3-VL 8B model (can set this with the SELECTED_MODEL environment variable in config.py). This model is extremely good at identifying difficult to read handwriting and noisy documents. However, it is much slower than the above options.
Other models are available as you can see in the tools/run_vlm.py code file. This will conduct inference with the transformers package, and quantise with bitsandbytes if the QUANTISE_VLM_MODELS environment variable is set to True. Inference with this package is *much* slower than with e.g. llama.cpp or vllm servers, which can be used with the inference-server options described below.
- **Inference server (option 'inference-server')**: This can be used with OpenAI compatible API endpoints, for example [llama-cpp using llama-server](https://github.com/ggml-org/llama.cpp), or [vllm](https://docs.vllm.ai/en/stable). Both of these options will be much faster for inference than the VLM 'in-app' model calls described above, and produce results of a similar quality, but you will need to be able to set up the server separately.

#### Hybrid options

If the SHOW_HYBRID_MODELS environment variable is set to 'True' in your app_config.env file, you will see the hybrid model options available. The hybrid models call a smaller model (paddleOCR) to first identify bounding box position and text, and then pass text sections with low confidence to a more performant model (served in app or via an inference server such as llama.cpp or vllm) to suggest for replacement. **Note:** I have not found that the results from this analysis is significantly better than that from e.g. Paddle or VLM/inference server analysis alone (particularly when using Qwen 3 VL), but are provided for comparison.

- **Hybrid-paddle-vlm**: This uses PaddleOCR's line-level detection with a VLM's advanced recognition capabilities. PaddleOCR is better at identifying bounding boxes for difficult documents, and so this is probably the most usable of the three options, if you can get both Paddle and the VLM model working in the same environment.
- **Hybrid-paddle-inference-server**: This uses PaddleOCR's line-level detection with an inference server's advanced recognition capabilities. This is the same as the Hybrid-paddle-vlm option, but uses an inference server instead of a VLM model. This allows for the use of GGUF or AWQ/GPTQ quantised models via llama.cpp or vllm servers.

### Inference server options

If using a local inference server, I would suggest using (llama.cpp)[https://github.com/ggml-org/llama.cpp] as it is much faster than transformers/torch inference, and it will offload to cpu/ram automatically rather than failing as vllm tends to do. Here is the run command I use for my llama server locally ion a wsl or linux environment) to get deterministic results (need at least 16GB of VRAM to run with all gpu layers assigned to your graphics card to use the following model):

```
llama-server \
    -hf unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:UD-Q4_K_XL \
    --n-gpu-layers 99 \
    --jinja \
    --temp 0 \
    --top-k 1 \
    --top-p 1 \
    --min-p 1 \
    --frequency-penalty 1 \
    --presence-penalty 1 \
    --flash-attn on \
    --ctx-size 8192 \
    --host 0.0.0.0 \
    --port 7862 \
    --image-min-tokens 1600 \
    --image-max-tokens 2301 \
    --no-warmup \
    --n-cpu-moe 13
```

If running llama.cpp on the same computer as the doc redaction app, you can then set the following variable in config/app_config.env to run:

```
SHOW_INFERENCE_SERVER_OPTIONS=True
INFERENCE_SERVER_API_URL=http://localhost:7862
```

The above setup with host = 0.0.0.0 allows you to access this server from other computers in your home network. Find your internal ip for the computer hosting llama server (e.g. using ipconfig in Windows), and then replace 'localhost' in the above variable with this value.

### Identifying people and signatures with VLMs

If VLM or inference server options are enabled, you can also use the VLM to identify photos of people's faces and signatures in the document, and redact them accordingly.

On the 'Redaction Settings' tab, select the CUSTOM_VLM_PERSON and CUSTOM_VLM_SIGNATURE entities. When you conduct an OCR task with the VLM or inference server, it will identify the bounding boxes for photos of people's faces and signatures in the document, and redact them accordingly if a redaction option is selected.


## Command Line Interface (CLI)

The app includes a comprehensive command-line interface (`cli_redact.py`) that allows you to perform redaction, deduplication, and AWS Textract operations directly from the terminal. This is particularly useful for batch processing, automation, and integration with other systems.

### Getting started with the CLI

To use the CLI, you need to:

1. Open a terminal window
2. Navigate to the app folder containing `cli_redact.py`
3. Activate your virtual environment (conda or venv)
4. Run commands using `python cli_redact.py` followed by your options

### Basic CLI syntax

```bash
python cli_redact.py --task [redact|deduplicate|textract] --input_file [file_path] [additional_options]
```

### Redaction examples

**Basic PDF redaction with default settings:**
```bash
python cli_redact.py --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf
```

**Extract text only (no redaction) with whole page redaction:**
```bash
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --redact_whole_page_file example_data/partnership_toolkit_redact_some_pages.csv --pii_detector None
```

**Redact with custom entities and allow list:**
```bash
python cli_redact.py --input_file example_data/graduate-job-example-cover-letter.pdf --allow_list_file example_data/test_allow_list_graduate.csv --local_redact_entities TITLES PERSON DATE_TIME
```

**Redact with fuzzy matching and custom deny list:**
```bash
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --deny_list_file example_data/Partnership-Agreement-Toolkit_test_deny_list_para_single_spell.csv --local_redact_entities CUSTOM_FUZZY --page_min 1 --page_max 3 --fuzzy_mistakes 3
```

**Redact with AWS services:**
```bash
python cli_redact.py --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf --ocr_method "AWS Textract" --pii_detector "AWS Comprehend"
```

**Redact specific pages with signature extraction:**
```bash
python cli_redact.py --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --page_min 6 --page_max 7 --ocr_method "AWS Textract" --handwrite_signature_extraction "Extract handwriting" "Extract signatures"
```

### Tabular data redaction

**Anonymize CSV file with specific columns:**
```bash
python cli_redact.py --input_file example_data/combined_case_notes.csv --text_columns "Case Note" "Client" --anon_strategy replace_redacted
```

**Anonymize Excel file:**
```bash
python cli_redact.py --input_file example_data/combined_case_notes.xlsx --text_columns "Case Note" "Client" --excel_sheets combined_case_notes --anon_strategy redact
```

**Anonymize Word document:**
```bash
python cli_redact.py --input_file "example_data/Bold minimalist professional cover letter.docx" --anon_strategy replace_redacted
```

### Duplicate detection

**Find duplicate pages in OCR files:**
```bash
python cli_redact.py --task deduplicate --input_file example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv --duplicate_type pages --similarity_threshold 0.95
```

**Find duplicates at line level:**
```bash
python cli_redact.py --task deduplicate --input_file example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv --duplicate_type pages --similarity_threshold 0.95 --combine_pages False --min_word_count 3
```

**Find duplicate rows in tabular data:**
```bash
python cli_redact.py --task deduplicate --input_file example_data/Lambeth_2030-Our_Future_Our_Lambeth.pdf.csv --duplicate_type tabular --text_columns "text" --similarity_threshold 0.95
```

### AWS Textract operations

**Submit document for analysis:**
```bash
python cli_redact.py --task textract --textract_action submit --input_file example_data/example_of_emails_sent_to_a_professor_before_applying.pdf
```

**Submit with signature extraction:**
```bash
python cli_redact.py --task textract --textract_action submit --input_file example_data/Partnership-Agreement-Toolkit_0_0.pdf --extract_signatures
```

**Retrieve results by job ID:**
```bash
python cli_redact.py --task textract --textract_action retrieve --job_id 12345678-1234-1234-1234-123456789012
```

**List recent jobs:**
```bash
python cli_redact.py --task textract --textract_action list
```

### Common CLI options

#### General options

- `--task`: Choose between "redact", "deduplicate", or "textract"
- `--input_file`: Path to input file(s) - can specify multiple files separated by spaces
- `--output_dir`: Directory for output files (default: output/)
- `--input_dir`: Directory for input files (default: input/)
- `--language`: Language of document content (e.g., "en", "es", "fr")
- `--username`: Username for session tracking
- `--pii_detector`: Choose PII detection method ("Local", "AWS Comprehend", or "None")
- `--local_redact_entities`: Specify local entities to redact (space-separated list)
- `--aws_redact_entities`: Specify AWS Comprehend entities to redact (space-separated list)
- `--aws_access_key` / `--aws_secret_key`: AWS credentials for cloud services
- `--aws_region`: AWS region for cloud services
- `--s3_bucket`: S3 bucket name for cloud operations
- `--cost_code`: Cost code for tracking usage

#### PDF/Image redaction options

- `--ocr_method`: Choose text extraction method ("AWS Textract", "Local OCR", or "Local text")
- `--chosen_local_ocr_model`: Local OCR model to use (e.g., "tesseract", "paddle", "hybrid-paddle", "hybrid-vlm")
- `--page_min` / `--page_max`: Process only specific page range (0 for max means all pages)
- `--images_dpi`: DPI for image processing (default: 300.0)
- `--preprocess_local_ocr_images`: Preprocess images before OCR (True/False)
- `--compress_redacted_pdf`: Compress the final redacted PDF (True/False)
- `--return_pdf_end_of_redaction`: Return PDF at end of redaction process (True/False)
- `--allow_list_file` / `--deny_list_file`: Paths to custom allow/deny list CSV files
- `--redact_whole_page_file`: Path to CSV file listing pages to redact completely
- `--handwrite_signature_extraction`: Handwriting and signature extraction options for Textract ("Extract handwriting", "Extract signatures")
- `--extract_forms`: Extract forms during Textract analysis (flag)
- `--extract_tables`: Extract tables during Textract analysis (flag)
- `--extract_layout`: Extract layout during Textract analysis (flag)

#### Tabular/Word anonymization options

- `--anon_strategy`: Anonymization strategy (e.g., "redact", "redact completely", "replace_redacted", "encrypt", "hash")
- `--text_columns`: List of column names to anonymize (space-separated)
- `--excel_sheets`: Specific Excel sheet names to process (space-separated)
- `--fuzzy_mistakes`: Number of spelling mistakes allowed in fuzzy matching (default: 1)
- `--match_fuzzy_whole_phrase_bool`: Match fuzzy whole phrase (True/False)
- `--do_initial_clean`: Perform initial text cleaning for tabular data (True/False)

#### Duplicate detection options

- `--duplicate_type`: Type of duplicate detection ("pages" for OCR files or "tabular" for CSV/Excel)
- `--similarity_threshold`: Similarity threshold (0-1) to consider content as duplicates (default: 0.95)
- `--min_word_count`: Minimum word count for text to be considered (default: 10)
- `--min_consecutive_pages`: Minimum number of consecutive pages to consider as a match (default: 1)
- `--greedy_match`: Use greedy matching strategy for consecutive pages (True/False)
- `--combine_pages`: Combine text from same page number within a file (True/False)
- `--remove_duplicate_rows`: Remove duplicate rows from output (True/False)

#### Textract batch operations options

- `--textract_action`: Action to perform ("submit", "retrieve", or "list")
- `--job_id`: Textract job ID for retrieve action
- `--extract_signatures`: Extract signatures during Textract analysis (flag)
- `--textract_bucket`: S3 bucket name for Textract operations
- `--poll_interval`: Polling interval in seconds for job status (default: 30)
- `--max_poll_attempts`: Maximum polling attempts before timeout (default: 120)

### Output files

The CLI generates the same output files as the GUI:
- `...redacted.pdf`: Final redacted document
- `...redactions_for_review.pdf`: Document with redaction boxes for review
- `...review_file.csv`: Detailed redaction information
- `...ocr_results.csv`: Extracted text results
- `..._textract.json`: AWS Textract results (if applicable)

For more advanced options and configuration, refer to the help text by running:
```bash
python cli_redact.py --help
```