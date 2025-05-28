# Admin Management Guide

## Introduction

This guide provides an overview of how to configure the application using environment variables. The application loads configurations using `os.environ.get()`. It first attempts to load variables from the file specified by `APP_CONFIG_PATH` (which defaults to `config/app_config.env`). If `AWS_CONFIG_PATH` is also set (e.g., to `config/aws_config.env`), variables are loaded from that file as well. Environment variables set directly in the system will always take precedence over those defined in these `.env` files.

## App Configuration File

This section details variables related to the main application configuration file.

*   **`APP_CONFIG_PATH`**
    *   **Description:** Specifies the path to the application configuration `.env` file. This file contains various settings that control the application's behavior.
    *   **Default Value:** `config/app_config.env`
    *   **Configuration:** Set as an environment variable directly. This variable defines where to load other application configurations, so it cannot be set within `config/app_config.env` itself.

## AWS Options

This section covers configurations related to AWS services used by the application.

*   **`AWS_CONFIG_PATH`**
    *   **Description:** Specifies the path to the AWS configuration `.env` file. This file is intended to store AWS credentials and specific settings.
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly. This variable defines an additional source for AWS-specific configurations.

*   **`RUN_AWS_FUNCTIONS`**
    *   **Description:** Enables or disables AWS-specific functionalities within the application. Set to `"1"` to enable and `"0"` to disable.
    *   **Default Value:** `"0"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`AWS_REGION`**
    *   **Description:** Defines the AWS region where services like S3, Cognito, and Textract are located.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured).

*   **`AWS_CLIENT_ID`**
    *   **Description:** The client ID for AWS Cognito, used for user authentication.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured).

*   **`AWS_CLIENT_SECRET`**
    *   **Description:** The client secret for AWS Cognito, used in conjunction with the client ID for authentication.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured).

*   **`AWS_USER_POOL_ID`**
    *   **Description:** The user pool ID for AWS Cognito, identifying the user directory.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured).

*   **`AWS_ACCESS_KEY`**
    *   **Description:** The AWS access key ID for programmatic access to AWS services.
    *   **Default Value:** `''` (Note: Often found in the environment or AWS credentials file.)
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured). It's also commonly configured via shared AWS credentials files or IAM roles.

*   **`AWS_SECRET_KEY`**
    *   **Description:** The AWS secret access key corresponding to the AWS access key ID.
    *   **Default Value:** `''` (Note: Often found in the environment or AWS credentials file.)
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured). It's also commonly configured via shared AWS credentials files or IAM roles.

*   **`DOCUMENT_REDACTION_BUCKET`**
    *   **Description:** The name of the S3 bucket used for storing documents related to the redaction process.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/aws_config.env` (if `AWS_CONFIG_PATH` is configured).

*   **`CUSTOM_HEADER`**
    *   **Description:** Specifies a custom header name to be included in requests, often used for services like AWS CloudFront.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`CUSTOM_HEADER_VALUE`**
    *   **Description:** The value for the custom header specified by `CUSTOM_HEADER`.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

## Image Options

Settings related to image processing within the application.

*   **`IMAGES_DPI`**
    *   **Description:** Dots Per Inch (DPI) setting for image processing, affecting the resolution and quality of processed images.
    *   **Default Value:** `'300.0'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`LOAD_TRUNCATED_IMAGES`**
    *   **Description:** Controls whether the application attempts to load truncated images. Set to `'True'` to enable.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`MAX_IMAGE_PIXELS`**
    *   **Description:** Sets the maximum number of pixels for an image that the application will process. Leave blank for no limit. This can help prevent issues with very large images.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

## File I/O Options

Configuration for input and output file handling.

*   **`SESSION_OUTPUT_FOLDER`**
    *   **Description:** If set to `'True'`, the application will save output and input files into session-specific subfolders, helping to organize files from different user sessions.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`GRADIO_OUTPUT_FOLDER`** (aliased as `OUTPUT_FOLDER`)
    *   **Description:** Specifies the default output folder for files generated by Gradio components. Can be set to "TEMP" to use a temporary directory.
    *   **Default Value:** `'output/'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`GRADIO_INPUT_FOLDER`** (aliased as `INPUT_FOLDER`)
    *   **Description:** Specifies the default input folder for files used by Gradio components. Can be set to "TEMP" to use a temporary directory.
    *   **Default Value:** `'input/'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

## Logging Options

Settings for configuring application logging, including log formats and storage locations.

*   **`SAVE_LOGS_TO_CSV`**
    *   **Description:** Enables or disables saving logs to CSV files. Set to `'True'` to enable.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`USE_LOG_SUBFOLDERS`**
    *   **Description:** If enabled (`'True'`), logs will be stored in subfolders based on date and hostname, aiding in log organization.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`FEEDBACK_LOGS_FOLDER`**
    *   **Description:** Specifies the base folder for storing feedback logs. If `USE_LOG_SUBFOLDERS` is true, date/hostname subfolders will be created within this folder.
    *   **Default Value:** `'feedback/'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`ACCESS_LOGS_FOLDER`**
    *   **Description:** Specifies the base folder for storing access logs. If `USE_LOG_SUBFOLDERS` is true, date/hostname subfolders will be created within this folder.
    *   **Default Value:** `'logs/'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`USAGE_LOGS_FOLDER`**
    *   **Description:** Specifies the base folder for storing usage logs. If `USE_LOG_SUBFOLDERS` is true, date/hostname subfolders will be created within this folder.
    *   **Default Value:** `'usage/'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`DISPLAY_FILE_NAMES_IN_LOGS`**
    *   **Description:** If set to `'True'`, file names will be included in the log entries.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`CSV_ACCESS_LOG_HEADERS`**
    *   **Description:** Defines custom headers for CSV access logs. If left blank, component labels will be used as headers.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`CSV_FEEDBACK_LOG_HEADERS`**
    *   **Description:** Defines custom headers for CSV feedback logs. If left blank, component labels will be used as headers.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`CSV_USAGE_LOG_HEADERS`**
    *   **Description:** Defines custom headers for CSV usage logs.
    *   **Default Value:** A predefined list of header names. Refer to `tools/config.py` for the complete list.
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`SAVE_LOGS_TO_DYNAMODB`**
    *   **Description:** Enables or disables saving logs to AWS DynamoDB. Set to `'True'` to enable. Requires appropriate AWS setup.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`ACCESS_LOG_DYNAMODB_TABLE_NAME`**
    *   **Description:** The name of the DynamoDB table used for storing access logs.
    *   **Default Value:** `'redaction_access_log'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`DYNAMODB_ACCESS_LOG_HEADERS`**
    *   **Description:** Specifies the headers (attributes) for the DynamoDB access log table.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`FEEDBACK_LOG_DYNAMODB_TABLE_NAME`**
    *   **Description:** The name of the DynamoDB table used for storing feedback logs.
    *   **Default Value:** `'redaction_feedback'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`DYNAMODB_FEEDBACK_LOG_HEADERS`**
    *   **Description:** Specifies the headers (attributes) for the DynamoDB feedback log table.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`USAGE_LOG_DYNAMODB_TABLE_NAME`**
    *   **Description:** The name of the DynamoDB table used for storing usage logs.
    *   **Default Value:** `'redaction_usage'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`DYNAMODB_USAGE_LOG_HEADERS`**
    *   **Description:** Specifies the headers (attributes) for the DynamoDB usage log table.
    *   **Default Value:** `''`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`LOGGING`**
    *   **Description:** Enables or disables general console logging. Set to `'True'` to enable.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`LOG_FILE_NAME`**
    *   **Description:** Specifies the name for the CSV log file if `SAVE_LOGS_TO_CSV` is enabled.
    *   **Default Value:** `'log.csv'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

## Redaction Options

Configurations related to the text redaction process, including PII detection models and external tool paths.

*   **`TESSERACT_FOLDER`**
    *   **Description:** Path to the local Tesseract OCR installation folder. Required if using the local Tesseract OCR model for text extraction.
    *   **Default Value:** `""` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`POPPLER_FOLDER`**
    *   **Description:** Path to the local Poppler installation's `bin` folder. Poppler is used for PDF processing.
    *   **Default Value:** `""` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`SELECTABLE_TEXT_EXTRACT_OPTION`**
    *   **Description:** Display name in the UI for the text extraction method that processes selectable text directly from PDFs.
    *   **Default Value:** `"Local model - selectable text"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`TESSERACT_TEXT_EXTRACT_OPTION`**
    *   **Description:** Display name in the UI for the text extraction method using local Tesseract OCR (for PDFs without selectable text).
    *   **Default Value:** `"Local OCR model - PDFs without selectable text"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`TEXTRACT_TEXT_EXTRACT_OPTION`**
    *   **Description:** Display name in the UI for the text extraction method using AWS Textract service.
    *   **Default Value:** `"AWS Textract service - all PDF types"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`NO_REDACTION_PII_OPTION`**
    *   **Description:** Display name in the UI for the option to only extract text without performing any PII detection or redaction.
    *   **Default Value:** `"Only extract text (no redaction)"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`LOCAL_PII_OPTION`**
    *   **Description:** Display name in the UI for the PII detection method using a local model.
    *   **Default Value:** `"Local"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`AWS_PII_OPTION`**
    *   **Description:** Display name in the UI for the PII detection method using AWS Comprehend.
    *   **Default Value:** `"AWS Comprehend"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`SHOW_LOCAL_TEXT_EXTRACTION_OPTIONS`**
    *   **Description:** Controls whether local text extraction options (selectable text, Tesseract) are shown in the UI. Set to `'True'` to show.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`SHOW_AWS_TEXT_EXTRACTION_OPTIONS`**
    *   **Description:** Controls whether AWS Textract text extraction option is shown in the UI. Set to `'True'` to show.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`DEFAULT_TEXT_EXTRACTION_MODEL`**
    *   **Description:** Sets the default text extraction model selected in the UI. Defaults to `TEXTRACT_TEXT_EXTRACT_OPTION` if AWS options are shown; otherwise, defaults to `SELECTABLE_TEXT_EXTRACT_OPTION`.
    *   **Default Value:** Value of `TEXTRACT_TEXT_EXTRACT_OPTION` if `SHOW_AWS_TEXT_EXTRACTION_OPTIONS` is True, else value of `SELECTABLE_TEXT_EXTRACT_OPTION`.
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`. Provide one of the text extraction option display names.

*   **`SHOW_LOCAL_PII_DETECTION_OPTIONS`**
    *   **Description:** Controls whether the local PII detection option is shown in the UI. Set to `'True'` to show.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`SHOW_AWS_PII_DETECTION_OPTIONS`**
    *   **Description:** Controls whether the AWS Comprehend PII detection option is shown in the UI. Set to `'True'` to show.
    *   **Default Value:** `'True'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`DEFAULT_PII_DETECTION_MODEL`**
    *   **Description:** Sets the default PII detection model selected in the UI. Defaults to `AWS_PII_OPTION` if AWS options are shown; otherwise, defaults to `LOCAL_PII_OPTION`.
    *   **Default Value:** Value of `AWS_PII_OPTION` if `SHOW_AWS_PII_DETECTION_OPTIONS` is True, else value of `LOCAL_PII_OPTION`.
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`. Provide one of the PII detection option display names.

*   **`CHOSEN_COMPREHEND_ENTITIES`**
    *   **Description:** A list of AWS Comprehend PII entity types to be redacted when using AWS Comprehend.
    *   **Default Value:** A predefined list of entity types. Refer to `tools/config.py` for the complete list.
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`. This should be a string representation of a Python list.

*   **`FULL_COMPREHEND_ENTITY_LIST`**
    *   **Description:** The complete list of PII entity types supported by AWS Comprehend that can be selected for redaction.
    *   **Default Value:** A predefined list of entity types. Refer to `tools/config.py` for the complete list.
    *   **Configuration:** This is typically an informational variable reflecting the capabilities of AWS Comprehend and is not meant to be changed by users directly affecting redaction behavior (use `CHOSEN_COMPREHEND_ENTITIES` for that). Set as an environment variable directly, or include in `config/app_config.env`.

*   **`CHOSEN_REDACT_ENTITIES`**
    *   **Description:** A list of local PII entity types to be redacted when using the local PII detection model.
    *   **Default Value:** A predefined list of entity types. Refer to `tools/config.py` for the complete list.
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`. This should be a string representation of a Python list.

*   **`FULL_ENTITY_LIST`**
    *   **Description:** The complete list of PII entity types supported by the local PII detection model that can be selected for redaction.
    *   **Default Value:** A predefined list of entity types. Refer to `tools/config.py` for the complete list.
    *   **Configuration:** This is typically an informational variable reflecting the capabilities of the local model and is not meant to be changed by users directly affecting redaction behavior (use `CHOSEN_REDACT_ENTITIES` for that). Set as an environment variable directly, or include in `config/app_config.env`.

*   **`PAGE_BREAK_VALUE`**
    *   **Description:** Defines a page count after which a function might restart. (Note: Currently not activated).
    *   **Default Value:** `'99999'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`MAX_TIME_VALUE`**
    *   **Description:** Specifies the maximum time (in arbitrary units, likely seconds or milliseconds depending on implementation) for a process before it might be timed out.
    *   **Default Value:** `'999999'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`CUSTOM_BOX_COLOUR`**
    *   **Description:** Allows specifying a custom color for the redaction boxes drawn on documents (e.g., "grey", "red", "#FF0000"). If empty, a default color is used.
    *   **Default Value:** `""` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`REDACTION_LANGUAGE`**
    *   **Description:** Specifies the language for redaction processing. Currently, only "en" (English) is supported.
    *   **Default Value:** `"en"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`RETURN_PDF_END_OF_REDACTION`**
    *   **Description:** If set to `'True'`, the application will return a PDF document at the end of the redaction task.
    *   **Default Value:** `"True"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`COMPRESS_REDACTED_PDF`**
    *   **Description:** If set to `'True'`, the redacted PDF output will be compressed. This can reduce file size but may cause issues on systems with low memory.
    *   **Default Value:** `"False"`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

## App Run Options

General runtime configurations for the application.

*   **`TLDEXTRACT_CACHE`**
    *   **Description:** Path to the cache file used by the `tldextract` library, which helps in accurately extracting top-level domains (TLDs) from URLs.
    *   **Default Value:** `'tld/.tld_set_snapshot'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`COGNITO_AUTH`**
    *   **Description:** Enables or disables AWS Cognito authentication for the application. Set to `'1'` to enable.
    *   **Default Value:** `'0'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`RUN_DIRECT_MODE`**
    *   **Description:** If set to `'1'`, runs the application in a "direct mode", which might alter certain behaviors (e.g., UI elements, processing flow).
    *   **Default Value:** `'0'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`MAX_QUEUE_SIZE`**
    *   **Description:** The maximum number of requests that can be queued in the Gradio interface.
    *   **Default Value:** `'5'` (integer)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`MAX_FILE_SIZE`**
    *   **Description:** Maximum file size allowed for uploads (e.g., "250mb", "1gb").
    *   **Default Value:** `'250mb'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`GRADIO_SERVER_PORT`**
    *   **Description:** The network port on which the Gradio server will listen.
    *   **Default Value:** `'7860'` (integer)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`ROOT_PATH`**
    *   **Description:** The root path for the application, useful if running behind a reverse proxy (e.g., `/app`).
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`DEFAULT_CONCURRENCY_LIMIT`**
    *   **Description:** The default concurrency limit for Gradio event handlers, controlling how many requests can be processed simultaneously.
    *   **Default Value:** `'3'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`GET_DEFAULT_ALLOW_LIST`**
    *   **Description:** If set, enables the use of a default allow list for user access or specific functionalities. The exact behavior depends on application logic.
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`ALLOW_LIST_PATH`**
    *   **Description:** Path to a local CSV file containing an allow list (e.g., `config/default_allow_list.csv`).
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`S3_ALLOW_LIST_PATH`**
    *   **Description:** Path to an allow list CSV file stored in an S3 bucket (e.g., `default_allow_list.csv`). Requires `DOCUMENT_REDACTION_BUCKET` to be set.
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`FILE_INPUT_HEIGHT`**
    *   **Description:** Sets the height (in pixels or other CSS unit) of the file input component in the Gradio UI.
    *   **Default Value:** `'200'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

## Cost Code Options

Settings related to tracking and applying cost codes for application usage.

*   **`SHOW_COSTS`**
    *   **Description:** If set to `'True'`, cost-related information will be displayed in the UI.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`GET_COST_CODES`**
    *   **Description:** Enables fetching and using cost codes within the application. Set to `'True'` to enable.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`DEFAULT_COST_CODE`**
    *   **Description:** Specifies a default cost code to be used if cost codes are enabled but none is selected by the user.
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`COST_CODES_PATH`**
    *   **Description:** Path to a local CSV file containing available cost codes (e.g., `config/COST_CENTRES.csv`).
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`S3_COST_CODES_PATH`**
    *   **Description:** Path to a cost codes CSV file stored in an S3 bucket (e.g., `COST_CENTRES.csv`). Requires `DOCUMENT_REDACTION_BUCKET` to be set.
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`ENFORCE_COST_CODES`**
    *   **Description:** If set to `'True'` and `GET_COST_CODES` is also enabled, makes the selection of a cost code mandatory for users.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

## Whole Document API Options

Configurations for features related to processing whole documents via APIs, particularly AWS Textract for large documents.

*   **`SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS`**
    *   **Description:** Controls whether UI options for whole document Textract calls are displayed. (Note: Mentioned as not currently implemented in the source).
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET`**
    *   **Description:** The S3 bucket used for input and output of whole document analysis with AWS Textract.
    *   **Default Value:** `''` (empty string)
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER`**
    *   **Description:** The subfolder within `TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET` where input documents for Textract analysis are placed.
    *   **Default Value:** `'input'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER`**
    *   **Description:** The subfolder within `TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET` where output results from Textract analysis are stored.
    *   **Default Value:** `'output'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`LOAD_PREVIOUS_TEXTRACT_JOBS_S3`**
    *   **Description:** If set to `'True'`, the application will attempt to load data from previous Textract jobs stored in S3.
    *   **Default Value:** `'False'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`TEXTRACT_JOBS_S3_LOC`**
    *   **Description:** The S3 subfolder (within `TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET`) where Textract job data (output) is stored.
    *   **Default Value:** `'output'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`TEXTRACT_JOBS_S3_INPUT_LOC`**
    *   **Description:** The S3 subfolder (within `TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET`) where Textract job input is stored.
    *   **Default Value:** `'input'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env` (or `config/aws_config.env` if `AWS_CONFIG_PATH` is configured).

*   **`TEXTRACT_JOBS_LOCAL_LOC`**
    *   **Description:** The local subfolder where Textract job data is stored if not using S3 or as a cache.
    *   **Default Value:** `'output'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.

*   **`DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS`**
    *   **Description:** Specifies the number of past days for which to display whole document Textract jobs in the UI.
    *   **Default Value:** `'7'`
    *   **Configuration:** Set as an environment variable directly, or include in `config/app_config.env`.
