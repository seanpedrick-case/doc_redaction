---
title: Document redaction
emoji: 😎
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: agpl-3.0
---

# Document redaction

Redact personal information from documents (pdf, images), open text, or tabular data (xlsx/csv/parquet). Documents/images can be redacted using 'Quick' image analysis that works fine for typed text, but not handwriting/signatures. On the Redaction settings tab, choose 'Complex image analysis' OCR using AWS Textract (if you are using AWS) to redact these more complex elements (this service has a cost, so please only use for more complex redaction tasks). Also see the 'Redaction settings' tab to choose which pages to redact, the type of information to redact (e.g. people, places), or terms to exclude from redaction.

NOTE: In testing the app seems to find about 60% of personal information on a given (typed) page of text. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed.

This app accepts a maximum file size of 50mb. Please consider giving feedback for the quality of the answers underneath the redact buttons when the option appears, this will help to improve the app.