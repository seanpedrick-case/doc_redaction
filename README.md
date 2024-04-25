---
title: Document redaction
emoji: 😎
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# Introduction 
Redact PDF files using image-based OCR or direct text analysis from pdfminer.six. Personal information identification performed using Microsoft Presidio.

Take an image-based or text-based PDF document and redact any personal information. 'Image analysis' will convert PDF pages to image and the identify text via OCR methods before redaction. 'Text analysis' will analyse only selectable text that exists in the original PDF before redaction. Choose 'Image analysis' if you are not sure of the type of PDF document you are working with.

WARNING: This is a beta product. It is not 100% accurate, and it will miss some personal information. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed.

