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
Redact personal information from documents, open text, or xlsx/csv tabular data. See the 'Redaction settings' to change various settings such as which types of information to redact (e.g. people, places), or terms to exclude from redaction.

WARNING: This is a beta product. It is not 100% accurate, and it will miss some personal information. It is essential that all outputs are checked **by a human** to ensure that all personal information has been removed.

Other redaction entities are possible to include in this app easily, especially country-specific entities. If you want to use these, clone the repo locally and add entity names from [this link](https://microsoft.github.io/presidio/supported_entities/) to the 'full_entity_list' variable in app.py.