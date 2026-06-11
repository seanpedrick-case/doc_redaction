---
title: Agentic Document Redaction
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: agent-redact/pi/gradio_app.py
pinned: false
license: agpl-3.0
short_description: Agentic interface to redact PDF documents
---

# Pi agent — agentic document redaction

Orchestrate document redaction with **[Pi](https://github.com/earendil-works/pi)** and **Google Gemini**. Heavy redaction runs on a separate **private [doc_redaction](https://huggingface.co/spaces/seanpedrickcase/document_redaction)** Hugging Face Space (simple text extraction + Local PII).

## Before you start

1. **Gemini API key** — paste in **Agent backend** → **Apply backend** (session-only; not stored on disk).
2. **HF token** — Space admin should set `HF_TOKEN` under **Settings → Secrets** so this Space can call the private redaction backend. Users may optionally override per session in the UI.

## Limitations

- **No face or signature VLM** — text-layer PII only via Local spaCy/Presidio on the remote Space.
- **No Pass 2 VLM** on this deployment.
- **Ephemeral storage** — download deliverables from **Workspace output files** before the Space restarts.
- **Human review** — outputs are not guaranteed complete; review redacted PDFs before release.

## Defaults

| Setting | Value |
|---------|--------|
| Pi LLM | Gemini (`gemini-flash-latest` default) |
| Redaction backend | `https://seanpedrickcase-document-redaction.hf.space` |
| Text extraction | `Local model - selectable text` |
| PII detection | `Local` |

## Examples

Two sample PDFs load in **Redaction task** → **Try an example** (same demos as the main doc_redaction app). Examples are **on by default**; set Space variable `PI_GRADIO_SHOW_EXAMPLES=false` to hide them. (`SHOW_PI_EXAMPLES` is also accepted.)

If examples do not appear, the UI shows a short status message (usually missing PDFs in the image — rebuild after a successful sync with LFS materialization).

## Development

This Space is synced from the [doc_redaction monorepo](https://github.com/seanpedrick-case/doc_redaction) on pushes to **`dev`** (see `.github/workflows/sync-pi-agent-space.yml`). Space: [seanpedrickcase/agentic_document_redaction](https://huggingface.co/spaces/seanpedrickcase/agentic_document_redaction).
