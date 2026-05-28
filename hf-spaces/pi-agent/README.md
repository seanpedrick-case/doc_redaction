---
title: Pi Agentic Document Redaction
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
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

## Development

This Space is synced from the [doc_redaction monorepo](https://github.com/seanpedrick-case/doc_redaction) on pushes to **`dev`** (see `.github/workflows/sync-pi-agent-space.yml`). Space: [seanpedrickcase/agentic_document_redaction](https://huggingface.co/spaces/seanpedrickcase/agentic_document_redaction).
