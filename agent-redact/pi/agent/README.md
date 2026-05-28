# Pi agent config (Docker)

Runtime Pi config is **generated at container start** by [`agent-redact/pi/pi_agent_config.py`](../pi_agent_config.py) into `~/.pi/agent/models.json` and `~/.pi/agent/settings.json`.

Files in this folder (`settings.json`, `models.json`) are **templates/references** only — they are no longer bind-mounted into the container.

## LLM backends (Pi orchestration)

The Pi agent (chat + redaction orchestration) can use:

| Provider key | Label | Pi API | Auth |
|--------------|-------|--------|------|
| `llama-cpp` | Local (llama-cpp) | `openai-completions` | None (local llama-inference) |
| `google-gemini` | Gemini | `google-generative-ai` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `amazon-bedrock` | AWS Bedrock | `bedrock-converse-stream` | AWS SDK credentials (`AWS_ACCESS_KEY_ID`, etc.) |

This is separate from doc_redaction **Pass 2 VLM** (`{VLM_BASE_URL}` in redaction prompts), which still targets local llama-inference by default.

### Environment variables

Copy [`config/pi_agent.env.example`](../../../config/pi_agent.env.example) to `config/pi_agent.env` (gitignored) or set on the host before `docker compose up`:

| Variable | Purpose |
|----------|---------|
| `PI_DEFAULT_PROVIDER` | `llama-cpp` \| `google-gemini` \| `amazon-bedrock` |
| `PI_DEFAULT_MODEL` | Model id within provider |
| `PI_LLAMA_BASE_URL` | Local OpenAI-compatible URL (default `http://llama-inference:8080/v1`) |
| `PI_LLAMA_MODEL_ID` | Local model id |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini API key |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | Bedrock region |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | Bedrock credentials (when not using SSO) |
| `AWS_PROFILE` | Named profile for SSO / shared credentials file (**required for Pi Bedrock with SSO**) |
| `PI_AWS_PROFILE` | Alternative to `AWS_PROFILE`; also used to auto-select profile when only `~/.aws` is mounted |
| `RUN_AWS_FUNCTIONS` | When `True`, use the AWS default credential chain (SSO, profile, role) |
| `PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS` | When `True` with `RUN_AWS_FUNCTIONS`, prefer SSO/chain over static env keys (default `True`, same as main app) |
| `PI_MAX_PAGES` | Maximum PDF pages allowed per redaction upload (falls back to `MAX_PAGES` / `MAX_DOC_PAGES`, default `3000`) |

At startup, if only `GOOGLE_API_KEY` is set, it is mirrored to `GEMINI_API_KEY` for Pi.

### Gradio UI

Open **http://localhost:7862** → **Agent backend** accordion:

- Select provider and model
- Optionally enter Gemini / AWS credentials (**session-only** — not written to disk)
- Click **Apply backend** — regenerates config, restarts the Pi RPC subprocess, and starts a new session

Credential fields are cleared after apply.

## Local model id

After the llama.cpp service is healthy, confirm the model id:

```bash
curl http://localhost:8000/v1/models
```

If the returned `id` differs from `unsloth/Qwen3.6-27B-MTP-GGUF`, set `PI_LLAMA_MODEL_ID` in `config/pi_agent.env` or compose environment and restart `pi-agent`.

## In-container URLs for task prompts

When filling [`skills/doc-redaction-task-prompt/TASK_PROMPT_TEMPLATE.md`](../../../skills/doc-redaction-task-prompt/TASK_PROMPT_TEMPLATE.md) inside the Pi container, use:

| Placeholder | In-container value |
|-------------|-------------------|
| `{GRADIO_URL}` | `http://redaction-app-llama:7860` |
| `{VLM_BASE_URL}` | `http://llama-inference:8080` |
| `{INPUT_PATH}` | `/home/user/app/workspace/{session_hash}/{FILE_NAME}` (when `PI_SESSION_WORKSPACE=true`) |
| `{OUTPUT_BASE}` | `/home/user/app/workspace/{session_hash}/redact/{FILE_NAME}/` |

Host-side examples (`host.docker.internal`, `localhost:7861`) do not apply inside the compose network.

## Usage

Start the stack (27B profile):

```powershell
docker compose -f docker-compose_llama_agentic.yml --profile 27b_36 up -d --build
```

Interactive Pi TUI:

```powershell
docker compose -f docker-compose_llama_agentic.yml exec -it pi-agent pi
```

Gradio chat UI (browser):

Open **http://localhost:7862**. Use the **Redaction task** panel to upload a document, enter bullet-point requirements, and click **Start redaction task**. Pi receives the filled prompt from [`skills/Example prompt partnership.txt`](../../../skills/Example%20prompt%20partnership.txt) (file copied to `/home/user/app/workspace/`). The full prompt appears in the chat; Pi’s reply streams in the chat panel.

The UI also shows:

- **Agent backend** — switch between local, Gemini, and Bedrock
- **Chat** — streamed assistant text
- **Activity** — agent/turn lifecycle, compaction, auto-retry, tool start/end
- **Tool output** — live bash/read output from `tool_execution_update` / `tool_execution_end`
- **Thinking** — optional stream (`PI_GRADIO_SHOW_THINKING=true`)
- **Abort** — sends Pi RPC `abort` and cancels the in-flight Gradio handler
- **Workspace output files** — browse and download redaction artifacts

Optional env vars on `pi-agent`: `PI_GRADIO_SHOW_THINKING`, `PI_GRADIO_SHOW_TOOL_OUTPUT`, `PI_GRADIO_TOOL_OUTPUT_MAX`, `PI_GRADIO_ACTIVITY_MAX_LINES`.

Run the UI locally (outside Docker):

```powershell
cd agent-redact/pi
pip install -r ../requirements_pi_agent.txt
python pi_agent_config.py
python gradio_app.py
```

RPC mode (automation, no Gradio):

```powershell
docker compose -f docker-compose_llama_agentic.yml exec -T pi-agent pi --mode rpc
```

Skills are discovered from the repo mount at `/workspace/doc_redaction/skills/` (project skills). Use `/skill:doc-redaction-app` etc., or read paths under `skills/` relative to the working directory.

Sessions persist in the **`pi-agent-sessions`** Docker volume at **`~/.pi/agent/sessions/`** (Pi’s default session location inside the container). Override with `PI_SESSION_DIR` if needed.

On **HF Space** (`PI_DEPLOYMENT_PROFILE=hf-space`), sessions go to **`/tmp/pi-sessions`** instead (ephemeral; lost on restart).

## Python dependencies

The Pi image installs [`requirements_pi_agent.txt`](../requirements_pi_agent.txt) — Gradio UI + `gradio-client`, HTTP clients, CSV/PDF review helpers (`pandas`, `pymupdf`), and common utilities. It **does not** include spaCy, Presidio, or OCR; heavy redaction runs in `redaction-app-llama`.

Rebuild after changing that file:

```powershell
docker compose -f docker-compose_llama_agentic.yml --profile 27b_36 build pi-agent
```

## HF Space profile (remote redaction backend)

Set `PI_DEPLOYMENT_PROFILE=hf-space` to run the Pi Gradio UI as a **Hugging Face Docker Space** that orchestrates with **Gemini only** and calls a **remote** doc_redaction Space over HTTPS.

| Area | HF Space value |
|------|----------------|
| Pi LLM | Gemini only (`PI_DEFAULT_PROVIDER=google-gemini`) |
| Redaction app | `DOC_REDACTION_GRADIO_URL` (default `https://seanpedrickcase-document-redaction.hf.space`) |
| Auth to redaction | `HF_TOKEN` / `DOC_REDACTION_HF_TOKEN` (Space secret + optional UI override) |
| Text extraction / PII | Locked to `Local model - selectable text` + `Local` |
| VLM faces / signatures | Disabled |
| Port | `7860` |
| Pi session logs | `/tmp/pi-sessions` (`PI_SESSION_DIR`; ephemeral) |

Package and Dockerfile: [`agent-redact/pi-agent/`](../../pi-agent/). Pushes to [agentic_document_redaction](https://huggingface.co/spaces/seanpedrickcase/agentic_document_redaction) on **`dev`** branch via [`.github/workflows/sync-pi-agent-space.yml`](../../../.github/workflows/sync-pi-agent-space.yml) (GitHub secrets: `HF_TOKEN`, `HF_USERNAME`, `HF_EMAIL`).

Local build test from monorepo root:

```powershell
docker build -f agent-redact/pi-agent/Dockerfile -t pi-agent-hf-space .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=... -e HF_TOKEN=... pi-agent-hf-space
```

Pi uses `gradio_client` + `agent-redact/pi/remote_redaction.py` to upload/download from the remote Space; prompts include `{REMOTE_BACKEND_GUIDANCE}` (see [`redaction_prompt.py`](../redaction_prompt.py)).
