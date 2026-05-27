# Pi agent config (Docker)

These files are bind-mounted into the `pi-agent` container at `~/.pi/agent/`.

## Model id

After the llama.cpp service is healthy, confirm the model id:

```bash
curl http://localhost:8000/v1/models
```

If the returned `id` differs from `unsloth/Qwen3.6-27B-MTP-GGUF`, update `models.json` and `settings.json` `defaultModel` to match.

## In-container URLs for task prompts

When filling [`skills/doc-redaction-task-prompt/TASK_PROMPT_TEMPLATE.md`](../../../skills/doc-redaction-task-prompt/TASK_PROMPT_TEMPLATE.md) inside the Pi container, use:

| Placeholder | In-container value |
|-------------|-------------------|
| `{GRADIO_URL}` | `http://redaction-app-llama:7860` |
| `{VLM_BASE_URL}` | `http://llama-inference:8080` |
| `{INPUT_PATH}` | `/home/user/app/workspace/{FILE_NAME}` |
| `{OUTPUT_BASE}` | `/home/user/app/workspace/redact/{FILE_NAME}/` |

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

- **Chat** — streamed assistant text
- **Activity** — agent/turn lifecycle, compaction, auto-retry, tool start/end
- **Tool output** — live bash/read output from `tool_execution_update` / `tool_execution_end`
- **Thinking** — optional stream (`PI_GRADIO_SHOW_THINKING=true`)
- **Abort** — sends Pi RPC `abort` and cancels the in-flight Gradio handler

Optional env vars on `pi-agent`: `PI_GRADIO_SHOW_THINKING`, `PI_GRADIO_SHOW_TOOL_OUTPUT`, `PI_GRADIO_TOOL_OUTPUT_MAX`, `PI_GRADIO_ACTIVITY_MAX_LINES`.

Run the UI locally (outside Docker):

```powershell
cd docker/pi
pip install -r ../../requirements_pi_agent.txt
python gradio_app.py
```

RPC mode (automation, no Gradio):

```powershell
docker compose -f docker-compose_llama_agentic.yml exec -T pi-agent pi --mode rpc
```

Skills are discovered from the repo mount at `/workspace/doc_redaction/skills/` (project skills). Use `/skill:doc-redaction-app` etc., or read paths under `skills/` relative to the working directory.

Sessions persist in the `pi-agent-sessions` Docker volume at `~/.pi/agent/sessions/`.

## Python dependencies

The Pi image installs [`requirements_pi_agent.txt`](../../requirements_pi_agent.txt) — Gradio UI + `gradio-client`, HTTP clients, CSV/PDF review helpers (`pandas`, `pymupdf`), and common utilities. It **does not** include spaCy, Presidio, or OCR; heavy redaction runs in `redaction-app-llama`.

Rebuild after changing that file:

```powershell
docker compose -f docker-compose_llama_agentic.yml --profile 27b_36 build pi-agent
```

