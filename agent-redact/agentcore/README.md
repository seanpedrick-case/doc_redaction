# Bedrock AgentCore install guide

This folder contains the **AgentCore Runtime entrypoint** ([`entrypoint.py`](entrypoint.py)) — a LangGraph redaction agent wrapped in `BedrockAgentCoreApp`.

The **Gradio agent UI** (Pi Express / legacy ECS from [`cdk/cdk_install.py`](../../cdk/cdk_install.py)) stays the user-facing app. When `AGENT_ORCHESTRATOR=agentcore`, that UI proxies prompts to a **separately deployed** AgentCore Runtime via `AGENTCORE_RUNTIME_URL`.

You do **not** define `AGENTCORE_RUNTIME_URL` manually in the AWS console beforehand. It is the **invoke endpoint AWS returns after you deploy** an AgentCore Runtime.

## Architecture (two parts)

| Component | Deployed by | Role |
|-----------|-------------|------|
| **AgentCore Runtime** | [AgentCore CLI](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-cli.html) (`agentcore deploy`) | Runs the LangGraph agent (`entrypoint.py`) |
| **Gradio agent UI** | doc_redaction CDK / `cdk_install.py` (Pi Express or legacy) | Browser UI; streams to AgentCore when `AGENT_ORCHESTRATOR=agentcore` or `agentcore-harness` |

The main **doc_redaction** app (OCR, PII, `/doc_redact`, `/review_apply`) is unchanged.

- **Pi / LangGraph in the Pi Express container** can call the main app over ECS Service Connect (`http://redaction:7860`).
- **Bedrock AgentCore Runtime** (separate AWS service) uses the **main Express public HTTPS URL** (`ExpressServiceEndpoint` stack output). CDK sets that on Pi Express when `ENABLE_AGENTCORE_RUNTIME=True`; Gradio passes it to AgentCore on each invoke via `runtime_config`.

## Runtime vs Harness

| | **AgentCore Runtime** (`agentcore`) | **AgentCore Harness** (`agentcore-harness`) |
|--|-------------------------------------|---------------------------------------------|
| AWS resource | `arn:...:runtime/...` | `arn:...:harness/...` |
| Config | `AGENTCORE_RUNTIME_URL` (HTTP base, no `/invocations`) | `AGENTCORE_HARNESS_ARN` |
| Invoke | `InvokeAgentRuntime` / HTTP SSE | `InvokeHarness` (boto3 stream) |
| Agent code | Your LangGraph bundle (`package_runtime.py`) | AWS-managed Strands loop; tools/skills in console |
| Redaction prompt | Tool orchestrator (curated LangGraph tools) | Pi-like partnership prompt (skills + shell) |
| File upload from Gradio | Base64 `workspace_files` in invoke payload | S3 presigned URL prefix (`AGENTCORE_HARNESS_S3_INPUT_PREFIX`) |

There is **no HTTP invocation URL** for a Harness — configure the **ARN** and call the SDK. The console does not show a Runtime-style URL for Harness resources.

## What `AGENTCORE_RUNTIME_URL` is

In this repo, set the **base URL only** (no trailing slash, **no** `/invocations` suffix). The Gradio client in [`agentcore_runtime.py`](../pi/agentcore_runtime.py) calls:

```text
{AGENTCORE_RUNTIME_URL}/invocations
```

Example base (region and ARN are yours):

```text
https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aeu-west-2%3A123456789012%3Aruntime%2FRedactionAgent
```

Full invoke URLs often include `?qualifier=DEFAULT`; this project appends `/invocations` to the base you configure.

## Prerequisites

- AWS account with credentials configured (`aws sts get-caller-identity`)
- [Node.js 20+](https://nodejs.org/) for the AgentCore CLI
- Python 3.10+
- [AWS CDK bootstrapped](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html) in the target account/region (`cdk bootstrap`)
- Bedrock model access enabled if the agent uses Bedrock models
- IAM permissions for AgentCore deploy and `bedrock-agentcore:InvokeAgentRuntime` (see [Use the AgentCore CLI](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-cli.html))

Install the CLI:

```bash
npm install -g @aws/agentcore
agentcore --help
```

If you previously installed the old Python “starter toolkit” CLI, uninstall it first — both use the `agentcore` command name (`pip uninstall bedrock-agentcore-starter-toolkit` if applicable).

## Step 1 — Deploy the AgentCore runtime

### Option A: New AgentCore project (recommended first time)

The AgentCore CLI creates a **new project folder** next to where you run the command. It does **not** add `app/RedactionAgent/` inside `doc_redaction/agent-redact/agentcore/`.

**Run `create` from a parent directory** (repo root, `agent-redact/`, or `~/projects`):

```bash
# Code-based LangGraph agent (do NOT use --defaults — that creates a harness-only project)
agentcore create \
  --name RedactionAgent \
  --framework LangChain_LangGraph \
  --model-provider Bedrock \
  --memory none

# Or interactive (choose "Agent", then LangChain/LangGraph, Bedrock, memory none):
# agentcore create

cd RedactionAgent
ls app/RedactionAgent
```

**Why `RedactionAgent/` might not appear**

| What you ran | What happened |
|--------------|----------------|
| `... --framework LangChain_LangGraph --defaults` | **Invalid combo.** `--defaults` means “create a **harness** project”, not “fill in missing flags”. The CLI exits after: *Use --no-agent for project-only, or provide all: --framework, --model-provider, --memory* — **no folder is created.** |
| Missing `--model-provider` / `--memory` | Same message; add both flags (see command above). |

Non-interactive **code agent** requires all three: `--framework`, `--model-provider`, `--memory`. See `agentcore create --help`.

Optional: `--output-dir ..` to create the project next to `agent-redact/` instead of inside it.

Expected layout after `create` (project name = `--name` value):

```text
RedactionAgent/
├── agentcore/
│   ├── agentcore.json      # created by CLI — agent/runtime config
│   ├── aws-targets.json    # created by CLI — edit account/region (see below)
│   └── cdk/                # auto-managed CDK for deploy
└── app/
    └── RedactionAgent/     # same name as --name
        ├── main.py         # generated entrypoint — you edit this
        └── pyproject.toml
```

If you do **not** see `RedactionAgent/` (or `app/<name>/` after `cd`):

| Symptom | Likely cause |
|---------|----------------|
| No new folder at all | Incomplete non-interactive flags, or used `--defaults` with `--framework` — use full command above |
| Only `agentcore/` files where you ran the command | You may have run `create` inside an existing AgentCore tree; run it from a clean parent directory instead |
| No `app/` subdirectory | Interactive wizard chose **Harness** or **Skip** — run `agentcore add agent` or recreate with `--framework LangChain_LangGraph` |
| Looking for `app/RedactionAgent` inside `agent-redact/agentcore/` | Wrong place — that folder only holds this repo’s reference [`entrypoint.py`](entrypoint.py), not the CLI scaffold |

Official reference: [AgentCore get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html).

#### Using this repo’s `entrypoint.py`

Do **not** only rename `entrypoint.py` to `main.py`. The CLI already generates `app/RedactionAgent/main.py` with starter LangGraph code.

1. Open the generated `app/RedactionAgent/main.py`.
2. Replace its handler logic with the code from [`entrypoint.py`](entrypoint.py) in this repo (the `@app.entrypoint` async `handler` and `BedrockAgentCoreApp` setup).
3. Add dependencies to `app/RedactionAgent/pyproject.toml` (e.g. `bedrock-agentcore`, `langgraph`, `langchain-*`) matching [`requirements_pi_agent.txt`](../requirements_pi_agent.txt).
4. **Package monorepo code for deploy** — the generated project does not automatically include `redaction_langgraph/`, `tools/`, or `skills/`. Typical approaches:
   - **Container build** (`--build Container`): copy or mount the needed paths from `doc_redaction` in the Dockerfile the CLI scaffolds; or
   - **Vendor** `agent-redact/redaction_langgraph/` and required `tools/` modules into `app/RedactionAgent/` before deploy.

[`entrypoint.py`](entrypoint.py) is a **reference implementation** for this monorepo; AgentCore deploy packages whatever is under `app/<AgentName>/`, not the whole `doc_redaction` tree unless you wire that in.

#### `aws-targets.json` (created for you — edit, don’t invent)

You do **not** need to create this file manually. `agentcore create` writes `agentcore/aws-targets.json` inside the new project.

Edit it so `account` and `region` match where you will deploy (often `us-west-2` or `eu-west-2` for AgentCore):

```json
[
  {
    "name": "default",
    "description": "doc_redaction AgentCore deploy",
    "account": "123456789012",
    "region": "eu-west-2"
  }
]
```

Get your account ID: `aws sts get-caller-identity --query Account --output text`.

Schema reference: [agentcore-cli configuration](https://github.com/aws/agentcore-cli/blob/main/docs/configuration.md).

Then deploy:

```bash
# still inside RedactionAgent/
agentcore dev          # optional: local test at http://localhost:8080/invocations
agentcore deploy
```

### Option B: Package and deploy the doc_redaction agent (recommended)

Use [`package_runtime.py`](package_runtime.py) to sync `redaction_langgraph`, Pi helpers, session memory, and `main.py` into your AgentCore app folder — then deploy.

**Prerequisites:** `agentcore create` project at `agent-redact/RedactionAgent/` (or pass `--target`).

From the **doc_redaction repo root**:

```powershell
# Preview
python agent-redact/agentcore/package_runtime.py --dry-run

# Package into agent-redact/RedactionAgent/app/RedactionAgent/
python agent-redact/agentcore/package_runtime.py

# Package + deploy (set UV_LINK_MODE on Windows / OneDrive)
$env:UV_LINK_MODE = "copy"
python agent-redact/agentcore/package_runtime.py --deploy
```

Or manually after packaging:

```powershell
cd agent-redact\RedactionAgent
agentcore validate
agentcore deploy
```

**What the script copies**

| Source | Destination in `app/RedactionAgent/` |
|--------|--------------------------------------|
| `redaction_langgraph/` | `redaction_langgraph/` |
| `pi/bootstrap_pi_config.py`, `remote_redaction.py` | `pi/` |
| `agentcore/bundle_support/session_workspace.py` | `pi/session_workspace.py` (no Gradio dep) |
| `agentcore/invoke_agent.py`, `session_store.py` | app root |
| Generated `main.py` | replaces template `main.py` |
| Runtime deps | merged into `pyproject.toml` |
| `agentcore.env.example` | env vars to set on the **AWS runtime** |

**After deploy — runtime environment (AWS)**

Bedrock model settings (`PI_DEFAULT_PROVIDER`, `AWS_REGION`, …) belong in `agentcore.env` on the runtime bundle.

**`DOC_REDACTION_GRADIO_URL`:** the Gradio Pi UI sends this on **every invoke** in `runtime_config`, taken from your local `config/pi_agent.env`. That overrides any URL baked into `agentcore.env` (for example an old HF Space default). You should see `Redaction backend for this turn: …` in the activity log with the same URL as the session info panel.

For AWS CDK + AgentCore, `DOC_REDACTION_GRADIO_URL` is the **main Express HTTPS endpoint** (`ExpressServiceEndpoint` / `PiDocRedactionBackendUrl` stack output). Service Connect (`http://redaction:7860`) is only for in-container `pi` / `langgraph` orchestrators. For local Docker dev, set `DOC_REDACTION_GRADIO_URL=http://host.docker.internal:7861` in `pi_agent.env`.

```bash
PI_DEFAULT_PROVIDER=amazon-bedrock
PI_DEFAULT_MODEL=anthropic.claude-sonnet-4-6
AWS_REGION=eu-west-2
PI_WORKSPACE_DIR=/tmp/agentcore-workspace
# Optional fallback if Gradio does not send runtime_config:
# DOC_REDACTION_GRADIO_URL=https://<ExpressServiceEndpoint>  # CDK + AgentCore
```

**Session / follow-up chat:** [`session_store.py`](session_store.py) keeps conversation history per `session_hash` inside the running runtime process. Gradio passes the same `session_hash` for follow-ups. History is lost on cold start until [AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/) is configured.

**Workspace file sync (Gradio ↔ AgentCore):** AgentCore runs on AWS with its own filesystem (`PI_WORKSPACE_DIR`). When you **Start redaction task** from the Gradio UI, the uploaded PDF is base64-encoded in the invoke payload (`workspace_files`) and written into the remote session workspace before the LangGraph agent runs. After the turn completes, artifacts under `redact/` are streamed back as `workspace_file` events and saved into the local Gradio session workspace so the **Outputs** panel can refresh. Default per-file limit: 8 MB (`AGENTCORE_MAX_UPLOAD_BYTES`). Re-deploy the runtime after upgrading `invoke_agent.py` / `workspace_sync.py`.

#### Alternative: Container build

For a full monorepo checkout in the image, switch the runtime in `agentcore/agentcore.json` to `"build": "Container"`, add a `Dockerfile` under `app/RedactionAgent/` that `COPY`s the repo and sets `CMD` to run the entrypoint, then `agentcore deploy`.

## Step 2 — Get the runtime URL

After a successful deploy:

```bash
agentcore status
# or non-interactive (PowerShell-friendly):
agentcore status --json
```

- **`agentcore status`** — runtime ARN, `invocationUrl`, deployment state, log hints (this is what you want for the HTTP endpoint)
- **`agentcore fetch access`** — only for **gateways** or agents using **CUSTOM_JWT** inbound auth (fetches bearer token guidance). It does **not** apply to the default AWS_IAM runtime agent; running bare `agentcore fetch` only prints subcommand help.

Example (your deployed runtime):

```text
AGENTCORE_RUNTIME_URL=https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aeu-west-2%3A404053085091%3Aruntime%2FRedactionAgent_RedactionAgent-ye5Jfw7gKj
```

Use the `invocationUrl` from `agentcore status --json` but **drop the trailing `/invocations`** — the Gradio client appends that path. Auth is **SigV4 (AWS IAM)**, not a static API key, unless you later configure CUSTOM_JWT on the runtime.

You can also find the runtime in the AWS console under **Amazon Bedrock → AgentCore** (runtime resources created by deploy).

### Build the URL from the runtime ARN

If you only have the ARN, the HTTP base is typically:

```text
https://bedrock-agentcore.<region>.amazonaws.com/runtimes/<URL-encoded-ARN>
```

URL-encode the ARN (`:` → `%3A`, `/` → `%2F`). See the [HTTP protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html).

### Test invoke (CLI)

```bash
agentcore invoke --runtime RedactionAgent "Run Pass 1 redaction on the uploaded PDF"
agentcore invoke --runtime RedactionAgent "Hello" --stream
```

Programmatic invoke uses the AWS SDK `InvokeAgentRuntime` API with the runtime ARN ([AWS docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-cli.html)).

## Step 3 — Wire the Gradio agent UI

### Demonstration (CDK) install

The [CDK installer](../../cdk/cdk_install.py) demo profile (`--profile demo --enable-pi`) defaults to **AgentCore orchestration** for the Express agent Gradio UI. The Pi coding-agent CLI remains in the container image but is unused when `AGENT_ORCHESTRATOR=agentcore`.

Deploy is **two-phase** — the runtime URL does not exist until after `agentcore deploy`:

**Phase 1 — AgentCore runtime (before or after CDK; URL required before the agent UI works)**

```powershell
# From doc_redaction repo root (after agentcore create — see Option A/B above)
python agent-redact/agentcore/package_runtime.py `
  --target C:\path\to\RedactionAgent\app\RedactionAgent

cd C:\path\to\RedactionAgent
$env:UV_LINK_MODE = "copy"
agentcore deploy
agentcore status   # copy invocationUrl (base only, no /invocations)
```

Set runtime env on AWS so tools reach doc_redaction. For CDK + AgentCore use the **main Express HTTPS URL** (`ExpressServiceEndpoint`); `agentcore.env` is only a fallback — Gradio overrides via `runtime_config` each invoke.

**Phase 2 — CDK demo stack**

```powershell
# With runtime URL (recommended)
python cdk/cdk_install.py --profile demo --enable-pi `
  --agentcore-runtime-url "https://bedrock-agentcore....amazonaws.com/runtimes/..." --yes

# Or config-only first; add URL after runtime deploy
python cdk/cdk_install.py --profile demo --enable-pi --yes --config-only
# then re-run with --agentcore-runtime-url or edit config/pi_agent.env
```

The installer adds `policies/pi_agentcore_invoke_policy.json` to the ECS task role so the Gradio container can call `InvokeAgentRuntime` (SigV4, no `AGENTCORE_API_KEY` required on ECS).

**Order B (infra first):** deploy CDK with `--agent-orchestrator pi` or defer the URL in interactive mode, deploy the runtime, then `cdk_install.py --config-only --agentcore-runtime-url <URL>` and restart the Pi Express service.

**Fallback orchestrators:** `--agent-orchestrator pi` or `langgraph` runs orchestration inside the Express container (no separate AgentCore runtime).

---

Set in [`config/pi_agent.env`](../../config/pi_agent.env) (or via the CDK installer):

```bash
AGENT_ORCHESTRATOR=agentcore
AGENTCORE_RUNTIME_URL=https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/...
# Optional if the runtime uses bearer/OAuth inbound auth:
AGENTCORE_API_KEY=your-bearer-token
```

**Harness orchestrator** (console-created Harness, Pi-like skills/shell — not the LangGraph bundle):

```bash
AGENT_ORCHESTRATOR=agentcore-harness
AGENTCORE_HARNESS_ARN=arn:aws:bedrock-agentcore:eu-west-2:...:harness/YourHarness-xyz
# Optional endpoint name (default DEFAULT):
# AGENTCORE_HARNESS_ENDPOINT=DEFAULT
# S3 file bridge for Start redaction task (upload PDF + presigned URL in prompt):
# AGENTCORE_HARNESS_S3_INPUT_PREFIX=s3://your-bucket/harness-inputs/
# AGENTCORE_HARNESS_S3_MOUNT_PATH=/tmp/workspace
RUN_AWS_FUNCTIONS=True
```

Client: [`agentcore_harness_runtime.py`](../pi/agentcore_harness_runtime.py). Requires a recent `boto3` with `invoke_harness`.

### CDK installer (non-interactive)

```bash
python cdk/cdk_install.py --profile demo --enable-pi \
  --agent-orchestrator agentcore \
  --agentcore-runtime-url "https://bedrock-agentcore.eu-west-2.amazonaws.com/runtimes/..." \
  --yes
```

Interactive wizard: enable agent mode, then choose **Bedrock AgentCore** when prompted for the orchestration backend.

CDK writes `ENABLE_AGENTCORE_RUNTIME=True` and the URL into `config/cdk_config.env`; `pi_agent.env` gets `AGENT_ORCHESTRATOR` and `AGENTCORE_RUNTIME_URL` for the Pi Express container.

### Typical deployment order

1. Deploy **doc_redaction** main app (CDK) and note **ExpressServiceEndpoint** (main Express HTTPS URL).  
2. Deploy **AgentCore** runtime (`agentcore deploy`) and note the URL.  
3. Set `AGENTCORE_RUNTIME_URL` in `pi_agent.env` / installer and deploy or restart the **Pi Express** agent UI service.

You can deploy the Gradio UI first with `AGENT_ORCHESTRATOR=pi` or `langgraph` and switch to `agentcore` once the runtime URL is known.

## Authentication

| Caller | Typical auth |
|--------|----------------|
| **Same AWS account / SDK** | IAM — `bedrock-agentcore:InvokeAgentRuntime` on the runtime ARN |
| **Gradio container → HTTPS runtime** | Often **OAuth / bearer token** — set `AGENTCORE_API_KEY` for the Gradio → AgentCore HTTP client |

If invoke returns **401** or **403**, check AgentCore inbound auth configuration and that the Gradio task role or bearer token is allowed. See [Authenticate and authorize with Inbound Auth and Outbound Auth](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/) in the AgentCore docs.

## Alternatives (no AgentCore URL)

| `AGENT_ORCHESTRATOR` | When to use |
|----------------------|-------------|
| `pi` (default) | Current Pi agent; HF Space; bash + skills |
| `langgraph` | LangGraph inside the agent container (Docker / ECS); no managed runtime |
| `agentcore` | Managed AgentCore Runtime on AWS |

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Runtime init timeout / `RuntimeClientError: initialization time exceeded` | Container failed to import `main.py` within 30s. Check CloudWatch `/aws/bedrock-agentcore/runtimes/RedactionAgent_RedactionAgent-ye5Jfw7gKj/` **runtime-logs**. Common cause: packaged bootstrap calling Pi-only modules (`pi_workspace_skills`). Re-run `package_runtime.py` and redeploy. |
| 403 on `/invocations` | Runtime uses **AWS IAM**; Gradio must call via SigV4 (`boto3` `invoke_agent_runtime`) or set `AGENTCORE_API_KEY` for CUSTOM_JWT. Ensure `PI_AWS_PROFILE` / `~/.aws` in the pi-agent container and `bedrock-agentcore:InvokeAgentRuntime` on the runtime ARN. |
| Agent cannot reach doc_redaction | `DOC_REDACTION_GRADIO_URL` must be the **main Express HTTPS URL** for AgentCore (not Service Connect). Check `PiDocRedactionBackendUrl` stack output and activity log `Redaction backend for this turn: …` |
| CDK deploy fails | `cdk bootstrap`; `agentcore deploy -v` for verbose AgentCore errors |
| `Failed to parse: \`-\`` during **Synthesize CloudFormation** | Windows + path with spaces (e.g. `OneDrive - Lambeth Council`). AgentCore CDK runs `uv` with `shell: true` and unquoted paths; the `-` in the folder name is passed to `uv` as a bogus package. See below. |
| `hardlink` / `os error 396` during synth | Project on OneDrive; set `UV_LINK_MODE=copy` before deploy |
| Region errors | AgentCore availability per region; align `aws-targets.json` and Bedrock model region |

### Windows: `Failed to parse: \`-\`` (spaces in project path)

**Cause:** `agentcore deploy` packages Python deps with `uv` during CDK synth. On Windows the CDK subprocess uses a shell without quoting, so a path like:

`C:\Users\Sean\OneDrive - Lambeth Council\...\RedactionAgent`

is split at spaces and `uv` receives a lone `-` argument (from `OneDrive - Lambeth`).

**Fix (pick one):**

1. **Deploy from a short path without spaces** (recommended):

   ```powershell
   # Copy (not junction) to a local non-synced folder if OneDrive hardlinks also fail
   xcopy /E /I "...\agent-redact\RedactionAgent" C:\dev\RedactionAgent
   cd C:\dev\RedactionAgent
   $env:UV_LINK_MODE = "copy"   # needed if cache/staging still hits OneDrive
   agentcore deploy
   ```

2. **Junction** (quick test; staging may still land on OneDrive via the target):

   ```powershell
   mklink /J C:\dev\RedactionAgent "...\agent-redact\RedactionAgent"
   cd C:\dev\RedactionAgent
   $env:UV_LINK_MODE = "copy"
   agentcore deploy
   ```

Verified: `node dist/bin/cdk.js synth` succeeds from `C:\dev\RedactionAgent\agentcore\cdk` with `UV_LINK_MODE=copy`; it fails with the `-` parse error from the OneDrive path directly.

Post-deploy reminder from [`cdk/post_cdk_build_quickstart.py`](../../cdk/post_cdk_build_quickstart.py): when `ENABLE_AGENTCORE_RUNTIME=True`, run `agentcore deploy` for this entrypoint and ensure `AGENTCORE_RUNTIME_URL` is set before scaling the Pi agent service.

## References

- [Get started with the AgentCore CLI](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-cli.html)
- [HTTP protocol contract (`/invocations`)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)
- [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python)
- Repo: [`agent_runtime.py`](../pi/agent_runtime.py), [`agent-redact/pi/agent/README.md`](../pi/agent/README.md)
