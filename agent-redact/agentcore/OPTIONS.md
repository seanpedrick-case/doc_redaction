# AgentCore orchestration options

Bedrock AgentCore is how the Gradio agent UI can run redaction orchestration **on AWS** instead of inside the Pi Express / Docker container. There are two backends, selected with `AGENT_ORCHESTRATOR`:

| Value | What runs on AWS | Agent style |
|-------|------------------|-------------|
| `agentcore` | **AgentCore Runtime** | Your packaged **LangGraph** agent (curated tools, no shell) |
| `agentcore-harness` (alias `harness`) | **AgentCore Harness** | AWS-managed agent loop with **skills + shell** (Pi-like) |

Local equivalents (no AgentCore): `langgraph` and `pi`. Install and deploy steps live in the longer [AgentCore install guide](README.md). LangGraph behaviour: [redaction_langgraph overview](../redaction_langgraph/README.md).

---

## Big picture

```text
Browser → Gradio agent UI (ECS / Docker / local)
              │
              │  AGENT_ORCHESTRATOR=…
              ▼
     ┌────────────────┬──────────────────┬─────────────────┐
     │ agentcore      │ agentcore-harness│ pi / langgraph  │
     │ Runtime URL    │ Harness ARN      │ in-container    │
     └────────┬───────┴────────┬─────────┴─────────────────┘
              │                │
              ▼                ▼
     LangGraph tools     Strands + skills/shell
     (your code)         (console-configured)
              │                │
              └────────┬───────┘
                       ▼
              doc_redaction Gradio API
              (/doc_redact, /review_apply, …)
              via public HTTPS URL
```

The main **doc_redaction** app is unchanged. AgentCore only replaces *where the orchestrating LLM agent runs*. Both AgentCore modes must reach that app over a **public HTTPS** URL (`DOC_REDACTION_GRADIO_URL` — Express/CloudFront endpoint), not Docker Service Connect (`http://redaction:7860`).

---

## Option 1 — AgentCore Runtime (`agentcore`)

**Idea:** Deploy the same LangGraph redaction agent that runs locally with `AGENT_ORCHESTRATOR=langgraph`, wrapped in `BedrockAgentCoreApp`, as a managed AgentCore **Runtime**.

### How it works

1. Gradio sets `AGENT_ORCHESTRATOR=agentcore` and `AGENTCORE_RUNTIME_URL` (base URL from `agentcore status`, **without** `/invocations`).
2. [`AgentCoreAgentRuntime`](agentcore_runtime.py) proxies each chat/redaction turn to `{URL}/invocations` (HTTP + bearer if `AGENTCORE_API_KEY`, else SigV4 via boto3 `InvokeAgentRuntime`).
3. On AWS, [`entrypoint.py`](entrypoint.py) → [`invoke_agent.py`](invoke_agent.py) builds the LangGraph ReAct agent (`build_redaction_agent`) and streams events back.
4. Uploaded PDFs go out as base64 `workspace_files` in the invoke payload; after the turn, `redact/` artifacts can stream back as `workspace_file` events ([`workspace_sync.py`](workspace_sync.py)).
5. Gradio injects per-turn `runtime_config` (backend URL, OCR/PII defaults, auth cookies) so the remote agent hits the correct doc_redaction deployment.

### Agent behaviour

- **Same curated tools** as local LangGraph: `list_workspace_files`, `doc_redact`, `read`/`write_workspace_text`, `run_workspace_python_script`, `verify_coverage`, `review_apply`, `request_clarification` (optional `approve_review_apply`).
- **Does not use** repo [`skills/`](../../skills/) playbooks. The Gradio partnership prompt is adapted for tool orchestrators (`adapt_prompt_for_tool_orchestrator` in [`redaction_prompt.py`](../shared/redaction_prompt.py)): “read skills first” is replaced with the LangGraph tool workflow.
- Packaging: [`package_runtime.py`](package_runtime.py) vendors `redaction_langgraph/` (+ helpers) into an AgentCore CLI app folder. Runtime bootstrap deliberately **skips** Pi skills sync (`pi_workspace_skills`) so cold start does not hang.

### Key modules

| Module | Role |
|--------|------|
| [`entrypoint.py`](entrypoint.py) | `BedrockAgentCoreApp` handler |
| [`invoke_agent.py`](invoke_agent.py) | Shared invoke: session history, workspace sync, LangGraph stream |
| [`agentcore_runtime.py`](agentcore_runtime.py) | Gradio client for Runtime |
| [`package_runtime.py`](package_runtime.py) | Sync monorepo code into deployable app |
| [`session_store.py`](session_store.py) | In-process multi-turn history per `session_hash` |
| [`../redaction_langgraph/`](../redaction_langgraph/) | Graph, tools, compaction (same as local `langgraph`) |

### Configure

```bash
AGENT_ORCHESTRATOR=agentcore
AGENTCORE_RUNTIME_URL=https://bedrock-agentcore.<region>.amazonaws.com/runtimes/<urlencoded-arn>
# Optional CUSTOM_JWT inbound auth:
# AGENTCORE_API_KEY=…
```

---

## Option 2 — AgentCore Harness (`agentcore-harness`)

**Idea:** Use an AWS-managed **Harness** (Strands-style loop configured in the AgentCore console) that behaves more like the **Pi** coding agent: shell, workspace tools, and document-redaction **skills** from [`skills/`](../../skills/).

### How it works

1. Gradio sets `AGENT_ORCHESTRATOR=agentcore-harness` and `AGENTCORE_HARNESS_ARN` (`arn:aws:bedrock-agentcore:…:harness/…`). There is **no** Runtime-style HTTP invoke URL — only the ARN + SDK `InvokeHarness`.
2. [`AgentCoreHarnessRuntime`](agentcore_harness_runtime.py) sends the user prompt (plus optional file-bridge prefix) via boto3 and maps the event stream to Gradio `AgentStreamEvent`s.
3. On **Start redaction task**, Gradio keeps the full **Pi-style partnership prompt** (skills sections intact — harness is *not* a tool orchestrator per `uses_tool_orchestrator_prompt()`).
4. PDFs are not base64-inlined. [`harness_input_bridge.py`](harness_input_bridge.py) uploads the file to S3 (`AGENTCORE_HARNESS_S3_INPUT_PREFIX` or `S3_OUTPUTS_BUCKET`) and prepends a prompt prefix with a **presigned URL** and suggested workspace path so the harness agent can `curl` it into its mount.

### Skills and playbooks

Harness is meant to follow the same skill ladder as Pi:

| Skill | Purpose |
|-------|---------|
| [`skills/doc-redaction-task-prompt/`](../../skills/doc-redaction-task-prompt/) | Task prompt template |
| [`skills/doc-redaction-app/`](../../skills/doc-redaction-app/) | First-pass `/doc_redact` |
| [`skills/doc-redact-page-review/`](../../skills/doc-redact-page-review/) | Parallel page review → one `/review_apply` |
| [`skills/doc-redaction-modifications/`](../../skills/doc-redaction-modifications/) | CSV / preview / verify mechanics |
| [`skills/doc-redaction-tabular/`](../../skills/doc-redaction-tabular/) | Tabular redaction (when relevant) |

Those skills must be **attached or available inside the Harness** (console / harness project config). This repo’s Gradio client does not package LangGraph tools into the harness; it only forwards prompts and the S3 file bridge. Configure the harness similarly to how Pi loads `.pi/skills/` — with Gradio Client / HTTP access to your doc_redaction URL and a workspace where skills and scripts can run.

### Key modules

| Module | Role |
|--------|------|
| [`agentcore_harness_runtime.py`](agentcore_harness_runtime.py) | Gradio client for `InvokeHarness` |
| [`harness_input_bridge.py`](harness_input_bridge.py) | S3 upload + presigned prompt prefix |
| [`agentcore_boto.py`](agentcore_boto.py) | Shared boto3 client / timeouts |
| Partnership template | [`skills/Example prompt partnership.txt`](../../skills/Example%20prompt%20partnership.txt) via [`redaction_prompt.py`](../shared/redaction_prompt.py) |

### Configure

```bash
AGENT_ORCHESTRATOR=agentcore-harness
AGENTCORE_HARNESS_ARN=arn:aws:bedrock-agentcore:eu-west-2:ACCOUNT:harness/YourHarness-xyz
# AGENTCORE_HARNESS_ENDPOINT=DEFAULT
AGENTCORE_HARNESS_S3_INPUT_PREFIX=s3://your-bucket/harness-inputs/
# AGENTCORE_HARNESS_S3_MOUNT_PATH=/tmp/workspace
RUN_AWS_FUNCTIONS=True
```

IAM needs `bedrock-agentcore:InvokeHarness` (and S3 put/get for the input bridge).

---

## Option 3 — Harness inside the main app (`USE_AGENTCORE`, single container)

**Idea:** Skip the separate agent UI. The **main doc_redaction app** invokes an AgentCore Harness itself, and the harness calls the app's redaction functions as **inline function tools** that run in the main container. AWS hosts only the agent loop, so a production deployment stays one ECS service.

### How it works

1. With `USE_AGENTCORE=True` and `AGENTCORE_HARNESS_ARN` set, the **Redact PDFs/images** tab shows an *AI agent redaction instructions* box. When it has text, *Redact with AI agent* replaces the normal redact button. The same flow is exposed as the `agentic_redact` API (`gr.api`).
2. [`tools/agentcore_harness.py`](../../tools/agentcore_harness.py) copies the upload into a per-run session folder (`OUTPUT_FOLDER/agentcore/<run_id>/`) and builds the partnership prompt (user requirements last). It then calls `InvokeHarness` with the inline tools from [`tools/agentcore_harness_tools.json`](../../tools/agentcore_harness_tools.json).
3. When the harness stream hands off inline tools (`stopReason` usually `tool_use`), the app runs the requested tools in-process: `doc_redact`, `read_session_text` / `write_session_text`, `verify_redaction_coverage`, `preview_redaction_boxes`, `review_apply` and `finish_task`. Paths are confined to the session folder. The app then re-invokes with the `toolResult`s on the same `runtimeSessionId`.
4. **Page review with subagents.** Harness subagents cannot call inline tools. Instead, `stage_page_review_bundles` uploads per-page CSV slices to `AGENTCORE_HARNESS_S3_STAGING_PREFIX` and returns presigned GET/PUT URLs, which the subagents use with `curl`. `merge_page_reviews` then validates and merges the uploads into one review CSV for a single `review_apply`.
5. The agent follows [`skills/doc-redaction-harness/SKILL.md`](../../skills/doc-redaction-harness/SKILL.md), which CDK uploads to S3 and attaches to the harness as a skill.

The harness never connects to the app (no public URL, Service Connect or Gradio auth is needed). Only the app's outbound `InvokeHarness` calls, plus S3 for staging, cross the network.

### Configure

```bash
USE_AGENTCORE=True
AGENTCORE_HARNESS_ARN=arn:aws:bedrock-agentcore:eu-west-2:ACCOUNT:harness/RedactionHarness-xyz
# AGENTCORE_HARNESS_ENDPOINT=DEFAULT
# Required for parallel page review (presigned URLs); defaults to s3://$S3_OUTPUTS_BUCKET/harness-staging/
AGENTCORE_HARNESS_S3_STAGING_PREFIX=s3://your-output-bucket/harness-staging/
# AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW=True
# AGENTCORE_HARNESS_MAX_TOOL_ROUNDS=60
```

The task role needs `bedrock-agentcore:InvokeHarness` ([`cdk/policies/agentcore_harness_invoke_policy.json`](../../cdk/policies/agentcore_harness_invoke_policy.json)) and S3 read/write on the staging prefix. It needs `boto3>=1.44.3`.

**CDK:** `python cdk/cdk_install.py --enable-agentcore-harness` (or answer *yes* in the wizard) sets `ENABLE_AGENTCORE_HARNESS=True`. The stack then:

- creates the `AWS::BedrockAgentCore::Harness` resource, its execution role and the S3 skill;
- grants the main task role `InvokeHarness` on that harness;
- injects `USE_AGENTCORE`, `AGENTCORE_HARNESS_ARN` and the staging prefix into the main container.

Tunables: `AGENTCORE_HARNESS_MODEL_ID` (default `anthropic.claude-sonnet-5`; set explicitly if your account requires a cross-region inference profile such as `eu.anthropic.*`), `AGENTCORE_HARNESS_MAX_ITERATIONS`, `AGENTCORE_HARNESS_TIMEOUT_SECONDS` and `AGENTCORE_HARNESS_MAX_TOKENS`.

**Inline tools (critical):**

1. Harness ``Tools: []`` on the deployed resource (CDK default). Do **not** attach inline schemas on the harness itself — AgentCore then runs them on the VM (`Unknown tool`, no client handoff).
2. Each ``InvokeHarness`` from the main app sends the schemas in ``tools`` (`tools/agentcore_harness_tools.json`).
3. Each ``InvokeHarness`` must also set ``allowedTools: ["*"]`` (see ``harness_invoke_allowed_tools()``). The harness ``AllowedTools`` property is builtins-only; without an invoke-time override, inline tools never appear in the model tool list (the agent correctly reports only ``shell`` / ``skills`` / … and cannot hand off).
4. Expect ``stopReason: tool_use`` and **Client inline dispatch** in the Gradio activity log.

Redeploy after CDK updates: `python cdk/deploy_harness_local.py --yes`.

**Limits to keep in mind:**

- Subagents need `curl` in the harness's default environment; if it's missing, configure an ARM64 `EnvironmentArtifact`.
- Very long documents are bounded by the harness timeout, iteration limits and model context; page-range runs are the workaround.
- There is no Pass 2 VLM step in this mode.

**Local app only (minimal AWS footprint):** deploy just the harness, two S3 buckets, and a developer IAM policy—no ECS or VPC—with [`cdk/deploy_harness_local.py`](../../cdk/deploy_harness_local.py). Details: [`cdk/docs/harness_local_stack.md`](../../cdk/docs/harness_local_stack.md).

**Eval (main app):** With ``APP_EVAL_ENABLED=True`` (or ``AGENT_EVAL_DEEPEVAL_TRACING=1``), completed harness runs write ``eval/run_*/eval_summary.json`` under the harness session folder, using the real inline tool trace (``doc_redact``, ``verify_coverage``, …). Gold auto-match uses the same ``APP_EVAL_*`` / ``AGENT_EVAL_*`` variables as standard redaction and agent-redact. Try the two agent examples on the Redact tab when ``USE_AGENTCORE=True``.

---

## Side-by-side

| | **Runtime (`agentcore`)** | **Harness (`agentcore-harness`)** |
|--|---------------------------|-----------------------------------|
| AWS resource | `…:runtime/…` | `…:harness/…` |
| Gradio config | `AGENTCORE_RUNTIME_URL` | `AGENTCORE_HARNESS_ARN` |
| Invoke API | `/invocations` or `InvokeAgentRuntime` | `InvokeHarness` (ARN only) |
| Orchestration code | **Yours** — LangGraph bundle | **AWS-managed** loop; tools/skills in console |
| Closest local mode | `langgraph` | `pi` |
| Skills under `skills/` | No — system prompt + curated tools | Yes — Pi-like partnership prompt + skill playbooks |
| Shell / open coding | No | Yes (if harness tools allow) |
| File handoff from Gradio | Base64 `workspace_files` in payload | S3 + presigned URL in prompt prefix |
| Artifact sync back | Optional `workspace_file` stream events | Depends on harness workspace / your download steps |
| Package with this repo | `package_runtime.py` / CDK ECR runtime image | Console harness + attach skills; Gradio is client-only |
| Session memory | `session_store.py` in runtime process | Harness runtime session id derived from Gradio `session_hash` |

---

## When to choose which

- **Main-app harness (Option 3)** — You want agent-driven redaction on the standard production deployment with **one container**, Pass 1 review (including parallel page review) and a single apply, without running the separate agent UI.

- **Runtime** — You want the **same LangGraph tool agent** as local `langgraph`, managed on AWS, with a tighter sandbox (no shell). Good fit for CDK demo / ECS Gradio that only needs Pass 1 tool orchestration against a remote doc_redaction URL.
- **Harness** — You want **Pi-style** workflows: read `SKILL.md` playbooks, shell, flexible recovery, page-review skills. You accept configuring tools/skills in the AgentCore console and using S3 for document ingress.
- **Neither** — Stay on `pi` or `langgraph` inside the agent container (no AgentCore URL/ARN). See [pi/agent/README.md](../pi/agent/README.md).

---

## Gradio wiring (shared)

Factory: [`shared/agent_runtime.py`](../shared/agent_runtime.py) → `create_agent_runtime()`.

| Orchestrator | Runtime class |
|--------------|---------------|
| `agentcore` | `AgentCoreAgentRuntime` |
| `agentcore-harness` | `AgentCoreHarnessRuntime` |

Both surface the same chat/activity UX as Pi/LangGraph. Redaction task prompts are built in [`redaction_prompt.py`](../shared/redaction_prompt.py); only `langgraph` and `agentcore` get the tool-orchestrator skill strip.

---

## Related docs

- [Install / deploy guide](README.md) — CLI create, `package_runtime.py`, CDK, auth, troubleshooting  
- [LangGraph overview](../redaction_langgraph/README.md) — tools, workflow, vs Pi  
- [Pi agent README](../pi/agent/README.md) — local Pi, skills sync, env vars  
- Repo skills: [`skills/`](../../skills/)
