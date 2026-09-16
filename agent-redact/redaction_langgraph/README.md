# LangGraph redaction orchestration

In-process ReAct agent that drives Pass 1 document redaction through a **fixed set of Python tools** (no shell). Selected via `AGENT_ORCHESTRATOR=langgraph`. Shares the Gradio UI contract with Pi through [`shared/agent_runtime.py`](../shared/agent_runtime.py).

For Pi (default) setup and Docker notes, see [`pi/agent/README.md`](../pi/agent/README.md). AgentCore hosts this graph as a managed Runtime (or use Harness for skills-style agents): [`agentcore/OPTIONS.md`](../agentcore/OPTIONS.md). Parent overview: [`agent-redact/README.md`](../README.md).

---

## How it works

```text
Gradio / AgentCore
        │
        ▼
 LangGraphAgentRuntime.prompt_events()     ← runtime.py
        │
        ▼
 create_react_agent(llm, tools)            ← graph.py
        │
        ├── list_workspace_files
        ├── doc_redact          → remote /doc_redact (gradio_client)
        ├── read/write_workspace_text
        ├── run_workspace_python_script   ← CSV policy edits
        ├── verify_coverage
        ├── approve_review_apply (optional gate)
        └── review_apply        → remote /review_apply
```

1. **Factory** — `create_agent_runtime()` builds `LangGraphAgentRuntime` when `AGENT_ORCHESTRATOR=langgraph`.
2. **Graph** — `build_redaction_agent(session_hash)` compiles a LangGraph ReAct loop (`create_react_agent`) with a redaction system prompt and session-scoped tools.
3. **LLM** — `_build_llm()` selects llama.cpp (OpenAI-compatible), Bedrock Converse, or Gemini from `AGENT_DEFAULT_PROVIDER` / `AGENT_DEFAULT_MODEL`.
4. **Turn** — `prompt_events(message)` streams graph updates as `AgentStreamEvent`s (`text_snapshot`, `tool_start` / `tool_end`, `compaction_*`, `done`).
5. **Workspace** — All paths are relative to the Gradio `session_hash` folder under `AGENT_WORKSPACE_DIR` (same isolation as Pi).
6. **Remote redaction** — `doc_redact` / `review_apply` call the doc_redaction Gradio API via [`shared/remote_redaction.py`](../shared/remote_redaction.py) and pull artifacts into the session workspace.

### Intended Pass 1 workflow (system prompt)

The agent is instructed to finish in one turn unless the user stops it:

1. `list_workspace_files` — find the uploaded PDF  
2. `doc_redact` — Pass 1; note `review_csv_relative_path` / OCR words CSV  
3. Edit review CSV — write a short `.py` with `write_workspace_text`, then `run_workspace_python_script`  
4. `verify_coverage` until `pass_strict`  
5. `review_apply` once  
6. `verify_coverage` again on the post-apply `*_redacted.pdf`

Skill markdown under `skills/` / `.pi/skills/` is **not** used; playbooks are Pi-only.

### Resilience inside a turn

| Mechanism | Module | Behaviour |
|-----------|--------|-----------|
| Context compaction | `message_context.py` | `pre_model_hook` trims older messages before each LLM call |
| Overflow retry | `runtime.py` + `llm_errors.py` | One rebuild with aggressive trim if the prompt exceeds the window |
| Tool JSON retry | `workflow_continue.py` | One nudge if tool-call args fail to parse (common on local models) |
| Auto-continue | `workflow_continue.py` | Extra rounds when `review_apply` never ran (`LANGGRAPH_AUTO_CONTINUE_WORKFLOW`); **skipped** when `request_clarification` / `CLARIFICATION_NEEDED:` |
| Identical-error break | `runtime.py` / `tools.py` | Stops repeating the same tool error (`LANGGRAPH_IDENTICAL_ERROR_STOP`) |
| Write-storm guard | `tools.py` | Blocks rewriting the same `.py` without running it |

---

## How the LangGraph graph works (nodes and edges)

LangGraph is built around **state + nodes + edges**. This project uses that model, but **not** as a hand-drawn “Pass 1 → edit CSV → apply” flowchart. It uses LangGraph’s **prebuilt ReAct agent**, which is a small loop. The redaction workflow lives mostly in the **system prompt + tools**, not as separate graph nodes.

### LangGraph concepts

| Piece | Meaning |
|--------|---------|
| **State** | Shared data the graph carries — here mainly a `messages` list (chat + tool calls + tool results) |
| **Node** | A step that reads state, does work, returns a state update |
| **Edge** | What runs next — fixed (`A → B`) or **conditional** (`A → B or END` based on state) |
| **Recursion limit** | Cap on how many node steps one `graph.stream()` may take (`LANGGRAPH_RECURSION_LIMIT`, default 150) |

### Pipeline vs ReAct vs what this repo implements

**ReAct** is a *control pattern* (the model reasons, may call tools, sees results, repeats until it stops). **Nodes/edges** are LangGraph’s *runtime*. They are not alternatives — a ReAct agent *is* a small loop graph. You can also build non-ReAct graphs (fixed pipelines, branches, human approval, etc.) on the same substrate.

#### Simple pipeline (not what we ship)

A hand-built `StateGraph` could force Pass 1 as fixed stages:

```text
START → list_files → doc_redact → edit_csv → verify → review_apply → END
```

| Property | Simple pipeline |
|----------|-----------------|
| Who chooses “what next?” | **Your edges** (code) |
| Same step multiple times? | Only if you add retry/loop edges (e.g. verify fail → edit → verify) |
| Typical linear pipeline | Each stage **once**, in order, then stop |
| Determinism | Higher — model cannot skip `review_apply` by “deciding it’s done” early |

Here the LLM might still write text *inside* a stage, but it does **not** freely pick any tool from the whole set at every step.

#### ReAct loop (what `create_react_agent` builds)

```python
graph = create_react_agent(llm, tools, pre_model_hook=hook)
```

| Property | ReAct (this graph) |
|----------|---------------------|
| Who chooses “what next?” | **The LLM** (`tool_calls` or stop) |
| Same tool multiple times? | **Yes** — `verify_coverage` thrice, rewrite a script, etc. |
| When does it stop? | Model emits a final answer with **no** `tool_calls` (or hits recursion / outer abort) |
| Determinism | Lower — can call tools in any order, skip steps, or stop early |

The model can keep looping over its **entire tool set** until it decides it is done.

#### What this repo actually implements

**ReAct in the middle**, not a Pass 1 pipeline graph:

1. **Graph** — generic ReAct loop only (`pre_model_hook` → `agent` ↔ `tools` → END). No nodes named `doc_redact` / `review_apply`.
2. **Soft workflow** — system prompt in [`graph.py`](graph.py) *asks* the model to follow list → redact → script → verify → apply.
3. **Outer guardrails** — [`runtime.py`](runtime.py) / [`workflow_continue.py`](workflow_continue.py) may nudge or retry if e.g. `review_apply` never ran; they do **not** turn the graph into a forced once-through pipeline.

So: free tool looping by default; prompt + continue nudges *encourage* the Pass 1 sequence; nothing in the compiled graph *forces* each step once in order.

| | Simple pipeline | This repo (ReAct + nudges) |
|--|-----------------|----------------------------|
| Next step chosen by | Graph edges | LLM (+ soft prompt / continue) |
| Call `verify_coverage` 3×? | Only if you coded a retry loop | Yes, whenever the model wants |
| Skip `review_apply`? | Graph would not allow it | Possible; auto-continue tries to recover |
| Pass 1 order | Hard | Soft |

---

### The actual graph: prebuilt ReAct

With compaction on (default), the topology is roughly:

```text
                    ┌─────────────────────────────────┐
                    │                                 │
                    ▼                                 │
START ──► pre_model_hook ──► agent (LLM) ──► should_continue?
                                    │              │
                                    │         has tool_calls?
                                    │         /            \
                                    │        yes            no
                                    │         ▼              ▼
                                    │       tools          END
                                    │         │
                                    └─────────┘
```

#### Nodes in this project

1. **`pre_model_hook`** ([`message_context.py`](message_context.py))  
   Runs before every LLM call. Trims a *copy* of history into `llm_input_messages` so the prompt fits the context window. Full `messages` in graph state stay intact for tool routing.

2. **`agent`**  
   Calls the LLM (Bedrock / Gemini / llama.cpp) with tools bound. Appends an `AIMessage` — either plain text and/or `tool_calls`.

3. **`tools`** (`ToolNode` inside `create_react_agent`)  
   Executes whatever tools the model requested (`doc_redact`, `verify_coverage`, …). Appends `ToolMessage`s with results.

There is **no** node named `doc_redact` or `review_apply`. Those are **tools** the `tools` node can run when the LLM asks for them.

#### The important edge: conditional routing

After `agent`:

- If the last AI message has **`tool_calls`** → go to **`tools`**, then back toward **`agent`** (via `pre_model_hook` again).
- If it has **no** tool calls → **END** (model decided it’s done).

So “edges” here are mostly: *loop while tools are requested; stop when the model answers without tools*.

### Where the redaction “workflow” actually lives

Pass 1 steps (list → redact → script → verify → apply) are **not** graph edges. They are:

1. **System prompt** in [`graph.py`](graph.py) telling the model the sequence  
2. **Tool implementations** in [`tools.py`](tools.py) doing the real work  
3. **Outer Python logic** in [`runtime.py`](runtime.py) that can nudge or retry if the inner graph stops early

So the graph is a **generic ReAct engine**; redaction policy is **prompt + tools + wrappers**.

A typical turn looks like many hops through the same two/three nodes:

```text
hook → agent → tools (list_workspace_files)
hook → agent → tools (doc_redact)
hook → agent → tools (write_workspace_text)
hook → agent → tools (run_workspace_python_script)
hook → agent → tools (verify_coverage)
hook → agent → tools (review_apply)
hook → agent → END   (final text, no tool_calls)
```

Each hop burns recursion budget — hence the high default limit of 150.

### What `runtime.py` adds *outside* the graph

`LangGraphAgentRuntime.prompt_events` does **not** redefine LangGraph nodes. It wraps `graph.stream(...)`:

1. Build input: system message + prior session messages + new user message  
2. Stream `updates` (each node’s state delta) into Gradio events (`tool_start`, `tool_end`, …)  
3. **Outside** the compiled graph:
   - context-overflow → rebuild with aggressive compaction, retry once  
   - bad tool JSON → inject a human nudge, retry once  
   - workflow incomplete (e.g. no `review_apply`) → auto-continue rounds with another human nudge  
   - identical tool errors → break the loop  

Those continue/retry rounds are **another invocation** of the same ReAct graph with more messages appended — not extra nodes inside `create_react_agent`.

### Mental model

| Layer | What it is |
|--------|------------|
| LangGraph graph | Tiny ReAct loop: trim → LLM → tools → … → stop |
| Tools | Side effects: call Gradio `/doc_redact`, edit files, verify, apply |
| System prompt | Soft “workflow” the model should follow (not hard edges) |
| `runtime.py` | Harder guardrails: continue, compact, abort, stream to UI |

### Why this design

- Fast to ship: `create_react_agent` + tools  
- Flexible: one graph for chat and full redaction; model can retry tools when something fails  
- Trade-off: less deterministic than a fixed pipeline; failures show up as wrong tool args, early stop, or hitting the recursion limit — which is why compaction, arg normalisers, and auto-continue exist around it  

A custom Pass 1 **pipeline** graph would force order and usually run each stage once (unless you add retry edges). This repo deliberately keeps the free ReAct loop and steers behaviour with prompt + outer nudges instead.

### Why ReAct suits complex document redaction

For complex redaction, the hard part is rarely “run five named steps once.” It’s **adapt after seeing the document and the tool results**. ReAct fits that better than a simple once-through pipeline.

1. **The path isn’t fixed up front**  
   Some jobs need only `doc_redact` + light CSV tweaks; others need deny-lists, VLM faces/signatures, custom policy scripts, or several verify/fix cycles. A linear pipeline either over-runs unused stages or under-runs when the doc needs more. ReAct can call only what’s needed, as many times as needed.

2. **Failures are normal and content-specific**  
   Wrong path, bad tool JSON, verify `pass_strict=false`, OCR quirks, odd layouts — the next action depends on the *error* and the *file*, not a static edge. ReAct can read the tool output, change args, rewrite a script, and retry. A simple pipeline stops or needs every retry branch coded in advance.

3. **Policy edits are exploratory**  
   Editing `*_review_file.csv` often means: peek at OCR/review rows → write a small `.py` → run it → verify → maybe edit again. That is naturally a loop. Pipelines that assume “edit once → verify once → apply” break on the first failed coverage check.

4. **Documents vary a lot**  
   Multi-column PDFs, scanned pages, names vs orgs vs faces, partial requirements — branching on content is easier when the LLM chooses tools than when every branch is a graph edge you must maintain.

5. **Recovery without redeploying the graph**  
   New failure modes often look like “try a different tool sequence,” which ReAct can attempt from the prompt/tools. A pipeline needs new nodes/edges (and a code change) for each new recovery path.

**Trade-off:** ReAct is less deterministic and can wander, skip `review_apply`, or burn tokens — which is why this package adds prompt guidance and continue/error breakers. For *simple, always-identical* jobs, a pipeline can be cheaper and more predictable. For *messy, review-CSV-heavy* redaction, the flexibility of looping and reacting to tool results usually matters more than a forced once-through path.

---

## Package map

| File | Role |
|------|------|
| [`runtime.py`](runtime.py) | `LangGraphAgentRuntime` — Gradio `AgentRuntime` adapter; streams events; auto-continue / abort / compaction UX |
| [`graph.py`](graph.py) | System prompt, `_build_llm()`, `build_redaction_agent()`, `graph_recursion_limit()` |
| [`tools.py`](tools.py) | Curated tools + arg normalisation, path discovery, CSV repair helpers |
| [`workflow_continue.py`](workflow_continue.py) | Incomplete-workflow detection and continue / breaker prompts |
| [`message_context.py`](message_context.py) | Token budget and `pre_model_hook` compaction |
| [`llm_errors.py`](llm_errors.py) | Classify context-overflow vs tool-JSON parse failures |
| [`verify_coverage_lib.py`](verify_coverage_lib.py) | Coverage check implementation used by `verify_coverage` |
| [`main.py`](main.py) | Bedrock AgentCore entrypoint wrapping the same graph |
| [`headless_pass1.py`](headless_pass1.py) | CLI spike (`--direct-tool` or full agent) |

### Key functions

| Function | Where | Purpose |
|----------|-------|---------|
| `LangGraphAgentRuntime.prompt_events` | `runtime.py` | Drive one user turn; yield UI events |
| `LangGraphAgentRuntime._stream_graph_round` | `runtime.py` | One `graph.stream(..., stream_mode="updates")` pass |
| `build_redaction_agent` | `graph.py` | Compile ReAct graph + system message |
| `build_langgraph_tools` | `tools.py` | Bind tools to `session_hash` |
| `run_doc_redact` / `run_review_apply` / `run_verify_coverage` | `tools.py` | Tool implementations |
| `write_workspace_text` / `run_workspace_python_script` | `tools.py` | Safe file I/O + sandboxed script run |
| `redaction_workflow_incomplete` | `workflow_continue.py` | Detect missing `review_apply` |
| `build_workflow_continue_prompt` | `workflow_continue.py` | Auto-continue / error-breaker nudges |
| `reset_langgraph_tool_session_state` | `tools.py` | Clear per-session loop counters on **New session** |
| `create_agent_runtime` | `shared/agent_runtime.py` | Orchestrator factory (`pi` \| `langgraph` \| …) |

### Tools exposed to the model

| Tool | What it does |
|------|----------------|
| `list_workspace_files` | List session workspace files |
| `doc_redact` | Pass 1 via `/doc_redact` |
| `read_workspace_text` | Read CSV / JSON / `.py` (size-capped) |
| `write_workspace_text` | Write text (soft ~24KB body cap for tool JSON) |
| `run_workspace_python_script` | Run a workspace `.py` (timeout; no arbitrary shell) |
| `verify_coverage` | Pre/post-apply coverage QA |
| `approve_review_apply` | Optional human gate (`LANGGRAPH_REQUIRE_REVIEW_APPROVAL`) |
| `request_clarification` | Pause for ambiguous policy; auto-continue will not nudge |
| `review_apply` | Apply review CSV via `/review_apply` |

---

## Enable and run

```bash
AGENT_ORCHESTRATOR=langgraph
# Optional:
# LANGGRAPH_AUTO_CONTINUE_WORKFLOW=true
# LANGGRAPH_WORKFLOW_CONTINUATIONS=2
# LANGGRAPH_RECURSION_LIMIT=150
# LANGGRAPH_COMPACTION_ENABLED=true
# LANGGRAPH_REQUIRE_REVIEW_APPROVAL=true
```

Headless:

```bash
python agent-redact/redaction_langgraph/headless_pass1.py --pdf path/to.pdf --direct-tool
```

AgentCore uses the same graph via [`main.py`](main.py) (`AGENT_ORCHESTRATOR=agentcore`).

---

## Pass 2 visual review vs this LangGraph flow

Open agentic flows (Pi / AgentCore Harness) follow playbooks such as [`skills/doc-redaction-modifications/SKILL.md`](../../skills/doc-redaction-modifications/SKILL.md): a **fast Pass 1** (OCR / CSV / text) and an **optional Pass 2** where a VLM visually inspects rendered page PNGs or preview overlays. This LangGraph package implements **Pass 1-style orchestration only**. That is a deliberate scope choice, not an oversight of the skill.

### What the skill’s visual path needs

Pass 2 typically:

1. Renders preview or redacted page PNGs (`preview_redaction_boxes`, `/preview_boxes`, or rasterize `*_redacted.pdf`)
2. Calls an OpenAI-compatible **multimodal** VLM per page (`image_url` + policy prompt)
3. Parses findings → conservative CSV edits → optional second `/review_apply`

Pi/Harness can do that with bash, skills, Gradio Client, and ad-hoc Python. Cost is high (~1–2 min/page on local VLMs) and scales with page count — the skill therefore defaults to Pass 1 and runs VLM only when asked or on `pages_flagged_for_vlm`.

### Why that path is not suitable in the current LangGraph agent

| Constraint | Effect |
|------------|--------|
| **Closed tool set** | No `preview_boxes`, no “render page PNG”, no `vlm_review_page`. The agent cannot *see* pages — only text/CSV tool results. |
| **Text ReAct loop** | `create_react_agent` here drives the orchestrator with messages/tool text. It never attaches multimodal page images to the agent turn, even if the backend model supports vision. |
| **No shell / no skills** | Cannot run the skill’s VLM snippet, `preview_redaction_boxes.py`, or curl a vision endpoint unless those become first-class tools. |
| **Context & tool-JSON budget** | Page images (base64) and long VLM transcripts fight compaction and the soft caps that keep `write_workspace_text` parseable on local models. |
| **Latency / cost** | Full-doc visual QA would dominate turn time and tokens; this runtime is tuned for a single Pass 1 tool loop with continue nudges, not a second vision phase. |

Do not confuse this with **`CUSTOM_VLM_FACES` / `CUSTOM_VLM_SIGNATURE`** on `doc_redact`: those run **inside** the redaction server during Pass 1 detection. They are not the orchestrator visually reviewing pages after apply.

### Limitations vs Pi (on visual QA)

| | LangGraph (this package) | Pi (+ modifications skill) |
|--|--------------------------|----------------------------|
| Pass 1 OCR/CSV/verify/apply | Yes — curated tools | Yes — skills + bash / Gradio Client |
| Preview overlay PNGs | Not exposed as a tool | `/preview_boxes` or local `preview_redaction_boxes` |
| Pass 2 per-page VLM | Not supported | Optional; targeted to flagged pages |
| Parallel page-review children | No | Yes ([`doc-redact-page-review`](../../skills/doc-redact-page-review/SKILL.md)) |
| When visual QA is required | Use Pi / Harness, or extend LangGraph with new tools | Follow the skill’s Pass 2 loop |

### Why visual Pass 2 is often unnecessary

The skill itself treats visual VLM as **optional**. For many documents, Pass 1 is enough:

1. **`verify_coverage` replaces most per-page visual review** — programmatic checks for uncovered policy terms, over-redaction, text-layer leaks, and (optionally) pixel sampling. The modifications skill describes this as QA that replaces per-page visual review in most cases.
2. **Word/line OCR + CSV edits** catch missing names/phrases and bad boxes without looking at pixels.
3. **Default deliverable is Pass 1** — apply once when `pass_strict` is true; run Pass 2 only if the user asks for visual QA, Pass 1 is inconclusive (handwriting, stamps, OCR-blind ink), or coverage flags `pages_flagged_for_vlm`.
4. **Cost control** — skipping routine VLM keeps agent turns affordable and within context limits, which matters more for LangGraph’s constrained tool/JSON path than for an open coding agent.

So LangGraph optimises for the **common, text-grounded path**. When you truly need “look at the black boxes on the page,” prefer Pi (or Harness with skills), or add dedicated preview/VLM tools and an explicit Pass 2 phase later.

---

## Limitations vs the Pi agent approach

Pi is a **coding agent** (`pi --mode rpc` via [`pi/pi_rpc_client.py`](../pi/pi_rpc_client.py)): long-lived Node subprocess, JSONL RPC, full bash/read/write/edit, and repo **skills** playbooks. LangGraph is an **in-process** ReAct loop with a closed tool surface. Trade-offs:

| Area | LangGraph | Pi |
|------|-----------|-----|
| **Execution model** | In-process Python; no Node/`pi` CLI | Subprocess `pi --mode rpc`; needs `@earendil-works/pi-coding-agent` |
| **Tools** | Fixed curated set only | Bash + filesystem tools; can call Gradio Client, curl, arbitrary scripts |
| **Skills / playbooks** | Explicitly ignored; behaviour baked into system prompt + tools | `/skill:doc-redaction-*` markdown workflows; richer multi-step guidance |
| **Pass 2 visual VLM / page PNGs** | Not in tool set; Pass 1 text/CSV + `verify_coverage` only | Supported via skills (preview + multimodal VLM); optional and costly |
| **Flexibility** | Can only do what tools allow (workspace text + one Python runner) | Open-ended recovery (inspect PDFs, install helpers, multi-file refactors) |
| **CSV edits** | Must fit write/run script pattern; large hard-coded row lists often break tool JSON on local LLMs | Can edit files incrementally with bash/`edit`, or run longer ad-hoc scripts |
| **Steer / follow-up mid-turn** | Abort only; no Pi-style `steer` / `follow_up` queues | `steer`, `follow_up`, settle grace after `agent_end` |
| **Session / usage** | Lightweight in-memory message list; weaker built-in token/cost stats | Pi session JSONL, `get_session_stats`, usage logging hooks |
| **Compaction** | Trim older messages before each LLM call; aggressive overflow retry | Native Pi compaction (`AGENT_COMPACTION_*`) |
| **Thinking / tool UX** | Coarse `text_snapshot` + tool events | Richer streaming (thinking blocks, partial tool updates, commentary-only bash) |
| **Safety** | Stronger sandbox: no shell, path confined to session workspace | More powerful and more footguns (shell in workspace) |
| **Local small models** | More failure modes (nested/empty tool args, truncated JSON) — mitigated by normalisers and continue nudges | Still hard, but bash + skills give more recovery paths |
| **Dependencies** | LangChain / LangGraph / provider SDKs | Node + Pi CLI + Python Gradio bridge |

### When to prefer which

- **LangGraph** — Deployments that should not ship a coding agent (HF Space, AgentCore); tighter sandbox; Python-only stack; Bedrock AgentCore bundle of the same graph; Pass 1 text/CSV workflows where `verify_coverage` is enough.
- **Pi** — Complex or messy documents; need skills playbooks, shell, mid-turn steering, open-ended debugging, or **Pass 2 visual / VLM page review**.

Both still call the same doc_redaction APIs for actual OCR/PII/`review_apply`; orchestration differs, not the core redaction engine.

---

## Related env vars (LangGraph-focused)

| Variable | Default / notes |
|----------|-----------------|
| `AGENT_ORCHESTRATOR` | `langgraph` to select this runtime |
| `AGENT_DEFAULT_PROVIDER` / `AGENT_DEFAULT_MODEL` | LLM backend |
| `AGENT_LLAMA_BASE_URL` / `AGENT_LLAMA_MODEL_ID` | Local OpenAI-compatible endpoint |
| `AGENT_LLAMA_CONTEXT_WINDOW` | Used by compaction budget |
| `LANGGRAPH_COMPACTION_ENABLED` | Default on |
| `LANGGRAPH_RECURSION_LIMIT` | Default `150` (multi-tool turns exhaust lower limits) |
| `LANGGRAPH_AUTO_CONTINUE_WORKFLOW` | Default on |
| `LANGGRAPH_WORKFLOW_CONTINUATIONS` | Default `2` |
| `LANGGRAPH_IDENTICAL_ERROR_STOP` | Default `3` |
| `LANGGRAPH_REQUIRE_REVIEW_APPROVAL` | Gate `review_apply` behind `approve_review_apply` |
| `LANGGRAPH_MAX_WRITE_CONTENT_BYTES` | Soft cap for `write_workspace_text` bodies |
| `LANGGRAPH_WORKSPACE_SCRIPT_TIMEOUT` | Script run timeout (seconds) |
