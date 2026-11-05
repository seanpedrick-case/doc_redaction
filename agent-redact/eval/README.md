# Agent evaluation and observability

Layered quality measurement for agentic Pass 1 redaction (`pi` / `langgraph` / `agentcore`).

| Tier | Question | Tooling |
|------|----------|---------|
| **1. Process health** | Did the orchestrator run reliably? | Arize AX / Phoenix ([`arize_monitoring.py`](arize_monitoring.py)); optional Bedrock AgentCore Evaluations |
| **2. Policy gate** | Did boxes satisfy instructions? | [`run_policy_gate.py`](run_policy_gate.py) → [`tools/verify_redaction_coverage.py`](../../tools/verify_redaction_coverage.py) |
| **2b. Policy extraction** | Did the agent derive the right `must_redact` / `must_not_redact` / `deny_list` from free-text instructions? | [`compare_policy_lists.py`](compare_policy_lists.py) + recorded `tool_args.json` |
| **3. Gold regression** | Do boxes match an ideal review CSV? | [`compare_review_csvs.py`](compare_review_csvs.py) + [`gold/`](gold/) |
| **3b. OCR gold** | Does run word OCR match fixture gold OCR? | [`compare_ocr_words_csvs.py`](compare_ocr_words_csvs.py) |
| **4. LLM-as-judge** (optional) | Human-readable summary: concision, instruction coverage, narrative accuracy | Not implemented as a gate — see [§4](#4-llm-as-judge-optional-reporting) |

Telemetry alone does not prove redaction correctness. Gold geometry alone is brittle when alternate boxings are policy-OK. Use tiers 1–3 (and 2b when tool args are recorded) for pass/fail; use an LLM judge only for explainability when needed.

---

## 1. Process health (Arize / Phoenix / AgentCore)

### Enable tracing

See [`config/agent.env.example`](../../config/agent.env.example) (`ARIZE_*` / `PHOENIX_*`).

```bash
ARIZE_TRACING_ENABLED=true
ARIZE_BACKEND=phoenix          # or ax
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
# AX: ARIZE_SPACE_ID=… ARIZE_API_KEY=… ARIZE_ENDPOINT=europe
```

- **LangGraph / AgentCore Runtime:** OpenInference LangChain spans + `session.id` = Gradio `session_hash`.
- **Pi:** coarser AGENT/TOOL spans via `iter_pi_events_with_tracing` (LLM stays in Node).

### Recommended process KPIs

| KPI | How to read it |
|-----|----------------|
| Turn success | Gradio/agent `done` without `workflow_incomplete` (LangGraph meta / continue prompts); also root attr `redaction.workflow_incomplete` |
| Policy gate on traces | `redaction.pass_strict` / `redaction.final_pass_strict` on TOOL + AGENT spans (see below) |
| Tool error rate | Fraction of TOOL spans with error / error-like tool JSON payloads |
| Identical-error streaks | Repeated same tool error (see `LANGGRAPH_IDENTICAL_ERROR_STOP`) |
| Latency p50/p95 | End-to-end turn; break out `doc_redact` vs script vs `review_apply` |
| Trajectory completeness | Session includes `doc_redact` … `review_apply` (and preferably `verify_coverage`) |
| Compaction / overflow retries | Status events or span attributes around context trim |
| Cost proxies | Token attributes on spans when available; remote `doc_redact` wall time often dominates |

### `verify_coverage` span attributes

When the agent runs `verify_coverage`, reports are no longer only buried in truncated `OUTPUT_VALUE`. They are set as first-class OTEL attributes:

| Attribute | Span | Meaning |
|-----------|------|---------|
| `redaction.pass_strict` | TOOL (each call) | That coverage report’s strict pass |
| `redaction.pass_with_cleanup` | TOOL / root | Soft cleanup flag |
| `redaction.pages_flagged_for_vlm_count` | TOOL / root | Count of policy-fail pages |
| `redaction.pages_needing_csv_cleanup_count` | TOOL / root | Suspicious-row pages |
| `redaction.verify_has_redacted_pdf` | TOOL / root | Whether post-apply PDF was checked |
| `redaction.final_pass_strict` | Root AGENT | Last `verify_coverage` in the turn |
| `redaction.any_verify_fail` | Root AGENT | True if any call in the turn failed |
| `redaction.verify_coverage_calls` | Root AGENT | How many coverage calls this turn |
| `redaction.workflow_incomplete` | Root AGENT (LangGraph) | `review_apply` not completed |

**Phoenix / AX:** filter sessions where `redaction.final_pass_strict = false` or `redaction.any_verify_fail = true`. Wire: LangGraph + AgentCore Runtime via `run_verify_coverage` → `annotate_current_span_with_coverage`; Pi via tool-output JSON parse in `iter_pi_events_with_tracing`.

### Phoenix / AX tips

1. Filter by `session.id` (= browser `session_hash`) for multi-turn chats.  
2. Inspect tool spans for empty/`{}` path args and write-storm loops.  
3. Alarm on rising tool-error rate, p95 latency, or falling `redaction.final_pass_strict` rate — not on LLM “helpfulness” alone.

### Bedrock AgentCore Evaluations (optional)

**Runtime** (`AGENT_ORCHESTRATOR=agentcore`, LangGraph bundle): same `redaction.*` attributes as local LangGraph, as long as OTEL reaches your collector (Phoenix/AX via `ARIZE_*`, and/or AgentCore’s CloudWatch/OTLP path). Root span name: `agentcore.agent`.

**Evaluating those variables easily:**

| Approach | Effort |
|----------|--------|
| Phoenix / AX dashboards filtered on `redaction.final_pass_strict` | Easy — same as Gradio LangGraph |
| CloudWatch Logs Insights / Metrics on exported span attributes | Easy if OTEL→CloudWatch is enabled |
| AgentCore Evaluations **built-ins** (tool selection, goal success) | Process/trajectory only — they do **not** auto-score `pass_strict` |
| Custom AgentCore / offline eval that reads span attrs or re-runs `run_policy_gate.py` | Straight-forward; prefer offline policy gate for CI pass/fail |

**Harness** (skills/shell) does not go through `run_verify_coverage` in Python, so these attributes appear only if a tool returns the same JSON and you add Harness-side span mapping — Runtime is the easy path.

Pair AgentCore process scores with tier 2 policy gate + tier 3 gold F1 for product quality.

---

## 2. Policy gate (non-gold)

Instruction-tied product QA: does the agent’s `*_review_file.csv` (and optionally the applied PDF) satisfy the **task instructions**? It does **not** ask whether boxes match a hand-drawn ideal CSV, and it is not a process/telemetry check.

[`run_policy_gate.py`](run_policy_gate.py) wraps [`tools/verify_redaction_coverage.py`](../../tools/verify_redaction_coverage.py) — the same logic as the LangGraph `verify_coverage` tool.

```bash
python agent-redact/eval/run_policy_gate.py \
  --gold agent-redact/eval/gold/example_smoke

# Or explicit paths:
python agent-redact/eval/run_policy_gate.py \
  --review-csv path/to/*_review_file.csv \
  --ocr-words-csv path/to/*ocr_results_with_words*.csv \
  --must-redact-file must_redact.txt \
  --must-not-redact-file must_not_redact.txt \
  --redacted-pdf path/to/*_redacted.pdf \
  --require-pass-strict
```

### Inputs

| Input | Role |
|--------|------|
| Agent `*_review_file.csv` | Proposed redaction boxes (normalized coords + text/label) |
| Word OCR CSV (`*ocr_results_with_words*`) | Where policy terms appear on the page |
| `must_redact.txt` | Regexes for things that **must** be boxed (derived from user instructions; one pattern per line) |
| `must_not_redact.txt` | Regexes for things that **must not** be boxed |
| Optional `*_redacted.pdf` | Post-apply deliverable for text-layer / pixel checks |

Lists should come from the **same** custom instructions the agent saw. A wrong or incomplete list makes the gate misleading.

### Checks (per page)

1. **Uncovered must-redact (policy recall)** — For each OCR **word** that matches a `must_redact` pattern, require that some review box **intersects** that word’s bbox. If not → `uncovered_terms` → fails `pass_strict`. Also fails if the page has matching OCR but **zero** review rows.
2. **Over-redaction (policy precision)** — If a review row’s `text` matches `must_not_redact` (and is not also allowed by `must_redact`) → `over_redacted` → fails `pass_strict`.
3. **Suspicious rows (cleanup, softer)** — Very short / fragment boxes mark `pass_with_cleanup=false` and feed `pages_needing_csv_cleanup`. That does **not** by itself fail `pass_strict`.
4. **Post-apply text-layer leaks** (if `--redacted-pdf` given) — If a `must_redact` pattern still matches extractable text on `*_redacted.pdf` → `text_layer_leaks` → fails `pass_strict`. Use the deliverable PDF, not `_redactions_for_review.pdf`.
5. **Optional pixel sampling** (`--sample-pixels`) — Box centres on the redacted page should be dark; otherwise `pixel_failures` → fails `pass_strict`.

### Outcomes

| Flag | Meaning |
|------|---------|
| `pass_strict` | Policy OK: no uncovered must-redact, no over-redact, no leaks/pixel fails |
| `pass_with_cleanup` | Also no suspicious short rows |
| `pages_flagged_for_vlm` | Pages that failed policy (candidates for visual Pass 2) |
| `pages_needing_csv_cleanup` | Suspicious rows only — prune, don’t jump to VLM |

Default gate for eval/CI: **`pass_strict == true`**. Optionally also require `pass_with_cleanup` and empty `pages_flagged_for_vlm` for stricter CI.

### Caveat: word-level `must_redact`

Patterns are matched against **individual OCR words** (and against full page text after apply). Multi-word regexes like `Alice\s+Example` often **won’t** hit single-word OCR. Prefer word-suited patterns (e.g. `Alice`, or alternation that matches how OCR tokenises names).

### Policy gate vs gold CSV — when policy alone is enough

| Policy gate | Gold compare (tier 3) |
|-------------|------------------------|
| “Did we redact the **right things** per instructions?” | “Do boxes **look like** this ideal CSV?” |
| Uses OCR + regex lists | Uses IoU / containment vs gold boxes |
| Allows different box shapes if terms are covered | Can fail on merge/split even when policy is fine |
| No ideal CSV required | Needs hand-authored gold |

**Policy gate is adequate (gold optional) when:**

- Success means **instruction compliance** — required names/phrases are boxed, forbidden ones are not, and (after apply) text doesn’t leak — not pixel-perfect agreement with a particular boxing style.
- You care that “Alice Example” is covered whether the agent used **one** span or **two** word boxes (gold F1 would often fail the merge/split case; policy can still pass).
- Building and maintaining ideal review CSVs is expensive (many docs, shifting OCR, frequent instruction changes); regex lists derived from instructions are cheaper to keep in sync.
- You are comparing orchestrators (`pi` vs `langgraph`) on the **same** backend OCR/PII settings and want a stable, instruction-tied score rather than geometry regression.
- Pre-prod / CI smoke for “did Pass 1 meet the deny/allow intent?” before investing in gold fixtures.

**Add gold CSV when:**

- You need a **regression lock** on coordinates (e.g. known-good boxing for a compliance demo fixture).
- Instructions are vague or visual (“redact the signature in the bottom right”) where OCR/`must_redact` cannot express the target.
- You want to catch **extra** boxes that don’t violate `must_not_redact` but still differ from an agreed ideal (over-boxing that policy lists don’t name).
- You are tuning box quality (tightness, split vs merge) rather than only policy coverage.

In short: **policy gate = “right content redacted per the brief.”** **Gold = “boxes match this reference drawing.”** Many agentic evals only need the former; use gold for a smaller smoke subset or high-stakes fixtures.

### Gradio vs offline lists

| Context | Where policy lists come from |
|---------|------------------------------|
| Live Gradio app | User types instructions only; the **agent** should derive `deny_list` / `must_redact` / `must_not_redact` as **tool args** |
| Offline fixtures | Curated `must_redact.txt` (etc.) so application scoring is not self-graded |
| Extraction eval (2b) | Compare those curated files to recorded agent tool args |

---

## 2b. Policy extraction (instructions → tool args)

Separate from “did boxes cover the terms?”: **did the model correctly identify phrases to must-redact / must-not-redact (and deny_list)?**

This matters for Gradio: if the agent never passes `must_redact`, live `verify_coverage` barely checks instruction compliance even when boxing looks fine.

### Primary: set F1 vs gold lists

```bash
python agent-redact/eval/compare_policy_lists.py \
  --gold agent-redact/eval/gold/example_smoke \
  --min-must-redact-recall 0.9 \
  --min-must-not-redact-recall 0.9 \
  --require-guardrail \
  --expect-must-not-redact \
  --expect-deny-list \
  --min-deny-list-f1 0.9
```

- Gold: fixture `must_redact.txt`, `must_not_redact.txt`, optional `expected_deny_list.txt`  
- Observed: `tool_args.json` (`tool_calls` for `doc_redact` / `verify_coverage`)  
- Scores **precision / recall / F1** per list (exact normalized match; `--soft-match` allows substrings)  
- **Guardrail:** fail if `must_redact` (and optionally deny / must-not) is empty when expected  

Capture `tool_args.json` from Phoenix/Gradio tool events or a harness that records LangGraph `tool_start` args after a run.

### Companion: dual policy gate

Run application scoring twice (compliance = **gold** lists; agent lists are diagnostic):

```bash
python agent-redact/eval/run_policy_gate.py \
  --gold agent-redact/eval/gold/example_smoke \
  --dual
```

| Result pattern | Likely cause |
|----------------|--------------|
| Gold fail, agent pass | Extraction under-specified the real policy |
| Gold pass, agent empty `must_redact` | Extraction omitted; live verify was a paper tiger |
| Both fail | Boxing / apply problem (or both extraction and boxing) |

### Optional: LLM-as-judge on extraction only

Given `instructions.txt` + agent lists (not the PDF), a judge can score paraphrase coverage. Use for triage only — same caveats as [§4](#4-llm-as-judge-optional-reporting). Do not auto-generate gold lists with another LLM as the sole source of truth.

---

## 3. Gold vs agent review CSV

### Fixture protocol

Layout under [`gold/<case>/`](gold/):

| File | Required | Purpose |
|------|----------|---------|
| `input.pdf` | for full agent runs | Source document |
| `instructions.txt` | recommended | Custom redaction requirements (agent prompt tail) |
| `ideal_review_file.csv` | **yes** for gold compare | Hand-authored expected boxes |
| `agent_review_file.csv` | for offline compare | Agent output under test (or CI artifact) |
| `must_redact.txt` | for policy gate | Regexes derived from instructions |
| `must_not_redact.txt` | optional | Terms that must not be boxed |
| `ocr_results_with_words.csv` | for policy gate | Word OCR from the same redaction run |
| `redacted.pdf` | optional | Post-apply deliverable |
| `manifest.json` | recommended | PDF hash, OCR/PII methods, notes |

Version gold CSVs with PDF content hash + instruction text so fixtures stay reproducible. Pin `ocr_method` / `pii_method` / redaction backend URL when comparing orchestrators (`pi` vs `langgraph`).

### Gold geometry scoring ([`compare_review_csvs.py`](compare_review_csvs.py))

**Default (Gradio + regression):** `AGENT_EVAL_GOLD_GEOMETRY_SCORING=core_coverage` (set `legacy_box_match` for the old greedy 1:1 row matcher).

| Scoring | Meaning |
|---------|---------|
| **`core_coverage`** | Derive **core** rows from ideal/agent CSV (drop nested sub-spans like `Dr` inside `Dr David R.`). **Recall** = fraction of core gold spans whose area is covered by the **union** of agent boxes (≥ threshold). **Precision** = fraction of core agent boxes that overlap a core gold span. |
| **`legacy_box_match`** | Every CSV row counts; greedy pairwise match with containment/IoU modes below. |

Core coverage answers “are blessed **spans** redacted?” without requiring duplicate PII-detector layers. Tier 2 **policy gate** still answers “are **words** covered per regex lists?”

| Legacy mode (`AGENT_EVAL_GOLD_GEOMETRY_CONTAINMENT`) | Score | Use when |
|------|-------|----------|
| `gold_contained` | ∩ / area(gold) | Agent may draw larger boxes |
| `agent_contained` | ∩ / area(agent) | Penalize huge unnecessary boxes |
| `iou` | Intersection-over-union | Strict geometric agreement |

Threshold **0.85** default (`AGENT_EVAL_GOLD_IOU_THRESHOLD`). Legacy also supports **`AGENT_EVAL_GOLD_MANY_TO_ONE=true`**.

| Env | Default | Role |
|-----|---------|------|
| `AGENT_EVAL_GOLD_GEOMETRY_SCORING` | `core_coverage` | `core_coverage` or `legacy_box_match` |
| `AGENT_EVAL_GOLD_GEOMETRY_CONTAINMENT` | `gold_contained` | Legacy matcher only (replaces overloading `AGENT_EVAL_GOLD_MATCH_MODE` for geometry) |
| `AGENT_EVAL_GOLD_CORE_SUBSUME_THRESHOLD` | `0.9` | Drop nested core rows when smaller box is ≥90% inside larger + text sub-span |
| `AGENT_EVAL_GOLD_CORE_AGENT_OVERLAP` | `0.05` | Min intersection/area(agent) for precision TP |

```bash
python agent-redact/eval/compare_review_csvs.py \
  gold/example_smoke/ideal_review_file.csv \
  gold/example_smoke/agent_review_file.csv \
  --scoring core_coverage --iou-threshold 0.85

# Legacy greedy matcher
python agent-redact/eval/compare_review_csvs.py gold.csv agent.csv \
  --scoring legacy_box_match --mode gold_contained --many-to-one --min-f1 0.85
```

### Brittleness mitigations

- Prefer **core_coverage** for agentic runs; use **legacy** only when you need strict CSV row parity.  
- Keep gold CSVs as exported from review (PII stack rows OK — core is derived at score time).  
- Always run the **policy gate** on the same case — policy can pass when geometry F1 is low (alternate boxing).

---

## 4. LLM-as-judge (optional reporting)

A final **LLM-as-judge** step can summarise the whole redaction run from the artifacts above: whether the agent was **concise**, whether products look **accurate**, and whether they **answer the user’s instructions**. That is **additionally useful for explainability**, but **not required** for deciding pass/fail when tiers 1–3 already score the run.

### Where it can help

- **Human-facing reports** — CI comments or dashboards that explain *why* a run failed beyond `f1=0.72` or `pass_strict=false`.
- **Concision / trajectory narrative** — tool-call count, retries, rewrite-storms, early stop vs over-tooling; a judge can narrate traces + tool logs better than a single KPI.
- **Instruction coverage in prose** — e.g. user asked for faces + an org name, but `must_redact` lists were incomplete; the judge can flag intent gaps that regex/gold missed.
- **Cases without gold CSVs** — or faces/handwriting where geometry eval is weak; soft triage when only telemetry + partial policy exist.

### Where it is unnecessary or risky

- **Pass/fail gates** — prefer deterministic metrics (coverage, gold F1, `workflow_incomplete`). Judges are noisy, prompt-sensitive, and can rubber-stamp bad boxes if evidence is weak.
- **Cost / latency** — another model call over traces + CSVs + coverage JSON; overkill for every smoke turn.
- **False confidence** — polished prose can hide a missing `review_apply` or failed `pass_strict` unless the judge is forced to ground on those fields.

### If you add a judge later

Treat it as **reporting / triage**, not the source of truth:

1. Always run telemetry + policy gate (+ gold F1 when fixtures exist) first.  
2. Call the judge on **failures**, sampled production traces, or release reviews — not every successful turn.  
3. Feed **structured inputs only**: coverage JSON, compare F1 report, tool trajectory / `workflow_incomplete`, instruction text (not raw page images unless you add Pass 2 visual review).  
4. Require fixed scores (e.g. concision 1–5, instruction coverage 1–5, accuracy confidence 1–5) **plus** citations to those artifacts, and **fail closed** if `pass_strict` is false regardless of the summary prose.

**Bottom line:** useful for explainability and soft process quality; unnecessary for “is this redaction good enough?” when policy + gold (+ apply checks) already say pass/fail.

---

## 5. Local agent regression (DeepEval-ready)

Fast **offline** tests (no LLM) run in default `pytest`: comparators and [`run_agent_regression.py --gates-only`](run_agent_regression.py) on [`gold/example_smoke/`](gold/example_smoke/).

**Live** regression runs on your machine only (not GitHub Actions in this phase): Docker compose + the same orchestrator models as the Gradio agent (`AGENT_DEFAULT_PROVIDER` / `AGENT_DEFAULT_MODEL` in [`config/agent.env`](../../config/agent.env.example)).

### Orchestrator providers (same as Gradio)

| Provider | `AGENT_DEFAULT_PROVIDER` | Typical local setup |
|----------|--------------------------|---------------------|
| llama.cpp | `llama-cpp` | [`docker-compose_llama_agentic.yml`](../../docker-compose_llama_agentic.yml) + `AGENT_LLAMA_BASE_URL` |
| Gemini | `google-gemini` | `GOOGLE_API_KEY` (or project env) in `config/agent.env` |
| Bedrock | `amazon-bedrock` | AWS creds + `AGENT_DEFAULT_MODEL` |

Set **`AGENT_ORCHESTRATOR`** to `langgraph` or `pi` before running the harness (or pass `--orchestrator`).

### Recipe

1. Start stack (example):

   ```bash
   docker compose -f docker-compose_llama_agentic.yml --profile 27b_38 up -d
   # Also start the matching agent-* compose service
   ```

2. Configure [`config/agent.env`](../../config/agent.env.example) (never commit secrets).

3. Run regression:

   ```powershell
   # Windows
   .\agent-redact\eval\run_local_agent_eval.ps1 -Gold agent-redact/eval/gold/example_live -Orchestrator langgraph

   # Linux/macOS
   GOLD=agent-redact/eval/gold/example_live ORCHESTRATOR=langgraph \
     ./agent-redact/eval/run_local_agent_eval.sh
   ```

   Or directly:

   ```bash
   python agent-redact/eval/run_agent_regression.py \
     --gold agent-redact/eval/gold/example_live \
     --orchestrator langgraph

   # Offline scoring only (recorded CSVs / example_smoke):
   python agent-redact/eval/run_agent_regression.py \
     --gold agent-redact/eval/gold/example_smoke --gates-only
   ```

4. Outputs land under [`eval/runs/`](runs/) (`agent_review_file.csv`, `tool_args.json`, `policy_gate.json`, `gold_geometry.json`, `ocr_gold.json`, …).

5. **Optional DeepEval** (dev venv):

   ```bash
   pip install -r agent-redact/requirements_agent.txt -r agent-redact/requirements_agent_eval.txt
   deepeval test run test/agent/test_agent_deepeval.py
   deepeval inspect   # requires pip install "deepeval[inspect]"
   ```

   **DeepEval metrics:** offline pytest uses custom **`PolicyStrictMetric`** only. Live runs can add DeepEval **`ToolCorrectnessMetric`** + **`TaskCompletionMetric`** via [`build_regression_deepeval_metrics()`](deepeval_metrics.py) (set `AGENT_EVAL_DEEPEVAL_TRACING=1` for LangGraph tool traces; judge API for task completion). See [`example_commands.txt`](example_commands.txt). Product pass/fail should still rely on policy gate / gold F1 ([§4](#4-llm-as-judge-optional-reporting)).

   **Gradio agent UI:** with `AGENT_EVAL_DEEPEVAL_TRACING=1`, each completed **Start redaction task** writes `{session_workspace}/eval/run_<timestamp>/eval_summary.json` (pass/fail/score for **policy_strict**, **tool_correctness**, **task_completion**, plus deterministic **gold_geometry**, **ocr_gold**, **policy_extraction** when a gold case applies). With **`AGENT_EVAL_AUTO_GOLD=1`**, the UI matches upload + instructions to one case under **`AGENT_EVAL_GOLD_ROOT`** at task start (see [`gold/README.md`](gold/README.md)); no unique match skips gold-tied metrics (fail closed). Legacy: **`AGENT_EVAL_GOLD=<case_dir>`** with auto off scores every task against that case. Task completion uses your redaction instructions and needs judge API credentials when DeepEval is installed. Optional **`AGENT_EVAL_MIRROR_RUNS=1`** copies the folder to [`eval/runs/`](runs/).

   **Run summary CSV:** when tracing is on (opt out with `AGENT_EVAL_RUN_SUMMARY_CSV=0`), each run appends one row to [`eval/outputs/eval_run_summary.csv`](outputs/eval_run_summary.csv) and `{session_workspace}/eval/eval_run_summary.csv` — same idea as topic-modeller `eval_run_summary.csv` (timestamps, pass/score columns, `output_files` / `intermediate_files`). Columns **`orchestrator`**, **`model_name`**, **`ocr_method`**, **`pii_method`**, **`encourage_vlm_faces`**, and **`encourage_vlm_signatures`** (Gradio “Encourage CUSTOM_VLM_*…” checkboxes at task start) are taken from the run folder’s `gradio_run_meta.json` (written at post-run eval from the Gradio task), then `eval_summary.json`, then `manifest.json` (`run_*` overlays or gold defaults). Rebuilds without meta still get OCR/PII from a copied gold `manifest.json`. Refresh geometry + rebuild from mirrored runs: [`refresh_eval_geometry.py`](refresh_eval_geometry.py) (or `rebuild_eval_run_summary_csv()` alone if JSON is already current).

   **Deterministic CSV columns** (from `gold_geometry.json`, `ocr_gold.json`, `policy_extraction.json` under each run dir): `gold_geometry_f1` / `_recall` / `_precision`, `ocr_gold_f1` / `_recall` / `_precision`, `policy_extraction_must_redact_f1`, `policy_extraction_deny_list_f1`, plus `*_skipped` flags when comparison did not run.

   **Gold / OCR env tunables** (optional):

   | Variable | Default | Role |
   |----------|---------|------|
   | `AGENT_EVAL_GOLD_GEOMETRY_SCORING` | `core_coverage` | `core_coverage` or `legacy_box_match` |
   | `AGENT_EVAL_GOLD_GEOMETRY_CONTAINMENT` | `gold_contained` | Legacy matcher: `iou`, `gold_contained`, `agent_contained` |
   | `AGENT_EVAL_GOLD_IOU_THRESHOLD` | `0.85` | Coverage threshold (core union recall + legacy match) |
   | `AGENT_EVAL_GOLD_MANY_TO_ONE` | `true` | Legacy: multiple agent boxes per gold box |
   | `AGENT_EVAL_GOLD_MATCH_MODE` | `strict_then_soft` | **Task auto-match only** (PDF + instructions), not geometry |
   | `AGENT_EVAL_MIN_GOLD_F1` | *(unset)* | Fail gates / optional `overall_pass` when below |
   | `AGENT_EVAL_MIN_OCR_F1` | *(unset)* | Fail gates when OCR gold F1 below threshold |
   | `AGENT_EVAL_OCR_IOU_THRESHOLD` | *(unset)* | Require word bbox IoU for OCR token matches |
   | `AGENT_EVAL_OVERALL_INCLUDE_GOLD=1` | off | Include min gold/OCR F1 in Gradio `overall_pass` |

   OCR gold compare uses fixture `ocr_results_with_words.csv` vs run copy; skipped when `manifest.json` `ocr_method` differs between fixture and run.

### Fixtures

- [`example_smoke/`](gold/example_smoke/) — synthetic CSVs, no PDF (`--gates-only`).
- [`example_live/`](gold/example_live/) — small public PDF via `manifest.json` `input_pdf`; tune gold lists after a blessed run.

GitHub Actions can reuse the same CLI later; default repo CI remains lightweight ([`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)).

### Main redaction app (`APP_EVAL_ENABLED`)

The **base** Gradio app ([`app.py`](../../app.py)) reuses this eval tree when `APP_EVAL_ENABLED=1` in [`config/app_config.env`](../../config/app_config.env):

- **Trigger:** **Extract text and redact document**, **Redact document**, and extract-text-only paths (via [`tools/app_eval_hook.py`](../../tools/app_eval_hook.py)).
- **Match:** PDF + derived instructions from entity/deny/OCR settings ([`derive_app_eval_instructions.py`](derive_app_eval_instructions.py)); same [`gold_match.py`](gold_match.py) tiers as the agent (`APP_EVAL_AUTO_GOLD`, `APP_EVAL_GOLD`, `APP_EVAL_GOLD_ROOT`).
- **Gates:** [`post_run_app_eval.py`](post_run_app_eval.py) → [`run_deterministic_gates`](run_agent_regression.py) (full redact) or OCR gold only (extract-text-only).
- **Policy gate:** still [`verify_redaction_coverage`](../../tools/verify_redaction_coverage.py) via [`run_policy_gate.py`](run_policy_gate.py) and gold `must_redact.txt`.
- **Metrics:** deterministic tiers + DeepEval **task_completion** when `AGENT_EVAL_DEEPEVAL_TRACING=1`; **tool_correctness** skipped.
- **CSV:** `run_source=app_gradio` in [`eval_run_summary.csv`](outputs/eval_run_summary.csv) when `APP_EVAL_RUN_SUMMARY_CSV` is not off.
- **CLI:** `python cli_redact.py … --app-eval` or offline `python agent-redact/eval/run_app_eval.py --pdf … --settings-json …`.

See [`gold/README.md`](gold/README.md) (main app authoring).

---

## Module map

| Path | Role |
|------|------|
| [`arize_monitoring.py`](arize_monitoring.py) | OTEL setup for Gradio agent runtimes |
| [`post_run_eval.py`](post_run_eval.py) | Gradio post-task `eval_summary.json` when `AGENT_EVAL_DEEPEVAL_TRACING=1` |
| [`post_run_app_eval.py`](post_run_app_eval.py) | Main app post-redact eval when `APP_EVAL_ENABLED=1` |
| [`derive_app_eval_instructions.py`](derive_app_eval_instructions.py) | Canonical instructions from main-app UI settings |
| [`app_eval_env.py`](app_eval_env.py) | `APP_EVAL_*` environment helpers |
| [`eval_run_summary_csv.py`](eval_run_summary_csv.py) | Append-only `eval_run_summary.csv` across Gradio + regression runs |
| [`compare_review_csvs.py`](compare_review_csvs.py) | Gold vs agent box F1 CLI / library |
| [`compare_ocr_words_csvs.py`](compare_ocr_words_csvs.py) | Gold vs run word OCR F1 CLI / library |
| [`compare_policy_lists.py`](compare_policy_lists.py) | Gold vs observed must_redact / deny_list extraction F1 + guardrail |
| [`run_policy_gate.py`](run_policy_gate.py) | Fixture-friendly coverage gate (+ `--dual`) |
| [`run_agent_regression.py`](run_agent_regression.py) | Local live/offline harness → deterministic gates |
| [`deepeval_metrics.py`](deepeval_metrics.py) | DeepEval metric wired to `pass_strict` |
| [`run_local_agent_eval.ps1`](run_local_agent_eval.ps1) / [`.sh`](run_local_agent_eval.sh) | Thin wrappers for local runs |
| [`gold/`](gold/) | Case layout + `example_smoke` / `example_live` |
