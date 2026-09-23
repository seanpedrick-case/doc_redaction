# Agent evaluation and observability

Layered quality measurement for agentic Pass 1 redaction (`pi` / `langgraph` / `agentcore`).

| Tier | Question | Tooling |
|------|----------|---------|
| **1. Process health** | Did the orchestrator run reliably? | Arize AX / Phoenix ([`arize_monitoring.py`](arize_monitoring.py)); optional Bedrock AgentCore Evaluations |
| **2. Policy gate** | Did boxes satisfy instructions? | [`run_policy_gate.py`](run_policy_gate.py) → [`tools/verify_redaction_coverage.py`](../../tools/verify_redaction_coverage.py) |
| **2b. Policy extraction** | Did the agent derive the right `must_redact` / `must_not_redact` / `deny_list` from free-text instructions? | [`compare_policy_lists.py`](compare_policy_lists.py) + recorded `tool_args.json` |
| **3. Gold regression** | Do boxes match an ideal review CSV? | [`compare_review_csvs.py`](compare_review_csvs.py) + [`fixtures/`](fixtures/) |
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
  --fixture agent-redact/eval/fixtures/example_smoke

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
  --fixture agent-redact/eval/fixtures/example_smoke \
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
  --fixture agent-redact/eval/fixtures/example_smoke \
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

Layout under [`fixtures/<case>/`](fixtures/):

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

### Matching rules ([`compare_review_csvs.py`](compare_review_csvs.py))

| Mode | Score | Use when |
|------|-------|----------|
| `iou` (default) | Intersection-over-union | Strict geometric agreement |
| `gold_contained` | ∩ / area(gold) | Agent may draw larger boxes |
| `agent_contained` | ∩ / area(agent) | Penalize huge unnecessary boxes |

Default threshold **0.5**. Reports **precision**, **recall**, **F1**, and **strict_pass** (every gold matched and no unmatched agent boxes).

```bash
python agent-redact/eval/compare_review_csvs.py \
  fixtures/example_smoke/ideal_review_file.csv \
  fixtures/example_smoke/agent_review_file.csv \
  --iou-threshold 0.5 --min-f1 0.9

# Smoke subset: require perfect 1:1 coverage
python agent-redact/eval/compare_review_csvs.py gold.csv agent.csv --require-strict

# Merged agent boxes vs one gold span
python agent-redact/eval/compare_review_csvs.py gold.csv agent.csv \
  --mode gold_contained --many-to-one --min-f1 0.85
```

### Brittleness mitigations

- Prefer F1 + containment modes over exact coordinate equality.  
- Use `--many-to-one` when agents split or merge word boxes.  
- Geometry-only by default; add `--require-label-match` for a separate label gate.  
- Keep fixtures small; re-generate gold after OCR pipeline changes.  
- Always run the **policy gate** on the same case — policy can pass when gold F1 fails (acceptable alternate boxing).

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

## 5. Suggested CI usage

Unit tests cover the comparator (`test/test_compare_review_csvs.py`) on synthetic CSVs (no PDF/OCR stack required).

For a nightly or manual agent regression (not in default lightweight CI):

1. Run agent (or LangGraph headless) on each fixture with pinned backend settings.  
2. Persist `tool_args.json` from the run; score extraction with `compare_policy_lists.py --fixture …`.  
3. Copy output `*_review_file.csv` → compare to `ideal_review_file.csv` with `--min-f1` (when gold boxes exist).  
4. Run `run_policy_gate.py --fixture …` (gold lists); optionally `--dual` with recorded tool args.  
5. Optionally assert process KPIs from Phoenix export / AgentCore evaluation results.  
6. Optionally attach an LLM-as-judge summary on failures only ([§4](#4-llm-as-judge-optional-reporting)).

---

## Module map

| Path | Role |
|------|------|
| [`arize_monitoring.py`](arize_monitoring.py) | OTEL setup for Gradio agent runtimes |
| [`compare_review_csvs.py`](compare_review_csvs.py) | Gold vs agent box F1 CLI / library |
| [`compare_policy_lists.py`](compare_policy_lists.py) | Gold vs observed must_redact / deny_list extraction F1 + guardrail |
| [`run_policy_gate.py`](run_policy_gate.py) | Fixture-friendly coverage gate (+ `--dual`) |
| [`fixtures/`](fixtures/) | Case layout + `example_smoke` synthetic CSVs / tool_args |
