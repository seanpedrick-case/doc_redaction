# Eval gold cases

Each **gold case** is a directory of reference artifacts (PDF, instructions, ideal review CSV, OCR gold, policy lists, etc.).

**Git:** only `example_smoke/` and `example_live/` (plus this README) are tracked for CI. Add a new `!agent-redact/eval/gold/<case_id>/` pair in the repo `.gitignore` if you intentionally want another case in GitHub. All other directories under `gold/` stay local (including PDFs).

```text
gold/<case_id>/
  input.pdf                 # source document (full agent runs)
  instructions.txt          # user redaction requirements (Gradio instruction box)
  ideal_review_file.csv     # gold boxes (utf-8-sig) — geometry tier (core spans derived at score time)
  agent_review_file.csv     # optional: recorded agent output for offline compare
  must_redact.txt           # gold phrases: extraction target + application gate
  must_not_redact.txt       # optional gold must-not phrases
  expected_deny_list.txt    # optional gold doc_redact deny_list (extraction)
  tool_args.json            # recorded tool_calls (deny_list / must_*) for extraction eval
  ocr_results_with_words.csv  # policy gate + gold OCR compare (run vs this file)
  redacted.pdf              # optional post-apply deliverable
  manifest.json             # case_id, input_pdf, input_pdf_sha256, instructions_sha256, …
```

**Gold phrase lists** (`must_redact.txt`, etc.) serve two roles:

1. **Application** — [`run_policy_gate.py`](../run_policy_gate.py) scores the agent review CSV against these lists.  
2. **Extraction** — [`compare_policy_lists.py`](../compare_policy_lists.py) scores whether the agent’s tool args matched the same lists.

In the live Gradio app, users only type `instructions.txt` content; the agent must derive tool args. Offline eval uses curated `.txt` files so scoring is not self-graded.

After editing a case PDF or `instructions.txt`, refresh manifest hashes:

```bash
python agent-redact/eval/update_gold_manifest_hashes.py --gold agent-redact/eval/gold/<case_id>
```

## Auto-match (Gradio)

When `AGENT_EVAL_AUTO_GOLD=1`, **Start redaction task** matches the session PDF and redaction requirements to **exactly one** case under `AGENT_EVAL_GOLD_ROOT` (default `agent-redact/eval/gold`):

| Tier | PDF | Instructions |
|------|-----|----------------|
| **Strict** | SHA-256 equals gold PDF or `manifest.json` `input_pdf_sha256` | Normalized text equals gold `instructions.txt` or `instructions_sha256` |
| **Soft** | Upload basename matches gold PDF / `input_pdf` | Every non-empty gold instruction line appears in user text (case-insensitive), or whitespace-collapsed equality |

`AGENT_EVAL_GOLD_MATCH_MODE`: `strict_only`, `soft_only`, or `strict_then_soft` (default).

No unique match → post-run eval **skips gold-tied gates/metrics** (no fallback to `AGENT_EVAL_GOLD`).

Legacy: `AGENT_EVAL_AUTO_GOLD=0` and `AGENT_EVAL_GOLD=<case_dir>` scores every completed task against that case.

`tool_args.json` shape:

```json
{
  "tool_calls": [
    {"tool": "doc_redact", "args": {"deny_list": ["…"]}},
    {"tool": "verify_coverage", "args": {"must_redact": ["…"], "must_not_redact": ["…"]}}
  ]
}
```

See parent [README.md](../README.md) (§2 policy gate, §2b policy extraction).

`example_smoke/` is a **synthetic** sample (no PDF) for unit tests and CLI `--gates-only` checks.
