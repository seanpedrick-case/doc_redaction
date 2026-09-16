# Eval fixtures

Each case directory should contain:

```text
fixtures/<case_id>/
  input.pdf                 # source document (full agent runs)
  instructions.txt          # user redaction requirements (Gradio instruction box)
  ideal_review_file.csv     # gold boxes (utf-8-sig) — geometry tier
  agent_review_file.csv     # optional: recorded agent output for offline compare
  must_redact.txt           # gold phrases: extraction target + application gate
  must_not_redact.txt       # optional gold must-not phrases
  expected_deny_list.txt    # optional gold doc_redact deny_list (extraction)
  tool_args.json            # recorded tool_calls (deny_list / must_*) for extraction eval
  ocr_results_with_words.csv
  redacted.pdf              # optional post-apply deliverable
  manifest.json             # pdf_sha256, ocr_method, pii_method, notes
```

**Gold phrase lists** (`must_redact.txt`, etc.) serve two roles:

1. **Application** — [`run_policy_gate.py`](../run_policy_gate.py) scores the agent review CSV against these lists.  
2. **Extraction** — [`compare_policy_lists.py`](../compare_policy_lists.py) scores whether the agent’s tool args matched the same lists.

In the live Gradio app, users only type `instructions.txt` content; the agent must derive tool args. Offline eval uses curated `.txt` files so scoring is not self-graded.

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

`example_smoke/` is a **synthetic** sample (no PDF) for unit tests and CLI smoke checks.
