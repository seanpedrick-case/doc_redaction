---
name: doc-redaction-task-prompt
description: "Copy-paste task prompt template for redact + Pass 1 review jobs. User fills setup placeholders and case-specific redaction requirements at the end; agent follows two-pass model (Pass 1 default, Pass 2 only on flagged pages)."
version: 1.0.0
author: repo-maintained
license: AGPL-3.0-only
---

## Purpose

[`TASK_PROMPT_TEMPLATE.md`](TASK_PROMPT_TEMPLATE.md) is a **user-facing prompt** to start an agent on a full redact-and-review job. It separates:

1. **Fixed workflow** — Pass 1 (OCR/CSV/coverage/prune/single apply) is the default deliverable; Pass 2 VLM only on strict criteria.
2. **User requirements (at the end)** — must redact, must not redact, OCR/PII settings, Pass 2 preference.

Agents invoked with this prompt should **read and follow** these skills (see **Required skills** table in the template):

| Order | Skill | Role |
|-------|--------|------|
| 1 | [`../doc-redaction-app/SKILL.md`](../doc-redaction-app/SKILL.md) | Initial `/doc_redact` |
| 2 | [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md) | Pass 1 review, coverage, prune, apply; Pass 2 if gated |
| 3 | [`../doc-redact-page-review/SKILL.md`](../doc-redact-page-review/SKILL.md) | Optional parallel Pass 1 per-page orchestration |

The template duplicates **task policy** (Pass 1 default, Pass 2 gate, user requirements at end). Skills duplicate **mechanics** (APIs, CSV, downloads). Both belong in an agentic prompt.

## When to use

- Starting a new redaction task in chat (copy template → fill **Setup** → fill **User redaction requirements**).
- Documenting a repeatable handoff pattern for colleagues or automation.

## User workflow

1. Open [`TASK_PROMPT_TEMPLATE.md`](TASK_PROMPT_TEMPLATE.md).
2. Copy from **Setup** through **User redaction requirements**.
3. Replace `{FILE_NAME}`, `{INPUT_PATH}`, `{OUTPUT_BASE}`, `{GRADIO_URL}`, `{PAGE_RANGE}`.
4. Complete **User redaction requirements** — that section overrides generic examples and drives `must_redact` / `must_not_redact` regexes for coverage checks. Include **Pass 2 VLM endpoint** only if Pass 2 may run; otherwise set **N/A — Pass 1 only**.
5. Send to the agent.

## Agent workflow

1. Read **Required skills** (in template) and open each listed `SKILL.md` before the matching phase.
2. Read **User redaction requirements** — authoritative for *what* to redact; overrides generic skill examples.
3. Execute Pass 1 end-to-end; do not run full-document VLM unless user Pass 2 preference allows it and criteria are met.
4. Deliver outputs under `{OUTPUT_BASE}` and a short summary markdown.
