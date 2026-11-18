"""Harness agent usage CSV row (task=agent)."""

from __future__ import annotations

import csv

from tools.agentcore_harness_stream import HarnessTokenUsage
from tools.harness_usage_log import AGENT_USAGE_TASK, record_harness_agent_usage


def test_record_harness_agent_usage_writes_agent_task(monkeypatch, tmp_path):
    import tools.harness_usage_log as hul

    monkeypatch.setattr(hul, "SAVE_LOGS_TO_CSV", True, raising=False)
    monkeypatch.setattr(hul, "SAVE_LOGS_TO_DYNAMODB", False, raising=False)
    monkeypatch.setattr(hul, "DISPLAY_FILE_NAMES_IN_LOGS", True, raising=False)
    logs = tmp_path / "logs"
    record_harness_agent_usage(
        session_hash="sess-abc",
        document_name="doc.pdf",
        time_taken=12.5,
        token_usage=HarnessTokenUsage(input_tokens=1000, output_tokens=200),
        pii_method="Local",
        text_extraction_method="Local text",
        cost_code="CC1",
        usage_logs_folder=str(logs),
    )
    csv_path = logs / "usage_log.csv"
    assert csv_path.is_file()
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    assert len(rows) >= 2
    data = rows[-1]
    headers = rows[0]
    assert "task" in headers
    task_idx = headers.index("task")
    assert data[task_idx] == AGENT_USAGE_TASK
    in_idx = headers.index("llm_total_input_tokens")
    out_idx = headers.index("llm_total_output_tokens")
    assert int(data[in_idx]) == 1000
    assert int(data[out_idx]) == 200
