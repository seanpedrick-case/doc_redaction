"""Unit tests for the main-app AgentCore Harness driver (no AWS calls)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from tools import agentcore_harness as ah
from tools.agentcore_harness_stream import (
    HarnessTurnCollector,
    load_inline_tool_specs,
    tool_result_message,
)

HARNESS_ARN = "arn:aws:bedrock-agentcore:eu-west-2:123456789012:harness/test-abc"
REVIEW_HEADER = "image,page,label,color,xmin,ymin,xmax,ymax,text,id\n"


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}: John Smith, 1 High Street")
    doc.save(path)
    doc.close()
    return path


def _session(sample_pdf: Path, tmp_path: Path, **kwargs) -> ah.HarnessSession:
    kwargs.setdefault("staging_prefix", "")
    return ah.HarnessSession(
        sample_pdf, output_root=tmp_path / "out", run_id="run1", **kwargs
    )


def _text_turn(text: str, stop: str = "end_turn") -> list[dict]:
    return [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
        {"messageStop": {"stopReason": stop}},
    ]


def _tool_turn(*calls: tuple[str, str, dict]) -> list[dict]:
    events: list[dict] = []
    for index, (tool_use_id, name, arguments) in enumerate(calls):
        events.append(
            {
                "contentBlockStart": {
                    "contentBlockIndex": index,
                    "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}},
                }
            }
        )
        events.append(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": index,
                    "delta": {"toolUse": {"input": json.dumps(arguments)}},
                }
            }
        )
    events.append({"messageStop": {"stopReason": "tool_use"}})
    return events


class FakeHarnessClient:
    def __init__(self, turns: list[list[dict]]):
        self.turns = list(turns)
        self.requests: list[dict] = []

    def invoke_harness(self, **request):
        self.requests.append(json.loads(json.dumps(request)))
        events = self.turns.pop(0) if self.turns else _text_turn("done")
        return {"stream": iter(events)}


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def generate_presigned_url(self, method, Params=None, ExpiresIn=None):
        return f"https://s3.example/{method}/{Params['Bucket']}/{Params['Key']}"

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            err = Exception("missing")
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def _run(session, client, **kwargs):
    return list(
        ah.run_agentic_redaction(
            session.source_pdf,
            "- Redact all personal names",
            session=session,
            client=client,
            harness_arn=HARNESS_ARN,
            endpoint="DEFAULT",
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


def test_collector_accumulates_inline_tool_input():
    collector = HarnessTurnCollector()
    for event in _tool_turn(("t1", "list_session_files", {"a": 1})):
        collector.feed(event)
    pending = collector.turn.pending_inline_calls({"list_session_files"})
    assert [p.tool_use_id for p in pending] == ["t1"]
    assert pending[0].arguments() == {"a": 1}


def test_collector_ignores_harness_resolved_tools():
    collector = HarnessTurnCollector()
    events = _tool_turn(("t1", "list_session_files", {}))
    events.insert(
        -1,
        {
            "contentBlockStart": {
                "contentBlockIndex": 5,
                "start": {"toolResult": {"toolUseId": "t1", "status": "success"}},
            }
        },
    )
    for event in events:
        collector.feed(event)
    assert collector.turn.pending_inline_calls({"list_session_files"}) == []


def test_tool_result_message_shape():
    msg = tool_result_message([("t1", "{}", True), ("t2", "bad", False)])
    assert msg["role"] == "user"
    statuses = [c["toolResult"]["status"] for c in msg["content"]]
    assert statuses == ["success", "error"]


def test_inline_tool_specs_are_well_formed():
    specs = load_inline_tool_specs()
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names))
    for spec in specs:
        assert spec["type"] == "inline_function"
        inline = spec["config"]["inlineFunction"]
        assert inline["description"]
        assert inline["inputSchema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Handoff loop
# ---------------------------------------------------------------------------


def test_text_only_run_finishes_in_one_round(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    client = FakeHarnessClient([_text_turn("All done.")])
    progress = _run(session, client)
    assert progress[-1].done
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["harnessArn"] == HARNESS_ARN
    assert request["runtimeSessionId"].startswith("harness-")
    assert "qualifier" not in request
    assert "User redaction requirements" in request["messages"][0]["content"][0]["text"]
    assert {t["name"] for t in request["tools"]} >= {"doc_redact", "finish_task"}


def test_single_inline_handoff_returns_tool_result(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    client = FakeHarnessClient(
        [_tool_turn(("t1", "list_session_files", {})), _text_turn("ok")]
    )
    _run(session, client)
    assert len(client.requests) == 2
    second = client.requests[1]
    assert second["runtimeSessionId"] == client.requests[0]["runtimeSessionId"]
    result = second["messages"][0]["content"][0]["toolResult"]
    assert result["toolUseId"] == "t1"
    assert result["status"] == "success"
    assert "source/sample.pdf" in result["content"][0]["text"]


def test_adjacent_inline_calls_returned_together(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    client = FakeHarnessClient(
        [
            _tool_turn(
                ("t1", "list_session_files", {}),
                ("t2", "read_session_text", {"path": "../../etc/passwd"}),
            ),
            _text_turn("ok"),
        ]
    )
    _run(session, client)
    content = client.requests[1]["messages"][0]["content"]
    assert [c["toolResult"]["toolUseId"] for c in content] == ["t1", "t2"]
    assert [c["toolResult"]["status"] for c in content] == ["success", "error"]


def test_finish_task_ends_run_and_sets_outputs(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    client = FakeHarnessClient(
        [
            _tool_turn(
                (
                    "t1",
                    "finish_task",
                    {
                        "summary": "Redacted names.",
                        "final_files": ["source/sample.pdf"],
                    },
                )
            ),
            _text_turn("should not be requested"),
        ]
    )
    progress = _run(session, client)
    assert len(client.requests) == 1
    assert session.finished
    assert progress[-1].done
    assert progress[-1].output_files == [str(session.source_pdf)]
    assert "Redacted names." in progress[-1].summary


def test_max_rounds_limits_invocations(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    turns = [_tool_turn((f"t{i}", "list_session_files", {})) for i in range(10)]
    client = FakeHarnessClient(turns)
    progress = _run(session, client, max_rounds=3)
    assert len(client.requests) == 3
    assert progress[-1].done


def test_should_stop_aborts_stream(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    client = FakeHarnessClient([_tool_turn(("t1", "list_session_files", {}))])
    progress = _run(session, client, should_stop=lambda: True)
    assert len(client.requests) == 1
    assert "stopped" in progress[-1].summary.lower()


def test_endpoint_qualifier_is_sent(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    client = FakeHarnessClient([_text_turn("ok")])
    list(
        ah.run_agentic_redaction(
            session.source_pdf,
            "- Redact names",
            session=session,
            client=client,
            harness_arn=HARNESS_ARN,
            endpoint="prod",
        )
    )
    assert client.requests[0]["qualifier"] == "prod"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def test_paths_are_confined_to_session(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    for bad in ("../outside.csv", str(tmp_path / "elsewhere.csv")):
        text, ok = session.dispatch("read_session_text", {"path": bad})
        assert not ok
        assert "error" in json.loads(text)


def test_write_cannot_overwrite_source(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    _, ok = session.dispatch(
        "write_session_text", {"path": "source/sample.pdf", "content": "x"}
    )
    assert not ok


def test_write_review_csv_validates_and_repairs(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    good = REVIEW_HEADER + "img,1,PERSON,black,0.1,0.1,0.2,0.2,John,a1\n"
    text, ok = session.dispatch(
        "write_session_text", {"path": "edits/doc_review_file.csv", "content": good}
    )
    assert ok, text
    saved = session.resolve("edits/doc_review_file.csv").read_text(encoding="utf-8-sig")
    assert '"(0, 0, 0)"' in saved

    bad = REVIEW_HEADER + "img,1,PERSON,(0, 0, 0),12,30,40,50,John,a1\n"
    text, ok = session.dispatch(
        "write_session_text", {"path": "edits/doc_review_file.csv", "content": bad}
    )
    assert not ok


def test_page_review_disabled_without_staging(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    text, ok = session.dispatch("stage_page_review_bundles", {})
    assert not ok
    assert "not available" in json.loads(text)["error"]


def test_stage_and_merge_page_reviews(sample_pdf, tmp_path):
    s3 = FakeS3()
    session = _session(
        sample_pdf,
        tmp_path,
        staging_prefix="s3://bucket/harness-staging/",
        s3_client=s3,
        enable_page_review=True,
    )
    redact_dir = session.session_dir / "output_redact"
    redact_dir.mkdir(parents=True)
    (redact_dir / "sample_review_file.csv").write_text(
        REVIEW_HEADER
        + 'img1,1,PERSON,"(0, 0, 0)",0.1,0.1,0.2,0.12,John,a\n'
        + 'img2,2,PERSON,"(0, 0, 0)",0.1,0.1,0.2,0.12,John,b\n',
        encoding="utf-8-sig",
    )
    (redact_dir / "sample_ocr_results_with_words.csv").write_text(
        "page,line,word_text,word_x0,word_y0,word_x1,word_y1\n"
        "1,1,John,0.1,0.1,0.2,0.12\n2,1,John,0.1,0.1,0.2,0.12\n",
        encoding="utf-8-sig",
    )
    review_rel = "output_redact/sample_review_file.csv"

    text, ok = session.dispatch(
        "stage_page_review_bundles", {"review_csv_path": review_rel}
    )
    assert ok, text
    staged = json.loads(text)
    assert [b["page"] for b in staged["pages"]] == [1, 2]
    assert staged["pages"][0]["result_put_url"].startswith(
        "https://s3.example/put_object/"
    )

    # Page 1 subagent uploads a result that drops the row and adds one with a
    # wrong page number (which the merge must discard); page 2 never uploads.
    result_key = session.staged_pages[1]["result_key"]
    s3.objects[("bucket", result_key)] = (
        REVIEW_HEADER
        + 'img1,1,ADDRESS,"(0, 0, 0)",0.3,0.1,0.5,0.12,1 High Street,\n'
        + 'img2,2,PERSON,"(0, 0, 0)",0.1,0.1,0.2,0.12,Wrong page,z\n'
    ).encode("utf-8")

    text, ok = session.dispatch("merge_page_reviews", {"review_csv_path": review_rel})
    assert ok, text
    merged = json.loads(text)
    assert merged["pages_replaced"] == {"1": 1}
    assert merged["pages_without_upload_kept_master_rows"] == [2]
    assert merged["rows_dropped_for_wrong_page"] == 1

    out = session.resolve(merged["merged_review_csv_path"]).read_text(
        encoding="utf-8-sig"
    )
    assert "1 High Street" in out
    assert "Wrong page" not in out
    assert out.count("John") == 1  # page 1 John removed, page 2 John kept
    assert merged["merged_review_csv_path"].endswith("sample_review_file_merged.csv")


# ---------------------------------------------------------------------------
# Prompt and config gating
# ---------------------------------------------------------------------------


def test_prompt_puts_user_requirements_last(sample_pdf, tmp_path):
    session = _session(sample_pdf, tmp_path)
    prompt = ah.build_harness_task_prompt(
        session, "Redact all personal names\n- Keep job titles"
    )
    tail = prompt[prompt.rindex("## User redaction requirements") :]
    assert "- Redact all personal names" in tail
    assert "- Keep job titles" in tail
    assert "source/sample.pdf" in prompt
    assert "{INPUT_PATH}" not in prompt


def test_feature_ready_requires_valid_arn(monkeypatch):
    monkeypatch.setattr(ah, "AGENTCORE_HARNESS_ARN", "")
    ready, reason = ah.harness_feature_ready()
    assert not ready and "AGENTCORE_HARNESS_ARN" in reason

    monkeypatch.setattr(ah, "AGENTCORE_HARNESS_ARN", "not-an-arn")
    assert not ah.harness_feature_ready()[0]

    monkeypatch.setattr(ah, "AGENTCORE_HARNESS_ARN", HARNESS_ARN)
    assert ah.harness_feature_ready() == (True, "")


def test_agentic_redact_api_rejects_when_not_configured(monkeypatch, sample_pdf):
    monkeypatch.setattr(ah, "AGENTCORE_HARNESS_ARN", "")
    with pytest.raises(ValueError, match="not configured"):
        ah.agentic_redact_api(str(sample_pdf), "- Redact names")


def test_ui_ocr_method_mapping():
    from tools.config import (
        LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION,
        SELECTABLE_TEXT_EXTRACT_OPTION,
        TEXTRACT_TEXT_EXTRACT_OPTION,
    )

    assert ah.ui_ocr_method_to_tool_value(SELECTABLE_TEXT_EXTRACT_OPTION, "") == (
        "Local text"
    )
    assert ah.ui_ocr_method_to_tool_value(TEXTRACT_TEXT_EXTRACT_OPTION, "") == (
        "AWS Textract"
    )
    assert (
        ah.ui_ocr_method_to_tool_value(LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION, "paddle")
        == "paddle"
    )


def test_ui_entities_follow_pii_method():
    from tools.config import AWS_PII_OPTION, LOCAL_PII_OPTION

    assert ah.ui_entities_for_pii_method(AWS_PII_OPTION, ["L"], ["C"], ["M"]) == ["C"]
    assert ah.ui_entities_for_pii_method(LOCAL_PII_OPTION, ["L"], ["C"], ["M"]) == ["L"]
