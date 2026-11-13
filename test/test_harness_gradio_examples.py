"""Tests for main-app harness Gradio examples."""

from __future__ import annotations

import pytest

from tools.harness_gradio_examples import (
    _show_harness_examples,
    available_harness_examples,
    harness_example_rows,
    harness_examples_status_md,
)


def test_show_harness_examples_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "MAIN_APP_AGENT_EXAMPLES",
        "SHOW_AGENT_HARNESS_EXAMPLES",
        "SHOW_EXAMPLES",
    ):
        monkeypatch.delenv(key, raising=False)
    assert _show_harness_examples() is True


def test_show_harness_examples_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAIN_APP_AGENT_EXAMPLES", "false")
    assert _show_harness_examples() is False
    assert available_harness_examples() == []
    assert "disabled" in harness_examples_status_md().lower()


def test_harness_example_rows_shape() -> None:
    rows, labels = harness_example_rows(
        local_entities=["NAME"],
        comprehend_entities=[],
        llm_entities=["NAME"],
    )
    if not rows:
        pytest.skip("example PDFs not present in this checkout")
    assert len(rows) == len(labels)
    assert len(rows[0]) == 13
    assert isinstance(rows[0][0], str)
    assert rows[0][0].endswith(".pdf")


def test_env_override_show_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOW_EXAMPLES", "0")
    assert _show_harness_examples() is False
