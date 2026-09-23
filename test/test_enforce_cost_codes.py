"""Cost-code enforcement should not depend on empty Gradio State."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.helper_functions import (
    _coerce_cost_codes_dataframe,
    enforce_cost_codes,
)


def test_coerce_cost_codes_dataframe_accepts_gradio_dict():
    payload = {
        "headers": ["Cost code", "Description"],
        "data": [["CC1", "Team A"], ["CC2", "Team B"]],
    }
    df = _coerce_cost_codes_dataframe(payload)
    assert list(df.iloc[:, 0]) == ["CC1", "CC2"]


def test_enforce_cost_codes_allows_choice_in_visible_table():
    df = pd.DataFrame({"Cost code": ["CC1", "CC2"], "Description": ["A", "B"]})
    enforce_cost_codes(True, "CC1", df)


def test_enforce_cost_codes_reloads_csv_when_state_empty(tmp_path, monkeypatch):
    csv_path = tmp_path / "cost_codes.csv"
    csv_path.write_text("Cost code,Description\nCC9,Reloaded\n", encoding="utf-8")
    monkeypatch.setattr("tools.helper_functions.COST_CODES_PATH", str(csv_path))
    monkeypatch.setattr("tools.helper_functions.OUTPUT_COST_CODES_PATH", str(csv_path))

    enforce_cost_codes(True, "CC9", pd.DataFrame())


def test_enforce_cost_codes_still_errors_when_no_table_or_file(monkeypatch):
    monkeypatch.setattr("tools.helper_functions.COST_CODES_PATH", "")
    monkeypatch.setattr(
        "tools.helper_functions.OUTPUT_COST_CODES_PATH",
        str(Path("definitely_missing_cost_codes.csv")),
    )
    with pytest.raises(Exception, match="No cost codes present"):
        enforce_cost_codes(True, "DEFAULT", pd.DataFrame())
