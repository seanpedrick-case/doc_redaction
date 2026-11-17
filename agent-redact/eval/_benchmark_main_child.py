#!/usr/bin/env python3
"""Run cli_redact in a fresh process so APP_CONFIG_PATH reloads before tools.config."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

cfg = (os.environ.get("BENCHMARK_APP_CONFIG_PATH") or "").strip()
if cfg:
    os.environ["APP_CONFIG_PATH"] = str(Path(cfg).expanduser().resolve())

raw = os.environ.get("BENCHMARK_DIRECT_MODE_JSON") or "{}"
direct = json.loads(raw)
if not isinstance(direct, dict):
    raise SystemExit("BENCHMARK_DIRECT_MODE_JSON must be a JSON object")

import cli_redact  # noqa: E402

cli_redact.main(direct_mode_args=direct)
