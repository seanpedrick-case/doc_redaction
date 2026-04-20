"""
CLI entrypoint for packaging.

Re-exports the existing repo-root `cli_redact.py` implementation so that
`pyproject.toml` console scripts can target a stable package path.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

_root_cli = importlib.import_module("cli_redact")

build_cli_argument_parser = getattr(_root_cli, "build_cli_argument_parser")
get_cli_default_args_dict = getattr(_root_cli, "get_cli_default_args_dict")


def main(direct_mode_args: Dict[str, Any] | None = None):
    # Mirror the root signature but avoid a mutable default.
    if direct_mode_args is None:
        direct_mode_args = {}
    return _root_cli.main(direct_mode_args=direct_mode_args)


__all__ = ["build_cli_argument_parser", "get_cli_default_args_dict", "main"]
