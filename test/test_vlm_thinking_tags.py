import json

from tools.helper_functions import (
    extract_balanced_json_array,
    extract_last_balanced_json_array,
    strip_vlm_thinking_tags,
)
from tools.run_vlm import render_vlm_generation_prompt

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"


def test_strip_vlm_thinking_tags_orphan_close():
    raw = f"""{_THINK_CLOSE}

[
\t{{"bbox_2d": [34, 40, 153, 142], "text": "[FACE]", "conf": 0.95}}
]"""
    cleaned = strip_vlm_thinking_tags(raw)
    arr = extract_balanced_json_array(cleaned)
    assert arr is not None
    data = json.loads(arr)
    assert data[0]["text"] == "[FACE]"


def test_strip_vlm_thinking_tags_full_block():
    raw = f"{_THINK_OPEN}planning{_THINK_CLOSE}\n" + '[{"text": "hello", "conf": 0.9}]'
    cleaned = strip_vlm_thinking_tags(raw)
    assert "think" not in cleaned.lower()
    assert json.loads(cleaned)[0]["text"] == "hello"


def test_extract_last_balanced_json_array_skips_thinking_draft():
    raw = 'draft [1, 2\n[{"bbox": [10, 20, 30, 40], "text": "hello"}]'
    last = extract_last_balanced_json_array(raw)
    assert last is not None
    data = json.loads(last)
    assert data[0]["text"] == "hello"


def test_render_vlm_generation_prompt_passes_enable_thinking_false():
    captured = {}

    class _Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prompt = render_vlm_generation_prompt(
        _Processor(),
        [{"role": "user", "content": "hi"}],
        disable_thinking=True,
    )
    assert captured.get("enable_thinking") is False
    assert prompt.count(_THINK_CLOSE) == 1


def test_render_vlm_generation_prompt_appends_suffix_if_template_leaves_think_open():
    class _Processor:
        def apply_chat_template(self, messages, **kwargs):
            return "<|im_start|>assistant\n<think>\n"

    prompt = render_vlm_generation_prompt(
        _Processor(),
        [{"role": "user", "content": "hi"}],
        disable_thinking=True,
    )
    assert prompt.rstrip().endswith(_THINK_CLOSE)
