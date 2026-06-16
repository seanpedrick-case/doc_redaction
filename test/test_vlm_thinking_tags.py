import json

from tools.helper_functions import extract_balanced_json_array, strip_vlm_thinking_tags

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
