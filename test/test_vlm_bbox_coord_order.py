"""Tests for Gemma/Gemini yxyx VLM bounding-box conversion."""

import pytest

from tools.custom_image_analyser_engine import (
    _bbox_to_xyxy,
    _extract_openai_response_model_id,
    _fetch_inference_server_loaded_model_id,
    _inference_server_model_id_cache,
    _parse_vlm_line_item_to_geometry,
    _vlm_bbox_coord_order,
)


@pytest.fixture(autouse=True)
def _auto_bbox_order(monkeypatch):
    monkeypatch.setattr(
        "tools.custom_image_analyser_engine.VLM_BBOX_COORD_ORDER", "auto"
    )


def test_vlm_bbox_coord_order_gemma_ids():
    assert _vlm_bbox_coord_order("gemma-4-31B", config_order="auto") == "yxyx"
    assert (
        _vlm_bbox_coord_order("unsloth/gemma-4-31B-it-GGUF", config_order="auto")
        == "yxyx"
    )
    assert (
        _vlm_bbox_coord_order("gemma-4-31B-it-IQ4_NL.gguf", config_order="auto")
        == "yxyx"
    )
    assert _vlm_bbox_coord_order("unsloth/gemma-4-26B-A4B-it-GGUF") == "yxyx"
    assert _vlm_bbox_coord_order("Gemini") == "yxyx"
    assert _vlm_bbox_coord_order("paligemma-3b") == "yxyx"


def test_vlm_bbox_coord_order_qwen_ids():
    assert _vlm_bbox_coord_order("Qwen3.5-27B", config_order="auto") == "xyxy"
    assert _vlm_bbox_coord_order("unsloth/Qwen3.5-35B-A3B-GGUF") == "xyxy"
    assert _vlm_bbox_coord_order("qwen_3_5_27b") == "xyxy"
    assert _vlm_bbox_coord_order(None) == "xyxy"
    assert _vlm_bbox_coord_order("") == "xyxy"


def test_bbox_to_xyxy_swaps_gemma_native_line():
    # Native Gemma [y1, x1, y2, x2] for a wide horizontal line
    native = [50, 100, 80, 900]
    assert _bbox_to_xyxy(native, "yxyx") == [100, 50, 900, 80]
    assert _bbox_to_xyxy(native, "xyxy") == [50, 100, 80, 900]


def test_config_xyxy_overrides_gemma_detection():
    assert _vlm_bbox_coord_order("gemma-4-31B", config_order="xyxy") == "xyxy"
    assert _vlm_bbox_coord_order("Qwen3.5-27B", config_order="yxyx") == "yxyx"


def test_parse_vlm_line_item_gemma_yields_wide_box():
    item = {"bbox": [50, 100, 80, 900], "text": "Hello", "conf": 0.9}
    parsed = _parse_vlm_line_item_to_geometry(
        item, None, "test", model_name="gemma-4-31B"
    )
    assert parsed is not None
    text, xyxy, conf = parsed
    assert text == "Hello"
    assert xyxy == [100.0, 50.0, 900.0, 80.0]
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    assert width > height


def test_parse_vlm_line_item_qwen_keeps_xyxy():
    item = {"bbox": [100, 50, 900, 80], "text": "Hello", "conf": 0.9}
    parsed = _parse_vlm_line_item_to_geometry(
        item, None, "test", model_name="Qwen3.5-27B"
    )
    assert parsed is not None
    _text, xyxy, _conf = parsed
    assert xyxy == [100.0, 50.0, 900.0, 80.0]


def test_parse_vlm_line_item_force_xyxy_skips_gemma_swap():
    item = {"bbox": [50, 100, 80, 900], "text": "Hello", "conf": 0.9}
    parsed = _parse_vlm_line_item_to_geometry(
        item, None, "test", model_name="gemma-4-31B", bbox_order="xyxy"
    )
    assert parsed is not None
    _text, xyxy, _conf = parsed
    assert xyxy == [50.0, 100.0, 80.0, 900.0]


def test_extract_openai_response_model_id():
    assert (
        _extract_openai_response_model_id(
            {"model": "gemma-4-31B-it-IQ4_NL.gguf", "choices": []}
        )
        == "gemma-4-31B-it-IQ4_NL.gguf"
    )
    assert _extract_openai_response_model_id({"choices": []}) is None
    assert _extract_openai_response_model_id("not-a-dict") is None


def test_fetch_inference_server_loaded_model_id(monkeypatch):
    _inference_server_model_id_cache.clear()

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"}]}

    monkeypatch.setattr(
        "tools.custom_image_analyser_engine.requests.get",
        lambda *args, **kwargs: _Resp(),
    )
    mid = _fetch_inference_server_loaded_model_id("http://llama-inference:8080")
    assert mid == "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"
    # Cached: a second call must not need the network
    monkeypatch.setattr(
        "tools.custom_image_analyser_engine.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should use cache")
        ),
    )
    assert (
        _fetch_inference_server_loaded_model_id("http://llama-inference:8080")
        == "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"
    )
    _inference_server_model_id_cache.clear()
