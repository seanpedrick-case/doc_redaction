"""Device placement for transformers generation with accelerate device_map='auto'."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tools.llm_funcs import (
    _is_accelerate_dispatched_model,
    _transformers_input_device,
)


def test_dispatched_model_detected_from_hf_device_map():
    model = SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "lm_head": 1})
    assert _is_accelerate_dispatched_model(model) is True
    assert _is_accelerate_dispatched_model(SimpleNamespace()) is False


def test_input_device_uses_embedding_weights_not_generic_cuda():
    embed_device = SimpleNamespace(type="cuda", index=0)

    class _Emb:
        weight = SimpleNamespace(device=embed_device)

    model = MagicMock()
    model.get_input_embeddings.return_value = _Emb()
    model.hf_device_map = {"model.layers.0": "cuda:1"}
    model.language_model = None
    model.llm = None

    assert _transformers_input_device(model) is embed_device


def test_input_device_falls_back_to_nested_language_model_embeddings():
    embed_device = SimpleNamespace(type="cuda", index=1)

    class _Emb:
        weight = SimpleNamespace(device=embed_device)

    inner = SimpleNamespace(get_input_embeddings=lambda: _Emb())
    model = SimpleNamespace(
        language_model=inner,
        llm=None,
        model=None,
        hf_device_map={"visual": "cuda:0", "model": "cuda:1"},
    )

    def _no_embeddings():
        raise AttributeError("no embeddings")

    model.get_input_embeddings = _no_embeddings

    assert _transformers_input_device(model) is embed_device
