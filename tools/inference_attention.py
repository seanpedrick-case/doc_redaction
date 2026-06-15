"""Shared attention-backend selection for transformers / PaddleOCR inference."""

from __future__ import annotations

from tools.config import USE_FLASH_ATTENTION


def flash_attention_is_usable() -> bool:
    """True only when flash-attn is installed *and* transformers can call its kernels."""
    if not USE_FLASH_ATTENTION:
        return False
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available

        if not is_flash_attn_2_available():
            return False
        from transformers.modeling_flash_attention_utils import (
            lazy_import_flash_attention,
        )

        (flash_fn, _, _, _, _), process_fn = lazy_import_flash_attention(
            "flash_attention_2"
        )
        return callable(flash_fn) and callable(process_fn)
    except Exception:
        return False


def resolve_attn_implementation() -> str:
    """
    Pick an attention backend for VLM transformers models (Qwen, etc.).

    Uses flash_attention_2 only when USE_FLASH_ATTENTION=True and the installed
    flash-attn wheel matches torch (otherwise transformers can raise
    ``TypeError: the first argument must be callable`` mid-inference).
    Falls back to sdpa on GPU (faster than eager; no flash-attn dependency).
    """
    if flash_attention_is_usable():
        return "flash_attention_2"
    return "sdpa"


def resolve_paddle_attn_implementation() -> str:
    """
    Attention backend for PaddleOCR PP-OCRv6 transformers models.

    These architectures do not support sdpa yet (transformers raises ValueError).
    Use flash_attention_2 when a compatible flash-attn wheel is available,
    otherwise eager (still runs on GPU via device=gpu:0).
    """
    if flash_attention_is_usable():
        return "flash_attention_2"
    return "eager"


def log_attn_implementation_choice() -> None:
    chosen = resolve_attn_implementation()
    if USE_FLASH_ATTENTION and chosen != "flash_attention_2":
        print(
            "Warning: USE_FLASH_ATTENTION=True but flash-attn is unavailable or "
            f"incompatible with this torch build; using attn_implementation={chosen!r}. "
            "Install a matching flash-attn wheel for your torch/CUDA/Python versions, "
            "or set USE_FLASH_ATTENTION=False."
        )
    else:
        print(f"Using attn_implementation={chosen!r}")
