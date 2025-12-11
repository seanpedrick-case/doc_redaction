import os
import sys
from threading import Thread

import gradio as gr
import spaces
from PIL import Image

from tools.config import (
    LOAD_PADDLE_AT_STARTUP,
    MAX_NEW_TOKENS,
    MAX_SPACES_GPU_RUN_TIME,
    PADDLE_DET_DB_UNCLIP_RATIO,
    PADDLE_FONT_PATH,
    PADDLE_MODEL_PATH,
    PADDLE_USE_TEXTLINE_ORIENTATION,
    QUANTISE_VLM_MODELS,
    REPORT_VLM_OUTPUTS_TO_GUI,
    SHOW_VLM_MODEL_OPTIONS,
    USE_FLASH_ATTENTION,
    VLM_DEFAULT_DO_SAMPLE,
    VLM_DEFAULT_MIN_P,
    VLM_DEFAULT_PRESENCE_PENALTY,
    VLM_DEFAULT_REPETITION_PENALTY,
    VLM_DEFAULT_TEMPERATURE,
    VLM_DEFAULT_TOP_K,
    VLM_DEFAULT_TOP_P,
    VLM_MAX_IMAGE_SIZE,
    VLM_MIN_IMAGE_SIZE,
    VLM_SEED,
)
from tools.helper_functions import get_system_font_path

if LOAD_PADDLE_AT_STARTUP is True:
    # Set PaddleOCR environment variables BEFORE importing PaddleOCR
    # This ensures fonts are configured before the package loads

    # Set PaddleOCR model directory environment variable (only if specified).
    if PADDLE_MODEL_PATH and PADDLE_MODEL_PATH.strip():
        os.environ["PADDLEOCR_MODEL_DIR"] = PADDLE_MODEL_PATH
        print(f"Setting PaddleOCR model path to: {PADDLE_MODEL_PATH}")
    else:
        print("Using default PaddleOCR model storage location")

    # Set PaddleOCR font path to use system fonts instead of downloading simfang.ttf/PingFang-SC-Regular.ttf
    # This MUST be set before importing PaddleOCR to prevent font downloads
    if (
        PADDLE_FONT_PATH
        and PADDLE_FONT_PATH.strip()
        and os.path.exists(PADDLE_FONT_PATH)
    ):
        os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = PADDLE_FONT_PATH
        print(f"Setting PaddleOCR font path to configured font: {PADDLE_FONT_PATH}")
    else:
        system_font_path = get_system_font_path()
        if system_font_path:
            os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = system_font_path
            print(f"Setting PaddleOCR font path to system font: {system_font_path}")
        else:
            print(
                "Warning: No suitable system font found. PaddleOCR may download default fonts."
            )

    try:
        from paddleocr import PaddleOCR

        print("PaddleOCR imported successfully")

        paddle_kwargs = None

        # Default paddle configuration if none provided
        if paddle_kwargs is None:
            paddle_kwargs = {
                "det_db_unclip_ratio": PADDLE_DET_DB_UNCLIP_RATIO,
                "use_textline_orientation": PADDLE_USE_TEXTLINE_ORIENTATION,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "lang": "en",
            }
        else:
            # Enforce language if not explicitly provided
            paddle_kwargs.setdefault("lang", "en")

        try:
            PaddleOCR(**paddle_kwargs)
        except Exception as e:
            # Handle DLL loading errors (common on Windows with GPU version)
            if (
                "WinError 127" in str(e)
                or "could not be found" in str(e).lower()
                or "dll" in str(e).lower()
            ):
                print(
                    f"Warning: GPU initialization failed (likely missing CUDA/cuDNN dependencies): {e}"
                )
                print("PaddleOCR will not be available. To fix GPU issues:")
                print("1. Install Visual C++ Redistributables (latest version)")
                print("2. Ensure CUDA runtime libraries are in your PATH")
                print(
                    "3. Or reinstall paddlepaddle CPU version: pip install paddlepaddle"
                )
                raise ImportError(
                    f"Error initializing PaddleOCR: {e}. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
                )
            else:
                raise e

    except ImportError:
        PaddleOCR = None
        print(
            "PaddleOCR not found. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
        )


# Define module-level defaults for model parameters (always available for import)
# These will be overridden inside the SHOW_VLM_MODEL_OPTIONS block if enabled
model_default_prompt = """Read all the text in the image."""
model_default_do_sample = (
    VLM_DEFAULT_DO_SAMPLE if VLM_DEFAULT_DO_SAMPLE is not None else None
)
model_default_top_p = VLM_DEFAULT_TOP_P if VLM_DEFAULT_TOP_P is not None else None
model_default_min_p = VLM_DEFAULT_MIN_P if VLM_DEFAULT_MIN_P is not None else None
model_default_top_k = VLM_DEFAULT_TOP_K if VLM_DEFAULT_TOP_K is not None else None
model_default_temperature = (
    VLM_DEFAULT_TEMPERATURE if VLM_DEFAULT_TEMPERATURE is not None else None
)
model_default_repetition_penalty = (
    VLM_DEFAULT_REPETITION_PENALTY
    if VLM_DEFAULT_REPETITION_PENALTY is not None
    else None
)
model_default_presence_penalty = VLM_DEFAULT_PRESENCE_PENALTY
model_default_max_new_tokens = int(MAX_NEW_TOKENS)
model_default_seed = VLM_SEED if VLM_SEED is not None else None


if SHOW_VLM_MODEL_OPTIONS is True:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        TextIteratorStreamer,
    )

    from tools.config import (
        MAX_NEW_TOKENS,
        MODEL_CACHE_PATH,
        QUANTISE_VLM_MODELS,
        SELECTED_MODEL,
        USE_FLASH_ATTENTION,
        VLM_DEFAULT_DO_SAMPLE,
        VLM_DEFAULT_MIN_P,
        VLM_DEFAULT_PRESENCE_PENALTY,
        VLM_DEFAULT_REPETITION_PENALTY,
        VLM_DEFAULT_TEMPERATURE,
        VLM_DEFAULT_TOP_K,
        VLM_DEFAULT_TOP_P,
        VLM_SEED,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("torch.__version__ =", torch.__version__)
    print("torch.version.cuda =", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    print("Using device:", device)

    CACHE_PATH = MODEL_CACHE_PATH
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # Initialize model and processor variables
    processor = None
    model = None

    # Initialize model-specific generation parameters (will be set by specific models if needed)
    # If config values are provided, use them; otherwise leave as None to use model defaults
    model_default_prompt = """Read all the text in the image."""
    model_default_do_sample = (
        VLM_DEFAULT_DO_SAMPLE if VLM_DEFAULT_DO_SAMPLE is not None else None
    )
    model_default_top_p = VLM_DEFAULT_TOP_P if VLM_DEFAULT_TOP_P is not None else None
    model_default_min_p = VLM_DEFAULT_MIN_P if VLM_DEFAULT_MIN_P is not None else None
    model_default_top_k = VLM_DEFAULT_TOP_K if VLM_DEFAULT_TOP_K is not None else None
    model_default_temperature = (
        VLM_DEFAULT_TEMPERATURE if VLM_DEFAULT_TEMPERATURE is not None else None
    )
    model_default_repetition_penalty = (
        VLM_DEFAULT_REPETITION_PENALTY
        if VLM_DEFAULT_REPETITION_PENALTY is not None
        else None
    )
    model_default_presence_penalty = VLM_DEFAULT_PRESENCE_PENALTY
    model_default_max_new_tokens = int(MAX_NEW_TOKENS)
    # Track which models support presence_penalty (only Qwen3-VL models currently)
    model_supports_presence_penalty = False
    model_default_seed = VLM_SEED if VLM_SEED is not None else None

    if USE_FLASH_ATTENTION is True:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    # Setup quantisation config if enabled
    quantization_config = None
    if QUANTISE_VLM_MODELS is True:
        if not torch.cuda.is_available():
            print(
                "Warning: 4-bit quantisation requires CUDA, but CUDA is not available."
            )
            print("Falling back to loading models without quantisation")
            quantization_config = None
        else:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("4-bit quantization enabled using bitsandbytes")
            except Exception as e:
                print(f"Warning: Could not setup bitsandbytes quantization: {e}")
                print("Falling back to loading models without quantization")
                quantization_config = None

    print(f"Loading vision model: {SELECTED_MODEL}")

    # Load only the selected model based on configuration
    if SELECTED_MODEL == "Nanonets-OCR2-3B":
        MODEL_ID = "nanonets/Nanonets-OCR2-3B"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()
        if quantization_config is None:
            model = model.to(device)

        model_default_prompt = """Extract the text from the above document as if you were reading it naturally."""

    elif SELECTED_MODEL == "Dots.OCR":
        # Download and patch Dots.OCR model
        model_path_d_local = snapshot_download(
            repo_id="rednote-hilab/dots.ocr",
            local_dir=os.path.join(CACHE_PATH, "dots.ocr"),
            max_workers=20,
            local_dir_use_symlinks=False,
        )

        config_file_path = os.path.join(model_path_d_local, "configuration_dots.py")

        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                input_code = f.read()

            lines = input_code.splitlines()
            if "class DotsVLProcessor" in input_code and not any(
                "attributes = " in line for line in lines
            ):
                output_lines = []
                for line in lines:
                    output_lines.append(line)
                    if line.strip().startswith("class DotsVLProcessor"):
                        output_lines.append(
                            '    attributes = ["image_processor", "tokenizer"]'
                        )

                with open(config_file_path, "w") as f:
                    f.write("\n".join(output_lines))
                print("Patched configuration_dots.py successfully.")

        sys.path.append(model_path_d_local)

        MODEL_ID = model_path_d_local
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "attn_implementation": attn_implementation,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs).eval()

        model_default_prompt = """Extract the text content from this image."""
        model_default_max_new_tokens = MAX_NEW_TOKENS

    elif SELECTED_MODEL == "Qwen3-VL-2B-Instruct":
        MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = "auto"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_do_sample = False
        model_default_top_p = 0.8
        model_default_min_p = 0.0
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = (
            False  # I found that this doesn't work when using transformers
        )

    elif SELECTED_MODEL == "Qwen3-VL-4B-Instruct":
        MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "attn_implementation": attn_implementation,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = "auto"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_do_sample = False
        model_default_top_p = 0.8
        model_default_min_p = 0.0
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = (
            False  # I found that this doesn't work when using transformers
        )
    elif SELECTED_MODEL == "Qwen3-VL-8B-Instruct":
        MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "attn_implementation": attn_implementation,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = "auto"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_do_sample = False
        model_default_top_p = 0.8
        model_default_min_p = 0.0
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = (
            False  # I found that this doesn't work when using transformers
        )

    elif SELECTED_MODEL == "Qwen3-VL-30B-A3B-Instruct":
        MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        from transformers import Qwen3VLMoeForConditionalGeneration

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "attn_implementation": attn_implementation,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = "auto"
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_do_sample = False
        model_default_top_p = 0.8
        model_default_min_p = 0.0
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = (
            False  # I found that this doesn't work when using transformers
        )

    elif SELECTED_MODEL == "Qwen3-VL-235B-A22B-Instruct":
        MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
        from transformers import Qwen3VLMoeForConditionalGeneration

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        load_kwargs = {
            "attn_implementation": attn_implementation,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = "auto"
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_do_sample = False
        model_default_top_p = 0.8
        model_default_min_p = 0.0
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = (
            False  # I found that this doesn't work when using transformers
        )

    elif SELECTED_MODEL == "PaddleOCR-VL":
        MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
        load_kwargs = {
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs).eval()
        if quantization_config is None:
            model = model.to(device)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        model_default_prompt = """OCR:"""
        model_default_max_new_tokens = MAX_NEW_TOKENS

    elif SELECTED_MODEL == "None":
        model = None
        processor = None

    else:
        raise ValueError(
            f"Invalid model selected: {SELECTED_MODEL}. Valid options are: Nanonets-OCR2-3B, Dots.OCR, Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-30B-A3B-Instruct, PaddleOCR-VL"
        )

    # Override model defaults with user-provided config values if they are set
    # Priority: user config value > model default
    if VLM_DEFAULT_DO_SAMPLE is not None:
        model_default_do_sample = VLM_DEFAULT_DO_SAMPLE
    if VLM_DEFAULT_TOP_P is not None:
        model_default_top_p = VLM_DEFAULT_TOP_P
    if VLM_DEFAULT_MIN_P is not None:
        model_default_min_p = VLM_DEFAULT_MIN_P
    if VLM_DEFAULT_TOP_K is not None:
        model_default_top_k = VLM_DEFAULT_TOP_K
    if VLM_DEFAULT_TEMPERATURE is not None:
        model_default_temperature = VLM_DEFAULT_TEMPERATURE
    if VLM_DEFAULT_REPETITION_PENALTY is not None:
        model_default_repetition_penalty = VLM_DEFAULT_REPETITION_PENALTY
    if VLM_DEFAULT_PRESENCE_PENALTY is not None:
        model_default_presence_penalty = VLM_DEFAULT_PRESENCE_PENALTY
    if VLM_SEED is not None:
        model_default_seed = VLM_SEED

    print(f"Successfully loaded {SELECTED_MODEL}")


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def extract_text_from_image_vlm(
    text: str,
    image: Image.Image,
    max_new_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    min_p: float = None,
    top_k: int = None,
    repetition_penalty: float = None,
    do_sample: bool = None,
    presence_penalty: float = None,
    seed: int = None,
    model_default_prompt: str = None,
):
    """
    Generates responses using the configured vision model for image input.
    Streams text to console and returns complete text only at the end.

    Uses model-specific defaults if they were set during model initialization,
    falling back to function argument defaults if provided, and finally to sensible
    general defaults if neither are available.

    Args:
        text (str): The text prompt to send to the vision model. If empty and model
            has a default prompt, the model default will be used.
        image (Image.Image): The PIL Image to process. Must not be None.
        max_new_tokens (int, optional): Maximum number of new tokens to generate.
            Defaults to model-specific value (MAX_NEW_TOKENS for models with defaults) or MAX_NEW_TOKENS from config.
        temperature (float, optional): Sampling temperature for generation.
            Defaults to model-specific value (0.7 for Qwen3-VL models) or 0.7.
        top_p (float, optional): Nucleus sampling parameter (top-p).
            Defaults to model-specific value (0.8 for Qwen3-VL models) or 0.9.
        min_p (float, optional): Minimum probability threshold for token sampling.
            Defaults to model-specific value or 0.0.
        top_k (int, optional): Top-k sampling parameter.
            Defaults to model-specific value (20 for Qwen3-VL models) or 50.
        repetition_penalty (float, optional): Penalty for token repetition.
            Defaults to model-specific value (1.0 for Qwen3-VL models) or 1.3.
        do_sample (bool, optional): If True, use sampling (do_sample=True).
            If False, use sampling (do_sample=True). If None, defaults to False
            (sampling) for Qwen3-VL models, or True (sampling) for other models.
        presence_penalty (float, optional): Penalty for token presence.
            Defaults to model-specific value (1.5 for Qwen3-VL models) or None.
            Note: Not all models support this parameter.
        seed (int, optional): Random seed for generation. If None, uses VLM_SEED
            from config if set, otherwise no seed is set (non-deterministic).
        model_default_prompt (str, optional): The default prompt to use if no text is provided.
            Defaults to model-specific value (None for Dots.OCR, "Read all the text in the image." for Qwen3-VL models) or "Read all the text in the image."

    Returns:
        str: The complete generated text response from the model.
    """
    if image is None:
        return "Please upload an image."

    # Determine parameter values with priority: function args > model defaults > general defaults
    # Priority order: function argument (if not None) > model default > general default

    # Text/prompt handling
    if text and text.strip():
        actual_text = text
    elif model_default_prompt is not None:
        actual_text = model_default_prompt
    else:
        actual_text = "Read all the text in the image."  # General default

    # max_new_tokens: function arg > model default > general default
    if max_new_tokens is not None:
        actual_max_new_tokens = max_new_tokens
    elif model_default_max_new_tokens is not None:
        actual_max_new_tokens = model_default_max_new_tokens
    else:
        actual_max_new_tokens = MAX_NEW_TOKENS  # General default (from config)

    # temperature: function arg > model default (which may include config override)
    if temperature is not None:
        actual_temperature = temperature
    elif model_default_temperature is not None:
        actual_temperature = model_default_temperature
    else:
        # Fallback to a sensible default if neither function arg nor model default is set
        actual_temperature = 0.1

    # top_p: function arg > model default (which may include config override)
    if top_p is not None:
        actual_top_p = top_p
    elif model_default_top_p is not None:
        actual_top_p = model_default_top_p
    else:
        # Fallback to a sensible default if neither function arg nor model default is set
        actual_top_p = 0.8

    # min_p: function arg > model default (which may include config override)
    if min_p is not None:
        actual_min_p = min_p
    elif model_default_min_p is not None:
        actual_min_p = model_default_min_p
    else:
        # Fallback to a sensible default if neither function arg nor model default is set
        actual_min_p = 0.0

    # top_k: function arg > model default (which may include config override)
    if top_k is not None:
        actual_top_k = top_k
    elif model_default_top_k is not None:
        actual_top_k = model_default_top_k
    else:
        # Fallback to a sensible default if neither function arg nor model default is set
        actual_top_k = 20

    # repetition_penalty: function arg > model default (which may include config override)
    if repetition_penalty is not None:
        actual_repetition_penalty = repetition_penalty
    elif model_default_repetition_penalty is not None:
        actual_repetition_penalty = model_default_repetition_penalty
    else:
        # Fallback to a sensible default if neither function arg nor model default is set
        actual_repetition_penalty = 1.0

    # do_sample: function arg > model default (which may include config override)
    if do_sample is not None:
        actual_do_sample = do_sample
    elif model_default_do_sample is not None:
        actual_do_sample = model_default_do_sample
    else:
        # Fallback to a sensible default if neither function arg nor model default is set
        actual_do_sample = True

    # presence_penalty: function arg > model default (which may include config override) > None
    actual_presence_penalty = None
    if presence_penalty is not None:
        actual_presence_penalty = presence_penalty
    elif model_default_presence_penalty is not None:
        actual_presence_penalty = model_default_presence_penalty

    # seed: function arg > model default (which may include config override)
    actual_seed = None
    if seed is not None:
        actual_seed = seed
    elif model_default_seed is not None:
        actual_seed = model_default_seed

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": actual_text},
            ],
        }
    ]
    prompt_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=VLM_MIN_IMAGE_SIZE,
        max_pixels=VLM_MAX_IMAGE_SIZE,
    ).to(device)

    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )

    # Set random seed if specified
    if actual_seed is not None:
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

    # Build generation kwargs with resolved parameters
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": actual_max_new_tokens,
        "do_sample": actual_do_sample,
        "temperature": actual_temperature,
        "top_p": actual_top_p,
        "min_p": actual_min_p,
        "top_k": actual_top_k,
        "repetition_penalty": actual_repetition_penalty,
    }

    # Add presence_penalty if it's set and the model supports it
    # Only Qwen3-VL models currently support presence_penalty
    if actual_presence_penalty is not None and model_supports_presence_penalty:
        generation_kwargs["presence_penalty"] = actual_presence_penalty
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    line_buffer = ""  # Accumulate text for the current line
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        line_buffer += new_text

        # Print to console as it streams
        print(new_text, end="", flush=True)

        # If we hit a newline, report the entire accumulated line to GUI
        if REPORT_VLM_OUTPUTS_TO_GUI and "\n" in new_text:
            # Split by newline to handle the line(s) we just completed
            parts = line_buffer.split("\n")
            # Report all complete lines (everything except the last part which may be incomplete)
            for line in parts[:-1]:
                if line.strip():  # Only report non-empty lines
                    gr.Info(line, duration=2)
            # Keep the last part (after the last newline) for the next line
            line_buffer = parts[-1] if parts else ""

        # time.sleep(0.01)

    # Print final newline after streaming is complete
    print()  # Add newline at the end

    # Return the complete text only at the end
    return buffer


full_page_ocr_vlm_prompt = """Spot all the text in the image at line-level, and output in JSON format as [{'bb': [x1, y1, x2, y2], 'text': 'identified text'}, ...].

IMPORTANT: Extract each horizontal line of text separately. Do NOT combine multiple lines into paragraphs. Each line that appears on a separate horizontal row in the image should be a separate entry.

Rules:
- Each line must be on a separate horizontal row in the image
- Even if a sentence is split over multiple horizontal lines, it should be split into separate entries (one per line)
- If text spans multiple horizontal lines, split it into separate entries (one per line)
- Do NOT combine lines that appear on different horizontal rows
- Each bounding box should tightly fit around a single horizontal line of text
- Empty lines should be skipped

# Only return valid JSON, no additional text or explanation."""

full_page_ocr_people_vlm_prompt = """Spot all photos of people's faces in the image, and output in JSON format as [{'bb': [x1, y1, x2, y2], 'text': '[PERSON]'}, ...].

Always return the JSON format as [{'bb': [x1, y1, x2, y2], 'text': '[PERSON]'}, ...].

Rules:
- Each photo of a person's face must be a separate entry.
- Do NOT combine multiple photos into a single entry.
- Each photo of a person's face that appears in the image should be a separate entry.
- 'text' should always be exactly '[PERSON]'.
- Do NOT include any other text or information in the JSON.
- If there are no photos of people's faces in the image, return an empty JSON array.

# Only return valid JSON, no additional text or explanation."""

full_page_ocr_signature_vlm_prompt = """Spot all signatures in the image, and output in JSON format as [{'bb': [x1, y1, x2, y2], 'text': '[SIGNATURE]'}, ...].

Always return the JSON format as [{'bb': [x1, y1, x2, y2], 'text': '[SIGNATURE]'}, ...].

Rules:
- Each signature must be a separate entry.
- Do NOT combine multiple signatures into a single entry.
- Each signature that appears in the image should be a separate entry.
- 'text' should always be exactly '[SIGNATURE]'.
- Do NOT include any other text or information in the JSON.
- If there are no signatures in the image, return an empty JSON array.

# Only return valid JSON, no additional text or explanation."""

# Test for word-level OCR with VLMs - makes some mistakes but not bad
# full_page_ocr_vlm_prompt = """Spot all the text in the image at word-level, and output in JSON format as [{'bb': [x1, y1, x2, y2], 'text': 'identified word'}, ...].

# IMPORTANT: Extract each word in the image separately. Do NOT combine words into longer fragments, sentences, or paragraphs. Each entry must correspond to a single, individual word as visually separated in the image.

# Rules:
# - Each entry should correspond to a single distinct word (not groups of words, not whole lines).
# - For each word, provide a tight bounding box [x1, y1, x2, y2] around just that word.
# - Do not merge words. Do not split words into letters. Only return one entry per word.
# - Maintain the order of words as they appear spatially from top to bottom, left to right.
# - Skip any empty or whitespace-only entries.
# - Do not include extraneous text, explanations, or formatting beyond the required JSON.

# Only return valid JSON, no additional text or explanation."""
