import os
import sys
import time
from threading import Thread

import spaces
from PIL import Image

from tools.config import (
    LOAD_PADDLE_AT_STARTUP,
    MAX_SPACES_GPU_RUN_TIME,
    PADDLE_DET_DB_UNCLIP_RATIO,
    PADDLE_MODEL_PATH,
    PADDLE_USE_TEXTLINE_ORIENTATION,
    QUANTISE_VLM_MODELS,
    SHOW_VLM_MODEL_OPTIONS,
)

if LOAD_PADDLE_AT_STARTUP is True:
    try:
        from paddleocr import PaddleOCR

        paddle_kwargs = None

        # Set PaddleOCR model directory environment variable (only if specified).
        if PADDLE_MODEL_PATH and PADDLE_MODEL_PATH.strip():
            os.environ["PADDLEOCR_MODEL_DIR"] = PADDLE_MODEL_PATH
            print(f"Setting PaddleOCR model path to: {PADDLE_MODEL_PATH}")
        else:
            print("Using default PaddleOCR model storage location")

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

        PaddleOCR(**paddle_kwargs)

    except ImportError:
        PaddleOCR = None
        print(
            "PaddleOCR not found. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
        )


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
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
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
    model_default_prompt = None
    model_default_greedy = None
    model_default_top_p = None
    model_default_top_k = None
    model_default_temperature = None
    model_default_repetition_penalty = None
    model_default_presence_penalty = None
    model_default_max_new_tokens = None
    # Track which models support presence_penalty (only Qwen3-VL models currently)
    model_supports_presence_penalty = False

    if USE_FLASH_ATTENTION is True:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    # Setup quantisation config if enabled
    quantization_config = None
    if QUANTISE_VLM_MODELS is True:
        if not torch.cuda.is_available():
            print("Warning: 4-bit quantisation requires CUDA, but CUDA is not available.")
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
        model_default_greedy = False  # 'false' string converted to boolean
        model_default_top_p = 1.0
        model_default_top_k = 40
        model_default_temperature = 1.0
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 2.0
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = False # I found that this doesn't work when using transformers

    elif SELECTED_MODEL == "Qwen3-VL-4B-Instruct":
        MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
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
        model_default_greedy = False  # 'false' string converted to boolean
        model_default_top_p = 0.8
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS
        model_supports_presence_penalty = False # I found that this doesn't work when using transformers

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
            f"Invalid model selected: {SELECTED_MODEL}. Valid options are: Nanonets-OCR2-3B, Dots.OCR, Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, PaddleOCR-VL"
        )

    print(f"Successfully loaded {SELECTED_MODEL}")


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def extract_text_from_image_vlm(
    text: str,
    image: Image.Image,
    max_new_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    repetition_penalty: float = None,
    greedy: bool = None,
    presence_penalty: float = None,
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
        top_k (int, optional): Top-k sampling parameter.
            Defaults to model-specific value (20 for Qwen3-VL models) or 50.
        repetition_penalty (float, optional): Penalty for token repetition.
            Defaults to model-specific value (1.0 for Qwen3-VL models) or 1.3.
        greedy (bool, optional): If True, use greedy decoding (do_sample=False).
            If False, use sampling (do_sample=True). If None, defaults to False
            (sampling) for Qwen3-VL models, or True (sampling) for other models.
        presence_penalty (float, optional): Penalty for token presence.
            Defaults to model-specific value (1.5 for Qwen3-VL models) or None.
            Note: Not all models support this parameter.

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

    # temperature: function arg > model default > general default
    if temperature is not None:
        actual_temperature = temperature
    elif model_default_temperature is not None:
        actual_temperature = model_default_temperature
    else:
        actual_temperature = 0.7  # General default

    # top_p: function arg > model default > general default
    if top_p is not None:
        actual_top_p = top_p
    elif model_default_top_p is not None:
        actual_top_p = model_default_top_p
    else:
        actual_top_p = 0.9  # General default

    # top_k: function arg > model default > general default
    if top_k is not None:
        actual_top_k = top_k
    elif model_default_top_k is not None:
        actual_top_k = model_default_top_k
    else:
        actual_top_k = 50  # General default

    # repetition_penalty: function arg > model default > general default
    if repetition_penalty is not None:
        actual_repetition_penalty = repetition_penalty
    elif model_default_repetition_penalty is not None:
        actual_repetition_penalty = model_default_repetition_penalty
    else:
        actual_repetition_penalty = 1.3  # General default

    # greedy/do_sample: function arg > model default > general default
    # greedy=False means do_sample=True (sampling), greedy=True means do_sample=False (greedy)
    if greedy is not None:
        actual_do_sample = not greedy
    elif model_default_greedy is not None:
        actual_do_sample = not model_default_greedy
    else:
        actual_do_sample = True  # General default: use sampling

    # presence_penalty: function arg > model default > None (not used if not available)
    actual_presence_penalty = None
    if presence_penalty is not None:
        actual_presence_penalty = presence_penalty
    elif model_default_presence_penalty is not None:
        actual_presence_penalty = model_default_presence_penalty

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
        text=[prompt_full], images=[image], return_tensors="pt", padding=True
    ).to(device)

    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )

    # Build generation kwargs with resolved parameters
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": actual_max_new_tokens,
        "do_sample": actual_do_sample,
        "temperature": actual_temperature,
        "top_p": actual_top_p,
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
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")

        # Print to console as it streams
        print(new_text, end="", flush=True)

        time.sleep(0.01)

    # Print final newline after streaming is complete
    print()  # Add newline at the end

    # Return the complete text only at the end
    return buffer

full_page_ocr_vlm_prompt = """Extract ALL individual text lines from this document image. 

IMPORTANT: Extract each horizontal line of text separately. Do NOT combine multiple lines into paragraphs. Each line that appears on a separate horizontal row in the image should be a separate entry.

For each individual line of text, provide:
1. The text content (only the text on that single horizontal line)
2. The bounding box coordinates as [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right of that specific line
3. A confidence score between 0-100

Rules:
- Each line must be on a separate horizontal row in the image
- If text spans multiple horizontal lines, split it into separate entries (one per line)
- Do NOT combine lines that appear on different horizontal rows
- Each bounding box should tightly fit around a single horizontal line of text
- Empty lines should be skipped

Return the results as a JSON array of objects with this exact format:
[
  {
    "text": "first line text here",
    "bbox": [x1, y1, x2, y2],
    "confidence": 95
  },
  {
    "text": "second line text here",
    "bbox": [x1, y1, x2, y2],
    "confidence": 95
  },
  ...
]

Only return valid JSON, no additional text or explanation."""
