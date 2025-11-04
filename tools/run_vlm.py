import os
import sys
import time
from threading import Thread

import spaces
from PIL import Image

from tools.config import MAX_SPACES_GPU_RUN_TIME, SHOW_VLM_MODEL_OPTIONS

if SHOW_VLM_MODEL_OPTIONS is True:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        TextIteratorStreamer,
    )

    from tools.config import (
        MAX_NEW_TOKENS,
        MODEL_CACHE_PATH,
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

    if USE_FLASH_ATTENTION is True:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    print(f"Loading vision model: {SELECTED_MODEL}")

    # Load only the selected model based on configuration
    if SELECTED_MODEL == "Nanonets-OCR2-3B":
        MODEL_ID = "nanonets/Nanonets-OCR2-3B"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )

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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        model_default_prompt = """Extract the text content from this image."""
        model_default_max_new_tokens = MAX_NEW_TOKENS

    elif SELECTED_MODEL == "Qwen3-VL-2B-Instruct":
        MODEL_ID = "Qwen/3-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype="auto", device_map="auto", trust_remote_code=True
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_greedy = False  # 'false' string converted to boolean
        model_default_top_p = 0.8
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS

    elif SELECTED_MODEL == "Qwen3-VL-4B-Instruct":
        MODEL_ID = "Qwen/3-VL-4B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype="auto", device_map="auto", trust_remote_code=True
        ).eval()

        model_default_prompt = """Read all the text in the image."""
        model_default_greedy = False  # 'false' string converted to boolean
        model_default_top_p = 0.8
        model_default_top_k = 20
        model_default_temperature = 0.7
        model_default_repetition_penalty = 1.0
        model_default_presence_penalty = 1.5
        model_default_max_new_tokens = MAX_NEW_TOKENS

    elif SELECTED_MODEL == "PaddleOCR-VL":
        MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        model_default_prompt = """OCR:"""
        model_default_max_new_tokens = MAX_NEW_TOKENS

    else:
        raise ValueError(
            f"Invalid model selected: {SELECTED_MODEL}. Valid options are: Nanonets-OCR2-3B, Dots.OCR, Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, PaddleOCR-VL"
        )

    print(f"Successfully loaded {SELECTED_MODEL}")


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def generate_image(
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

    # Add presence_penalty if it's set (some models may support it)
    if actual_presence_penalty is not None:
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
