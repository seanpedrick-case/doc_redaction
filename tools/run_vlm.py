import os
import sys
import time
from threading import Thread

import spaces
from PIL import Image

from tools.config import SHOW_VLM_MODEL_OPTIONS, MAX_SPACES_GPU_RUN_TIME

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
        SELECTED_MODEL,
        USE_FLASH_ATTENTION,
        MODEL_CACHE_PATH,
    )

    # Configuration: Choose which vision model to load
    # Options: "olmOCR-2-7B-1025", "Nanonets-OCR2-3B", "Chandra-OCR", "Dots.OCR"
    # SELECTED_MODEL = os.getenv("VISION_MODEL", "Dots.OCR")

    # This code is uses significant amounts of code from the Hugging Face space here: https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR3 . Thanks!

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

    print(f"Loading vision model: {SELECTED_MODEL}")

    # Load only the selected model based on configuration
    if SELECTED_MODEL == "olmOCR-2-7B-1025":
        MODEL_ID = "allenai/olmOCR-2-7B-1025"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )

    elif SELECTED_MODEL == "Nanonets-OCR2-3B":
        MODEL_ID = "nanonets/Nanonets-OCR2-3B"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )

    elif SELECTED_MODEL == "Chandra-OCR":
        MODEL_ID = "datalab-to/chandra"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )

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

        if USE_FLASH_ATTENTION is True:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"

        MODEL_ID = model_path_d_local
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    else:
        raise ValueError(
            f"Invalid model selected: {SELECTED_MODEL}. Valid options are: olmOCR-2-7B-1025, Nanonets-OCR2-3B, Chandra-OCR, Dots.OCR"
        )

    print(f"Successfully loaded {SELECTED_MODEL}")


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def generate_image(
    text: str,
    image: Image.Image,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    """
    Generates responses using the configured vision model for image input.
    Streams text to console and returns complete text only at the end.
    """
    if image is None:
        return "Please upload an image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
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
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
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
