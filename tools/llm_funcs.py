import json
import os
import re
import time
from typing import List, Tuple

import boto3
import requests
import spaces

from tools.config import (
    MAX_SPACES_GPU_RUN_TIME,
    PRINT_TRANSFORMERS_USER_PROMPT,
    REPORT_LLM_OUTPUTS_TO_GUI,
)

# Import mock patches if in test mode
if os.environ.get("USE_MOCK_LLM") == "1" or os.environ.get("TEST_MODE") == "1":
    try:
        # Try to import and apply mock patches
        import sys

        # Add project root to sys.path so we can import test.mock_llm_calls
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # try:
        #     from test.mock_llm_calls import apply_mock_patches

        #     apply_mock_patches()
        # except ImportError:
        #     # If mock module not found, continue without mocking
        #     pass
    except Exception:
        # If anything fails, continue without mocking
        pass
try:
    from google import genai as ai
    from google.genai import types
except ImportError:
    print(
        "Warning: Google GenAI not found. Google GenAI functionality will not be available."
    )
    pass
from gradio import Progress
from huggingface_hub import hf_hub_download

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI not found. OpenAI functionality will not be available.")
    pass
from tqdm import tqdm

model_type = None  # global variable setup
full_text = (
    ""  # Define dummy source text (full text) just to enable highlight function to load
)

# Global variables for model and tokenizer
# Note: These are kept for backward compatibility but are no longer used.
# All model loading now uses the PII globals (_pii_model, _pii_tokenizer, _pii_assistant_model)
# via get_pii_model(), get_pii_tokenizer(), etc.
# _model = None
# _tokenizer = None
# _assistant_model = None

# Global variables for PII detection model and tokenizer
# These are now used for all LLM model loading (both general and PII-specific)
_pii_model = None
_pii_tokenizer = None
_pii_assistant_model = None

# Import config variables with defaults for missing ones
# This allows llm_funcs.py to work even if some config variables don't exist
from tools.config import (
    ASSISTANT_MODEL,
    BATCH_SIZE_DEFAULT,
    COMPILE_MODE,
    COMPILE_TRANSFORMERS,
    DEDUPLICATION_THRESHOLD,
    HF_TOKEN,
    INT8_WITH_OFFLOAD_TO_CPU,
    LLM_BATCH_SIZE,
    LLM_CONTEXT_LENGTH,
    LLM_LAST_N_TOKENS,
    LLM_MAX_GPU_LAYERS,
    LLM_MAX_NEW_TOKENS,
    LLM_MIN_P,
    LLM_REPETITION_PENALTY,
    LLM_RESET,
    LLM_SAMPLE,
    LLM_SEED,
    LLM_STOP_STRINGS,
    LLM_STREAM,
    LLM_TEMPERATURE,
    LLM_THREADS,
    LLM_TOP_K,
    LLM_TOP_P,
    LOAD_TRANSFORMERS_LLM_PII_MODEL_AT_START,
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE,
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE,
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER,
    LOCAL_TRANSFORMERS_LLM_PII_REPO_ID,
    MAX_COMMENT_CHARS,
    MAX_TIME_FOR_LOOP,
    MODEL_DTYPE,
    MULTIMODAL_PROMPT_FORMAT,
    NUM_PRED_TOKENS,
    NUMBER_OF_RETRY_ATTEMPTS,
    QUANTISE_TRANSFORMERS_LLM_MODELS,
    REASONING_SUFFIX,
    SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL,
    SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS,
    SPECULATIVE_DECODING,
    TIMEOUT_WAIT,
    USE_LLAMA_CPP,
    USE_LLAMA_SWAP,
    USE_TRANFORMERS_VLM_MODEL_AS_LLM,
)


def _report_llm_output_to_gui(text: str) -> None:
    """Report streamed LLM output to Gradio UI via gr.Info when REPORT_LLM_OUTPUTS_TO_GUI is True."""
    if not REPORT_LLM_OUTPUTS_TO_GUI or not (text and str(text).strip()):
        return
    try:
        import gradio as gr

        gr.Info(text, duration=2)
    except Exception:
        # gr.Info may not be available (e.g. in worker process or CLI), ignore
        pass


if isinstance(NUM_PRED_TOKENS, str):
    NUM_PRED_TOKENS = int(NUM_PRED_TOKENS)
if isinstance(LLM_MAX_GPU_LAYERS, str):
    LLM_MAX_GPU_LAYERS = int(LLM_MAX_GPU_LAYERS)
if isinstance(LLM_THREADS, str):
    LLM_THREADS = int(LLM_THREADS)

max_tokens = LLM_MAX_NEW_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS

temperature = LLM_TEMPERATURE
top_k = LLM_TOP_K
top_p = LLM_TOP_P
min_p = LLM_MIN_P
repetition_penalty = LLM_REPETITION_PENALTY
last_n_tokens = LLM_LAST_N_TOKENS
LLM_MAX_NEW_TOKENS: int = LLM_MAX_NEW_TOKENS
seed: int = LLM_SEED
reset: bool = LLM_RESET
stream: bool = LLM_STREAM
batch_size: int = LLM_BATCH_SIZE
context_length: int = LLM_CONTEXT_LENGTH
sample = LLM_SAMPLE
stop_strings = LLM_STOP_STRINGS
speculative_decoding = SPECULATIVE_DECODING
if LLM_MAX_GPU_LAYERS != 0:
    gpu_layers = int(LLM_MAX_GPU_LAYERS)
    torch_device = "cuda"
else:
    gpu_layers = 0
    torch_device = "cpu"

if not LLM_THREADS:
    threads = 1
else:
    threads = LLM_THREADS


class llama_cpp_init_config_gpu:
    def __init__(
        self,
        last_n_tokens=last_n_tokens,
        seed=seed,
        n_threads=threads,
        n_batch=batch_size,
        n_ctx=context_length,
        n_gpu_layers=gpu_layers,
        reset=reset,
    ):

        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.reset = reset
        # self.stop: list[str] = field(default_factory=lambda: [stop_string])

    def update_gpu(self, new_value):
        self.n_gpu_layers = new_value

    def update_context(self, new_value):
        self.n_ctx = new_value


class llama_cpp_init_config_cpu(llama_cpp_init_config_gpu):
    def __init__(self):
        super().__init__()
        self.n_gpu_layers = gpu_layers
        self.n_ctx = context_length


gpu_config = llama_cpp_init_config_gpu()
cpu_config = llama_cpp_init_config_cpu()


class LlamaCPPGenerationConfig:
    def __init__(
        self,
        temperature=temperature,
        top_k=top_k,
        min_p=min_p,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
        seed=seed,
        stream=stream,
        max_tokens=LLM_MAX_NEW_TOKENS,
        reset=reset,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.max_tokens = max_tokens
        self.stream = stream
        self.reset = reset

    def update_temp(self, new_value):
        self.temperature = new_value


# ResponseObject class for AWS Bedrock calls
class ResponseObject:
    def __init__(self, text, usage_metadata):
        self.text = text
        self.usage_metadata = usage_metadata


###
# LOCAL MODEL FUNCTIONS
###


def get_model_path(
    repo_id=LOCAL_TRANSFORMERS_LLM_PII_REPO_ID,
    model_filename=LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE,
    model_dir=LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER,
    hf_token=HF_TOKEN,
):
    # Construct the expected local path
    local_path = os.path.join(model_dir, model_filename)

    print("local path for model load:", local_path)

    try:
        if os.path.exists(local_path):
            print(f"Model already exists at: {local_path}")

            return local_path
        else:
            if hf_token:
                print("Downloading model from Hugging Face Hub with HF token")
                downloaded_model_path = hf_hub_download(
                    repo_id=repo_id, token=hf_token, filename=model_filename
                )

                return downloaded_model_path
            else:
                print(
                    "No HF token found, downloading model from Hugging Face Hub without token"
                )
                downloaded_model_path = hf_hub_download(
                    repo_id=repo_id, filename=model_filename
                )

                return downloaded_model_path

    except Exception as e:
        print("Error loading model:", e)
        raise Warning("Error loading model:", e)


def load_model(
    local_model_type: str = None,
    gpu_layers: int = gpu_layers,
    max_context_length: int = context_length,
    gpu_config: llama_cpp_init_config_gpu = gpu_config,
    cpu_config: llama_cpp_init_config_cpu = cpu_config,
    torch_device: str = torch_device,
    repo_id=LOCAL_TRANSFORMERS_LLM_PII_REPO_ID,
    model_filename=LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE,
    model_dir=LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER,
    compile_mode=COMPILE_MODE,
    model_dtype=MODEL_DTYPE,
    hf_token=HF_TOKEN,
    speculative_decoding=speculative_decoding,
    model=None,
    tokenizer=None,
    assistant_model=None,
):
    """
    Load in a model from Hugging Face hub via the transformers package, or using llama_cpp_python by downloading a GGUF file from Huggingface Hub.

    Args:
        local_model_type (str): The type of local model to load (e.g., "llama-cpp").
        gpu_layers (int): The number of GPU layers to offload to the GPU.
        max_context_length (int): The maximum context length for the model.
        gpu_config (llama_cpp_init_config_gpu): Configuration object for GPU-specific Llama.cpp parameters.
        cpu_config (llama_cpp_init_config_cpu): Configuration object for CPU-specific Llama.cpp parameters.
        torch_device (str): The device to load the model on ("cuda" for GPU, "cpu" for CPU).
        repo_id (str): The Hugging Face repository ID where the model is located.
        model_filename (str): The specific filename of the model to download from the repository.
        model_dir (str): The local directory where the model will be stored or downloaded.
        compile_mode (str): The compilation mode to use for the model.
        model_dtype (str): The data type to use for the model.
        hf_token (str): The Hugging Face token to use for the model.
        speculative_decoding (bool): Whether to use speculative decoding.
        model (Llama/transformers model): The model to load.
        tokenizer (list/transformers tokenizer): The tokenizer to load.
        assistant_model (transformers model): The assistant model for speculative decoding.
    Returns:
        tuple: A tuple containing:
            - model (Llama/transformers model): The loaded Llama.cpp/transformers model instance.
            - tokenizer (list/transformers tokenizer): An empty list (tokenizer is not used with Llama.cpp directly in this setup), or a transformers tokenizer.
            - assistant_model (transformers model): The assistant model for speculative decoding (if speculative_decoding is True).
    """

    # If model is provided, validate that tokenizer is also provided and compatible
    if model:
        if tokenizer is None:
            print(
                "Warning: Model provided but tokenizer is None. Attempting to load matching tokenizer..."
            )
            # Try to determine model_id from model config
            try:
                if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                    model_id = model.config._name_or_path
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
                    if not tokenizer.pad_token:
                        tokenizer.pad_token = tokenizer.eos_token
                    print(f"Loaded matching tokenizer from {model_id}")
                else:
                    print(
                        "Warning: Could not determine model source to load matching tokenizer"
                    )
            except Exception as e:
                print(f"Warning: Failed to load matching tokenizer: {e}")
        return model, tokenizer, assistant_model

    # Use LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE if local_model_type is not provided
    if local_model_type is None:
        local_model_type = LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE

    print("Loading model:", local_model_type)

    # Verify the device and cuda settings
    # Check if CUDA is enabled

    import torch

    torch.cuda.empty_cache()
    print("Is CUDA enabled? ", torch.cuda.is_available())
    print("Is a CUDA device available on this computer?", torch.backends.cudnn.enabled)
    if torch.cuda.is_available():
        torch_device = "cuda"
        gpu_layers = int(LLM_MAX_GPU_LAYERS)
        print("CUDA version:", torch.version.cuda)
        # try:
        #    os.system("nvidia-smi")
        # except Exception as e:
        #    print("Could not print nvidia-smi settings due to:", e)
    else:
        torch_device = "cpu"
        gpu_layers = 0

    print("Running on device:", torch_device)
    print("GPU layers assigned to cuda:", gpu_layers)

    if not LLM_THREADS:
        threads = torch.get_num_threads()
    else:
        threads = LLM_THREADS
    print("CPU threads:", threads)

    # GPU mode
    if torch_device == "cuda":
        torch.cuda.empty_cache()
        gpu_config.update_gpu(gpu_layers)
        gpu_config.update_context(max_context_length)

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print("Loading model from transformers")
        # Use the official model ID for Gemma 3 4B
        model_id = repo_id
        # 1. Set Data Type (dtype)
        # For H200/Hopper: 'bfloat16'
        # For RTX 3060/Ampere: 'float16'
        dtype_str = model_dtype  # os.environ.get("MODEL_DTYPE", "bfloat16").lower()
        if dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_str == "float16":
            torch_dtype = torch.float16
        elif dtype_str == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = torch.float32  # A safe fallback

        # 2. Set Compilation Mode
        # 'max-autotune' is great for both but can be slow initially.
        # 'reduce-overhead' is a faster alternative for compiling.

        print("--- System Configuration ---")
        print(f"Using model id: {model_id}")
        print(f"Using dtype: {torch_dtype}")
        print(f"Using compile mode: {compile_mode}")
        print(f"Using quantization: {QUANTISE_TRANSFORMERS_LLM_MODELS}")
        print("--------------------------\n")

        # --- Load Tokenizer and Model Atomically ---
        # Ensure both model and tokenizer are loaded from the same source
        # If either fails, both should fail together to prevent mismatched pairs

        try:
            # Setup quantization config if enabled
            quantization_config = None
            if QUANTISE_TRANSFORMERS_LLM_MODELS:
                if not torch.cuda.is_available():
                    print(
                        "Warning: Quantisation requires CUDA, but CUDA is not available."
                    )
                    print("Falling back to loading models without quantisation")
                    quantization_config = None
                else:
                    if INT8_WITH_OFFLOAD_TO_CPU:
                        # This will be very slow. Requires at least 4GB of VRAM and 32GB of RAM
                        print(
                            "Using bitsandbytes for quantisation to 8 bits, with offloading to CPU"
                        )
                        max_memory = {0: "4GB", "cpu": "32GB"}
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            max_memory=max_memory,
                            llm_int8_enable_fp32_cpu_offload=True,  # Note: if bitsandbytes has to offload to CPU, inference will be slow
                        )
                    else:
                        # For Gemma 4B, requires at least 6GB of VRAM
                        print("Using bitsandbytes for quantisation to 4 bits")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",  # Use the modern NF4 quantisation for better performance
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_use_double_quant=True,  # Optional: uses a second quantisation step to save even more memory
                        )

            # Prepare load kwargs
            # Match VLM behavior: always use device_map="auto" for better device handling
            load_kwargs = {
                # "max_seq_length": max_context_length,
                "token": hf_token,
                "device_map": "auto",  # Always use device_map="auto" like VLM
            }

            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                print("Loading model with bitsandbytes quantisation")
            else:
                # Use "auto" dtype like VLM for better compatibility
                load_kwargs["dtype"] = "auto" if model_dtype == "auto" else torch_dtype
                print("Loading model without quantisation")

            # Load tokenizer FIRST to validate the model_id is accessible
            # This ensures we catch tokenizer errors before loading the (larger) model
            print(f"Loading tokenizer from {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
            )

            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer loaded successfully")

            # Load model from the SAME model_id to ensure compatibility
            print(f"Loading model from {model_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
            )

            # Set model to evaluation mode (standard transformers approach)
            # Note: With device_map="auto", don't manually move model - let it handle device placement
            model.eval()
            print("Model loaded successfully")

            # Validate that model and tokenizer are from the same source
            if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                model_source = model.config._name_or_path
                if hasattr(tokenizer, "name_or_path"):
                    tokenizer_source = tokenizer.name_or_path
                    if model_source != tokenizer_source and model_id not in [
                        model_source,
                        tokenizer_source,
                    ]:
                        print(
                            f"Warning: Model source ({model_source}) and tokenizer source ({tokenizer_source}) may differ. Using model_id: {model_id}"
                        )

        except Exception as e:
            # If loading fails, ensure both model and tokenizer are None to prevent partial state
            print(f"Error loading model and tokenizer: {e}")
            model = None
            tokenizer = None
            raise RuntimeError(
                f"Failed to load model and tokenizer from {model_id}: {e}"
            ) from e

        # Compile the Model with the selected mode 🚀
        if COMPILE_TRANSFORMERS:
            try:
                model = torch.compile(model, mode=compile_mode, fullgraph=False)
            except Exception as e:
                print(f"Could not compile model: {e}. Running in eager mode.")

        print(
            "Loading with",
            gpu_config.n_gpu_layers,
            "model layers sent to GPU and a maximum context length of",
            gpu_config.n_ctx,
        )

    # CPU mode
    else:
        try:
            from transformers import AutoTokenizer

            model_id = (
                repo_id.split("https://huggingface.co/")[-1]
                if "https://huggingface.co/" in repo_id
                else repo_id
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
            )
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            print(
                f"Loaded tokenizer from {model_id} for compatibility (llama-cpp handles tokenization internally)"
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer for llama-cpp model: {e}")
            print(
                "Note: llama-cpp models handle tokenization internally, so tokenizer may not be needed"
            )
            tokenizer = None

        print(
            "Loading with",
            cpu_config.n_gpu_layers,
            "model layers sent to GPU and a maximum context length of",
            cpu_config.n_ctx,
        )

    print("Finished loading model:", local_model_type)
    print("GPU layers assigned to cuda:", gpu_layers)

    # Load assistant model for speculative decoding if enabled
    # Note: Assistant model typically shares the same tokenizer as the main model
    # for speculative decoding, so we don't load a separate tokenizer for it
    if speculative_decoding and torch_device == "cuda":
        print("Loading assistant model for speculative decoding:", ASSISTANT_MODEL)
        try:
            from transformers import (
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            # Setup quantization config for assistant model (same as main model)
            assistant_quantization_config = None
            if QUANTISE_TRANSFORMERS_LLM_MODELS and torch.cuda.is_available():
                if INT8_WITH_OFFLOAD_TO_CPU:
                    max_memory = {0: "4GB", "cpu": "32GB"}
                    assistant_quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        max_memory=max_memory,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                else:
                    assistant_quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                    )

            # Prepare load kwargs for assistant model
            assistant_load_kwargs = {
                "token": hf_token,
            }

            if assistant_quantization_config is not None:
                assistant_load_kwargs["quantization_config"] = (
                    assistant_quantization_config
                )
                assistant_load_kwargs["device_map"] = "auto"
                print("Loading assistant model with bitsandbytes quantisation")
            else:
                assistant_load_kwargs["dtype"] = torch_dtype
                print("Loading assistant model without quantisation")

            # Load the assistant model from ASSISTANT_MODEL
            # Note: Assistant model should be compatible with the main model's tokenizer
            # for speculative decoding to work correctly
            print(f"Loading assistant model from {ASSISTANT_MODEL}...")
            assistant_model = AutoModelForCausalLM.from_pretrained(
                ASSISTANT_MODEL, **assistant_load_kwargs
            )

            # For non-quantized assistant models, explicitly move to device (matching VLM behavior)
            if assistant_quantization_config is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                assistant_model = assistant_model.to(device)

            # Validate that assistant model can work with the main tokenizer
            # For speculative decoding, both models should use compatible tokenizers
            if hasattr(assistant_model, "config") and hasattr(
                assistant_model.config, "_name_or_path"
            ):
                assistant_source = assistant_model.config._name_or_path
                if hasattr(tokenizer, "name_or_path"):
                    tokenizer_source = tokenizer.name_or_path
                    if assistant_source != tokenizer_source:
                        print(
                            f"Warning: Assistant model ({assistant_source}) and tokenizer ({tokenizer_source}) are from different sources."
                        )
                        print(
                            "This may cause issues with speculative decoding. Ensure they are compatible."
                        )

            # Compile the assistant model if compilation is enabled
            if COMPILE_TRANSFORMERS:
                try:
                    assistant_model = torch.compile(
                        assistant_model, mode=compile_mode, fullgraph=False
                    )
                except Exception as e:
                    print(
                        f"Could not compile assistant model: {e}. Running in eager mode."
                    )

            print("Successfully loaded assistant model for speculative decoding")
            print("Note: Assistant model uses the same tokenizer as the main model")

        except Exception as e:
            print(f"Error loading assistant model: {e}")
            assistant_model = None
    else:
        assistant_model = None

    return model, tokenizer, assistant_model


# def # get_assistant_model():
#     """Get the globally loaded assistant model. Load it if not already loaded."""
#     global _pii_model, _pii_tokenizer, _pii_assistant_model
#     # Use PII globals to match get_pii_model() behavior
#     if _pii_assistant_model is None:
#         # Ensure model and tokenizer are loaded first
#         get_pii_model()
#         get_pii_tokenizer()
#     return _pii_assistant_model


# def set_model(model, tokenizer, assistant_model=None):
#     """Set the global model, tokenizer, and assistant model.

#     Note: This function now sets the PII globals to maintain consistency.
#     """
#     global _pii_model, _pii_tokenizer, _pii_assistant_model
#     _pii_model = model
#     _pii_tokenizer = tokenizer
#     _pii_assistant_model = assistant_model


# def get_pii_model():
#     """Get the globally loaded PII detection model. Load it if not already loaded."""
#     global _pii_model, _pii_tokenizer, _pii_assistant_model

#     # Check if model is already loaded
#     if _pii_model is not None:
#         print("PII model already loaded, reusing existing model instance.")
#         return _pii_model

#     # Determine which repo_id, model_file, and model_folder to use
#     # If PII-specific config is set, use it; otherwise fall back to general local model config
#     if LOCAL_TRANSFORMERS_LLM_PII_REPO_ID:
#         pii_repo_id = LOCAL_TRANSFORMERS_LLM_PII_REPO_ID
#         pii_model_file = LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE
#         pii_model_folder = LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER
#     else:
#         raise ValueError(
#             "LOCAL_TRANSFORMERS_LLM_PII_REPO_ID is not set. "
#             "Please configure the PII model repository ID."
#         )

#     print("Loading PII model for the first time...")
#     _pii_model, _pii_tokenizer, _pii_assistant_model = load_model(
#         local_model_type=LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE,
#         gpu_layers=gpu_layers,
#         max_context_length=context_length,
#         gpu_config=gpu_config,
#         cpu_config=cpu_config,
#         torch_device=torch_device,
#         repo_id=pii_repo_id,
#         model_filename=pii_model_file,
#         model_dir=pii_model_folder,
#         compile_mode=COMPILE_MODE,
#         model_dtype=MODEL_DTYPE,
#         hf_token=HF_TOKEN,
#         model=_pii_model,
#         tokenizer=_pii_tokenizer,
#         assistant_model=_pii_assistant_model,
#     )
#     print("PII model loaded successfully.")
#     return _pii_model


# def get_pii_tokenizer():
#     """Get the globally loaded PII detection tokenizer. Load it if not already loaded."""
#     global _pii_model, _pii_tokenizer, _pii_assistant_model

#     # Check if tokenizer is already loaded
#     if _pii_tokenizer is not None:
#         print("PII tokenizer already loaded, reusing existing tokenizer instance.")
#         return _pii_tokenizer

#     # Determine which repo_id, model_file, and model_folder to use
#     # If PII-specific config is set, use it; otherwise fall back to general local model config
#     if LOCAL_TRANSFORMERS_LLM_PII_REPO_ID:
#         pii_repo_id = LOCAL_TRANSFORMERS_LLM_PII_REPO_ID
#         pii_model_file = LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE
#         pii_model_folder = LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER
#     else:
#         raise ValueError(
#             "LOCAL_TRANSFORMERS_LLM_PII_REPO_ID is not set. "
#             "Please configure the PII model repository ID."
#         )

#     print("Loading PII tokenizer for the first time...")
#     _pii_model, _pii_tokenizer, _pii_assistant_model = load_model(
#         local_model_type=LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE,
#         gpu_layers=gpu_layers,
#         max_context_length=context_length,
#         gpu_config=gpu_config,
#         cpu_config=cpu_config,
#         torch_device=torch_device,
#         repo_id=pii_repo_id,
#         model_filename=pii_model_file,
#         model_dir=pii_model_folder,
#         compile_mode=COMPILE_MODE,
#         model_dtype=MODEL_DTYPE,
#         hf_token=HF_TOKEN,
#         model=_pii_model,
#         tokenizer=_pii_tokenizer,
#         assistant_model=_pii_assistant_model,
#     )
#     print("PII tokenizer loaded successfully.")
#     return _pii_tokenizer, _pii_tokenizer, _pii_assistant_model

# def get_model_and_tokenizer():
#     """Get both the globally loaded model and tokenizer together.

#     This is the recommended way to get both when you need them, as it ensures
#     they're loaded atomically from the same source and prevents mismatches.

#     Returns:
#         tuple: (model, tokenizer) - Both loaded and guaranteed to be from the same source
#     """
#     # Use PII versions which have better error handling
#     model = get_pii_model()
#     tokenizer = get_pii_tokenizer()
#     return model, tokenizer


# Initialize PII model at startup if configured (even if SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS is False)
# This allows PII model to be loaded independently for PII detection tasks
if (
    LOAD_TRANSFORMERS_LLM_PII_MODEL_AT_START
    and SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS
):
    try:
        print("Loading local PII model:", LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE)
        _pii_model, _pii_tokenizer, _pii_assistant_model = load_model(
            local_model_type=LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE,
            gpu_layers=gpu_layers,
            max_context_length=context_length,
            gpu_config=gpu_config,
            cpu_config=cpu_config,
            torch_device=torch_device,
            repo_id=LOCAL_TRANSFORMERS_LLM_PII_REPO_ID,
            model_filename=LOCAL_TRANSFORMERS_LLM_PII_MODEL_FILE,
            model_dir=LOCAL_TRANSFORMERS_LLM_PII_MODEL_FOLDER,
            compile_mode=COMPILE_MODE,
            model_dtype=MODEL_DTYPE,
            hf_token=HF_TOKEN,
            model=_pii_model,
            tokenizer=_pii_tokenizer,
            assistant_model=_pii_assistant_model,
        )
    except Exception as e:
        print(f"Warning: Could not load PII model at startup: {e}")
        print("PII model will be loaded on-demand when needed.")


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def call_transformers_model(
    prompt: str,
    system_prompt: str,
    gen_config: LlamaCPPGenerationConfig,
    model=_pii_model,
    tokenizer=_pii_tokenizer,
    assistant_model=_pii_assistant_model,
    speculative_decoding=speculative_decoding,
):
    """
    This function sends a request to a transformers model with the given prompt, system prompt, and generation configuration.
    """
    import torch
    from transformers import TextStreamer

    # Custom streamer that reports streamed output to gr.Info when REPORT_LLM_OUTPUTS_TO_GUI is True
    class _LLMGUIStreamer(TextStreamer):
        def __init__(self, tokenizer, skip_prompt=True):
            super().__init__(tokenizer, skip_prompt=skip_prompt)
            self._line_buffer = ""

        def on_finalized_text(self, text, stream_end=False):
            super().on_finalized_text(text, stream_end)
            if not REPORT_LLM_OUTPUTS_TO_GUI:
                return
            self._line_buffer += text
            if "\n" in text or stream_end:
                parts = self._line_buffer.split("\n")
                for line in parts[:-1]:
                    if line.strip():
                        _report_llm_output_to_gui(line)
                self._line_buffer = parts[-1] if parts else ""
                if stream_end and self._line_buffer.strip():
                    _report_llm_output_to_gui(self._line_buffer)

    # Load model and tokenizer together to ensure they're from the same source
    # This prevents mismatches that could occur if they're loaded separately
    if model is None or tokenizer is None:
        # Use get_model_and_tokenizer() to ensure both are loaded atomically
        # This is safer than calling get_pii_model() and get_pii_tokenizer() separately
        loaded_model, loaded_tokenizer, assistant_model = load_model()
        if model is None:
            model = loaded_model
        if tokenizer is None:
            tokenizer = loaded_tokenizer
    # if assistant_model is None and speculative_decoding:
    #     assistant_model = # get_assistant_model()

    if model is None or tokenizer is None:
        raise ValueError(
            "No model or tokenizer available. Either pass them as parameters or ensure LOAD_TRANSFORMERS_LLM_PII_MODEL_AT_START is True."
        )

    # Apply reasoning suffix to prompt if configured
    if REASONING_SUFFIX and REASONING_SUFFIX.strip():
        prompt = f"{prompt} {REASONING_SUFFIX}".strip()

    # 1. Define the conversation as a list of dictionaries
    # Note: The multimodal format [{"type": "text", "text": text}] is only needed for actual multimodal models
    # with images/videos. For text-only content, even multimodal models expect plain strings.

    # Check if system_prompt is meaningful (not empty/None)
    has_system_prompt = system_prompt and str(system_prompt).strip()

    # Always use string format for text-only content, regardless of MULTIMODAL_PROMPT_FORMAT setting
    # MULTIMODAL_PROMPT_FORMAT should only be used when you actually have multimodal inputs (images, etc.)
    if MULTIMODAL_PROMPT_FORMAT:
        conversation = []
        if has_system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": str(system_prompt)}],
                }
            )
        conversation.append(
            {"role": "user", "content": [{"type": "text", "text": str(prompt)}]}
        )
    else:
        conversation = []
        if has_system_prompt:
            conversation.append({"role": "system", "content": str(system_prompt)})
        conversation.append({"role": "user", "content": str(prompt)})

    if PRINT_TRANSFORMERS_USER_PROMPT:
        print("System prompt:", system_prompt)
        print("User prompt:", prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if assistant_model is not None:
        assistant_model = assistant_model.to(device)

    if PRINT_TRANSFORMERS_USER_PROMPT:
        print("Model device:", device)
        print("Model device type:", type(device))

    try:
        # Try applying chat template with system prompt (if present)
        # Create inputs dict like VLM does - this allows model to handle device placement automatically
        input_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(
            device
        )  # Ensure inputs match model device

        if PRINT_TRANSFORMERS_USER_PROMPT:
            print("Input IDs:", input_ids)
            print("Rendered prompt:")
            rendered = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            print(rendered)
            print("-" * 50)

    except (TypeError, KeyError, IndexError, ValueError) as e:
        # If chat template fails, try without system prompt (some models don't support it)
        if has_system_prompt:
            print(
                f"Chat template failed with system prompt ({e}), trying without system prompt..."
            )
            # Try again with only user prompt
            user_only_conversation = [{"role": "user", "content": str(prompt)}]
            try:
                input_ids = tokenizer.apply_chat_template(
                    user_only_conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                ).to(device)

                if PRINT_TRANSFORMERS_USER_PROMPT:
                    print("Input IDs:", input_ids)
                    print("Rendered prompt (without system):")
                    rendered = tokenizer.apply_chat_template(
                        user_only_conversation,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    print(rendered)
                    print("-" * 50)
            except Exception as e2:
                print(
                    f"Chat template failed without system prompt ({e2}), using manual tokenization"
                )
                # Combine system and user prompts manually as fallback
                full_prompt = (
                    f"{system_prompt}\n\n{prompt}" if has_system_prompt else prompt
                )
                # Tokenize manually with special tokens
                input_ids = tokenizer(
                    full_prompt, return_tensors="pt", add_special_tokens=True
                ).to(device)

        else:
            # No system prompt, but chat template still failed - use manual tokenization
            print(f"Chat template failed ({e}), using manual tokenization")
            full_prompt = str(prompt)
            input_ids = tokenizer(
                full_prompt, return_tensors="pt", add_special_tokens=True
            ).to(device)

    except Exception as e:
        print("Error applying chat template:", e)
        import traceback

        traceback.print_exc()
        raise

    attention_mask = torch.ones_like(input_ids).to(device)

    # Map LlamaCPP parameters to transformers parameters
    generation_kwargs = {
        "max_new_tokens": gen_config.max_tokens,
        "temperature": gen_config.temperature,
        "top_p": gen_config.top_p,
        "top_k": gen_config.top_k,
        "do_sample": True,
        "attention_mask": attention_mask,
        #'pad_token_id': tokenizer.eos_token_id
    }

    if gen_config.stream:
        streamer = (
            _LLMGUIStreamer(tokenizer, skip_prompt=True)
            if REPORT_LLM_OUTPUTS_TO_GUI
            else TextStreamer(tokenizer, skip_prompt=True)
        )
    else:
        streamer = None

    # Remove parameters that don't exist in transformers
    if hasattr(gen_config, "repeat_penalty"):
        generation_kwargs["repetition_penalty"] = gen_config.repeat_penalty

    if PRINT_TRANSFORMERS_USER_PROMPT:
        print("Generation kwargs:", generation_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Timed Inference Test ---
    print("\nStarting model inference...")
    start_time = time.time()

    # Use speculative decoding if assistant model is available
    try:
        if speculative_decoding and assistant_model is not None:
            if PRINT_TRANSFORMERS_USER_PROMPT:
                print("Using speculative decoding with assistant model")
            outputs = model.generate(
                input_ids,
                assistant_model=assistant_model,
                **generation_kwargs,
                streamer=streamer,
            )
        else:
            if PRINT_TRANSFORMERS_USER_PROMPT:
                print("Generating without speculative decoding")
            outputs = model.generate(input_ids, **generation_kwargs, streamer=streamer)
    except Exception as e:
        error_msg = str(e)
        # Check if this is a CUDA compilation error
        if (
            "sm_120" in error_msg
            or "LLVM ERROR" in error_msg
            or "Cannot select" in error_msg
        ):
            print("\n" + "=" * 80)
            print("CUDA COMPILATION ERROR DETECTED")
            print("=" * 80)
            print(
                "\nThe error is caused by torch.compile() trying to compile CUDA kernels"
            )
            print(
                "with incompatible settings. This is a known issue with certain CUDA/PyTorch"
            )
            print("combinations.\n")
            print(
                "SOLUTION: Disable model compilation by setting COMPILE_TRANSFORMERS=False"
            )
            print("in your config file (config/app_config.env).")
            print(
                "\nThe model will still work without compilation, just slightly slower."
            )
            print("=" * 80 + "\n")
            raise RuntimeError(
                "CUDA compilation error detected. Please set COMPILE_TRANSFORMERS=False "
                "in your config file to disable model compilation and avoid this error."
            ) from e
        else:
            # Re-raise other errors as-is
            raise

    end_time = time.time()

    # --- Decode and Display Results ---
    # Extract only the newly generated tokens (exclude input tokens)
    input_length = input_ids.shape[-1]

    # Handle different output formats from model.generate()
    # model.generate() returns a tensor with shape [batch_size, sequence_length]
    # that includes both input and generated tokens
    if isinstance(outputs, torch.Tensor):
        # If outputs is a tensor, extract the new tokens
        if outputs.dim() == 2:
            # Shape: [batch_size, sequence_length]
            new_tokens = outputs[0, input_length:].clone()
        elif outputs.dim() == 1:
            # Shape: [sequence_length] (single sequence)
            new_tokens = outputs[input_length:].clone()
        else:
            raise ValueError(f"Unexpected output tensor shape: {outputs.shape}")
    else:
        # If outputs is a sequence or other format
        if hasattr(outputs, "__getitem__"):
            new_tokens = (
                outputs[0][input_length:]
                if len(outputs) > 0
                else outputs[input_length:]
            )
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")

    # Ensure new_tokens is a tensor and on CPU for decoding
    if isinstance(new_tokens, torch.Tensor):
        new_tokens = new_tokens.cpu().clone()
        # Convert to list for decoding (some tokenizers prefer lists)
        new_tokens_list = new_tokens.tolist()
    else:
        new_tokens_list = (
            list(new_tokens) if hasattr(new_tokens, "__iter__") else [new_tokens]
        )

    if PRINT_TRANSFORMERS_USER_PROMPT:
        print(f"Input length: {input_length}")
        print(f"Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
        print(f"New tokens count: {len(new_tokens_list)}")
        print(f"First 20 new token IDs: {new_tokens_list[:20]}")

    # Decode the tokens
    # Use the token list for decoding (more reliable than tensor)
    try:
        assistant_reply = tokenizer.decode(
            new_tokens_list, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    except Exception as e:
        print(f"Warning: Error decoding tokens: {e}")
        print(f"New tokens count: {len(new_tokens_list)}")
        print(f"New tokens (first 20): {new_tokens_list[:20]}")
        # Try alternative decoding methods
        try:
            # Try with tensor directly
            if isinstance(new_tokens, torch.Tensor):
                assistant_reply = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            else:
                raise e
        except Exception as e2:
            print(f"Error with tensor decoding: {e2}")
            # Last resort: try to decode each token individually to see which ones fail
            try:
                decoded_parts = []
                failed_tokens = []
                for i, token_id in enumerate(
                    new_tokens_list[:200]
                ):  # Limit to first 200 to avoid issues
                    try:
                        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
                        decoded_parts.append(decoded)
                    except Exception as token_error:
                        failed_tokens.append((i, token_id, str(token_error)))
                        decoded_parts.append(f"<TOKEN_ERROR_{token_id}>")
                if failed_tokens:
                    print(
                        f"Warning: {len(failed_tokens)} tokens failed to decode individually"
                    )
                    print(f"First few failed tokens: {failed_tokens[:5]}")
                assistant_reply = "".join(decoded_parts)
            except Exception as e3:
                print(f"Error with individual token decoding: {e3}")
                assistant_reply = f"<DECODING_ERROR: {str(e3)}>"

    num_input_tokens = input_length
    num_generated_tokens = (
        len(new_tokens_list) if hasattr(new_tokens_list, "__len__") else 0
    )
    duration = end_time - start_time
    tokens_per_second = num_generated_tokens / duration if duration > 0 else 0

    if PRINT_TRANSFORMERS_USER_PROMPT:
        print(f"\nDecoded output length: {len(assistant_reply)} characters")
        print(f"First 200 chars of output: {assistant_reply[:200]}")

    print("\n--- Performance ---")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Generated tokens: {num_generated_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return assistant_reply, num_input_tokens, num_generated_tokens


# Function to send a request and update history
def send_request(
    prompt: str,
    conversation_history: List[dict],
    client: ai.Client | OpenAI,
    config: types.GenerateContentConfig,
    model_choice: str,
    system_prompt: str,
    temperature: float,
    bedrock_runtime: boto3.Session.client,
    model_source: str,
    local_model=_pii_model,
    tokenizer=_pii_tokenizer,
    assistant_model=_pii_assistant_model,
    assistant_prefill="",
    progress=Progress(track_tqdm=True),
    api_url: str = None,
) -> Tuple[str, List[dict]]:
    """Sends a request to a language model and manages the conversation history.

    This function constructs the full prompt by appending the new user prompt to the conversation history,
    generates a response from the model, and updates the conversation history with the new prompt and response.
    It handles different model sources (Gemini, AWS, Local, inference-server) and includes retry logic for API calls.

    Args:
        prompt (str): The user's input prompt to be sent to the model.
        conversation_history (List[dict]): A list of dictionaries representing the ongoing conversation.
                                           Each dictionary should have 'role' and 'parts' keys.
        client (ai.Client): The API client object for the chosen model (e.g., Gemini `ai.Client`, or Azure/OpenAI `OpenAI`).
        config (types.GenerateContentConfig): Configuration settings for content generation (e.g., Gemini `types.GenerateContentConfig`).
        model_choice (str): The specific model identifier to use (e.g., "gemini-pro", "claude-v2").
        system_prompt (str): An optional system-level instruction or context for the model.
        temperature (float): Controls the randomness of the model's output, with higher values leading to more diverse responses.
        bedrock_runtime (boto3.Session.client): The boto3 Bedrock runtime client object for AWS models.
        model_source (str): Indicates the source/provider of the model (e.g., "Gemini", "AWS", "Local", "inference-server").
        local_model (list, optional): A list containing the local model and its tokenizer (if `model_source` is "Local"). Defaults to [].
        tokenizer (object, optional): The tokenizer object for local models. Defaults to None.
        assistant_model (object, optional): An optional assistant model used for speculative decoding with local models. Defaults to None.
        assistant_prefill (str, optional): A string to pre-fill the assistant's response, useful for certain models like Claude. Defaults to "".
        progress (Progress, optional): A progress object for tracking the operation, typically from `tqdm`. Defaults to Progress(track_tqdm=True).
        api_url (str, optional): The API URL for inference-server calls. Required when model_source is 'inference-server'.

    Returns:
        Tuple[str, List[dict]]: A tuple containing the model's response text and the updated conversation history.
    """
    # Constructing the full prompt from the conversation history
    full_prompt = "Conversation history:\n"
    num_transformer_input_tokens = 0
    num_transformer_generated_tokens = 0
    response_text = ""

    for entry in conversation_history:
        role = entry[
            "role"
        ].capitalize()  # Assuming the history is stored with 'role' and 'parts'
        message = " ".join(entry["parts"])  # Combining all parts of the message
        full_prompt += f"{role}: {message}\n"

    # Adding the new user prompt
    full_prompt += f"\nUser: {prompt}"

    # Clear any existing progress bars
    tqdm._instances.clear()

    progress_bar = range(0, number_of_api_retry_attempts)

    # Generate the model's response
    if "Gemini" in model_source:

        for i in progress_bar:
            try:
                print("Calling Gemini model, attempt", i + 1)

                response = client.models.generate_content(
                    model=model_choice, contents=full_prompt, config=config
                )

                # print("Successful call to Gemini model.")
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print(
                    "Call to Gemini model failed:",
                    e,
                    " Waiting for ",
                    str(timeout_wait),
                    "seconds and trying again.",
                )

                time.sleep(timeout_wait)

            if i == number_of_api_retry_attempts:
                return (
                    ResponseObject(text="", usage_metadata={"RequestId": "FAILED"}),
                    conversation_history,
                    response_text,
                    num_transformer_input_tokens,
                    num_transformer_generated_tokens,
                )

    elif "AWS" in model_source:
        for i in progress_bar:
            try:
                print("Calling AWS Bedrock model, attempt", i + 1)
                response = call_aws_bedrock(
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    model_choice,
                    bedrock_runtime=bedrock_runtime,
                    assistant_prefill=assistant_prefill,
                )

                # print("Successful call to Claude model.")
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print(
                    "Call to Bedrock model failed:",
                    e,
                    " Waiting for ",
                    str(timeout_wait),
                    "seconds and trying again.",
                )
                time.sleep(timeout_wait)

            if i == number_of_api_retry_attempts:
                return (
                    ResponseObject(text="", usage_metadata={"RequestId": "FAILED"}),
                    conversation_history,
                    response_text,
                    num_transformer_input_tokens,
                    num_transformer_generated_tokens,
                )
    elif "Azure/OpenAI" in model_source:
        for i in progress_bar:
            try:
                print("Calling Azure/OpenAI inference model, attempt", i + 1)

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]

                response_raw = client.chat.completions.create(
                    messages=messages,
                    model=model_choice,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )

                response_text = response_raw.choices[0].message.content
                usage = getattr(response_raw, "usage", None)
                input_tokens = 0
                output_tokens = 0
                if usage is not None:
                    input_tokens = getattr(
                        usage, "input_tokens", getattr(usage, "prompt_tokens", 0)
                    )
                    output_tokens = getattr(
                        usage, "output_tokens", getattr(usage, "completion_tokens", 0)
                    )
                response = ResponseObject(
                    text=response_text,
                    usage_metadata={
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                    },
                )
                break
            except Exception as e:
                print(
                    "Call to Azure/OpenAI model failed:",
                    e,
                    " Waiting for ",
                    str(timeout_wait),
                    "seconds and trying again.",
                )
                time.sleep(timeout_wait)
            if i == number_of_api_retry_attempts:
                return (
                    ResponseObject(text="", usage_metadata={"RequestId": "FAILED"}),
                    conversation_history,
                    response_text,
                    num_transformer_input_tokens,
                    num_transformer_generated_tokens,
                )
    elif "Local" in model_source:
        # This is the local model. When USE_TRANFORMERS_VLM_MODEL_AS_LLM and model_choice is the VLM model, use the loaded VLM model/tokenizer.
        vlm_model, vlm_tokenizer = None, None
        if (
            USE_TRANFORMERS_VLM_MODEL_AS_LLM
            and model_choice == SELECTED_LOCAL_TRANSFORMERS_VLM_MODEL
        ):
            try:
                from tools.run_vlm import get_loaded_vlm_model_and_tokenizer

                vlm_model, vlm_tokenizer = get_loaded_vlm_model_and_tokenizer()
            except Exception as e:
                print(
                    f"Could not get VLM model for LLM task (USE_TRANFORMERS_VLM_MODEL_AS_LLM): {e}"
                )

        for i in progress_bar:
            try:
                print("Calling local model, attempt", i + 1)

                gen_config = LlamaCPPGenerationConfig()
                gen_config.update_temp(temperature)

                # Call transformers model; use VLM model/tokenizer when USE_TRANFORMERS_VLM_MODEL_AS_LLM and available
                if vlm_model is not None and vlm_tokenizer is not None:
                    (
                        response,
                        num_transformer_input_tokens,
                        num_transformer_generated_tokens,
                    ) = call_transformers_model(
                        prompt,
                        system_prompt,
                        gen_config,
                        model=vlm_model,
                        tokenizer=vlm_tokenizer,
                    )
                else:
                    (
                        response,
                        num_transformer_input_tokens,
                        num_transformer_generated_tokens,
                    ) = call_transformers_model(
                        prompt,
                        system_prompt,
                        gen_config,
                    )
                response_text = response

                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print(
                    "Call to local model failed:",
                    e,
                    " Waiting for ",
                    str(timeout_wait),
                    "seconds and trying again.",
                )

                time.sleep(timeout_wait)

            if i == number_of_api_retry_attempts:
                return (
                    ResponseObject(text="", usage_metadata={"RequestId": "FAILED"}),
                    conversation_history,
                    response_text,
                    num_transformer_input_tokens,
                    num_transformer_generated_tokens,
                )
    elif "inference-server" in model_source:
        # This is the inference-server API
        for i in progress_bar:
            try:
                print("Calling inference-server API, attempt", i + 1)

                if api_url is None:
                    raise ValueError(
                        "api_url is required when model_source is 'inference-server'"
                    )

                gen_config = LlamaCPPGenerationConfig()
                gen_config.update_temp(temperature)

                response = call_inference_server_api(
                    prompt,
                    system_prompt,
                    gen_config,
                    api_url=api_url,
                    model_name=model_choice,
                    use_llama_swap=USE_LLAMA_SWAP,
                )

                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print(
                    "Call to inference-server API failed:",
                    e,
                    " Waiting for ",
                    str(timeout_wait),
                    "seconds and trying again.",
                )

                time.sleep(timeout_wait)

            if i == number_of_api_retry_attempts:
                return (
                    ResponseObject(text="", usage_metadata={"RequestId": "FAILED"}),
                    conversation_history,
                    response_text,
                    num_transformer_input_tokens,
                    num_transformer_generated_tokens,
                )
    else:
        print("Model source not recognised")
        return (
            ResponseObject(text="", usage_metadata={"RequestId": "FAILED"}),
            conversation_history,
            response_text,
            num_transformer_input_tokens,
            num_transformer_generated_tokens,
        )

    # Update the conversation history with the new prompt and response
    conversation_history.append({"role": "user", "parts": [prompt]})

    # Check if is a LLama.cpp model response or inference-server response
    if isinstance(response, ResponseObject):
        response_text = response.text
    elif "choices" in response:  # LLama.cpp model response or inference-server response
        # Check for GPT-OSS thinking models (case-insensitive, handle both hyphen and underscore)
        if "gpt-oss" in model_choice.lower() or "gpt_oss" in model_choice.lower():
            content = response["choices"][0]["message"]["content"]
            # Split on the final channel marker to extract only the final output (not thinking tokens)
            parts = content.split("<|start|>assistant<|channel|>final<|message|>")
            if len(parts) > 1:
                response_text = parts[1]
            # Following format may be from llama.cpp inference-server response
            elif len(parts) == 1:
                parts = content.split("<|end|>")
                if len(parts) > 1:
                    response_text = parts[1]
                else:
                    print(
                        "Warning: Could not find final channel marker in GPT-OSS response. Using full content."
                    )
                    response_text = content
            else:
                # Fallback: if marker not found, use the full content (may include thinking tokens)
                print(
                    "Warning: Could not find final channel marker in GPT-OSS response. Using full content."
                )
                response_text = content
        else:
            response_text = response["choices"][0]["message"]["content"]
    elif model_source == "Gemini":
        response_text = response.text
    else:  # Assume transformers model response
        # Check for GPT-OSS thinking models (case-insensitive, handle both hyphen and underscore)
        if "gpt-oss" in model_choice.lower() or "gpt_oss" in model_choice.lower():
            # Split on the final channel marker to extract only the final output (not thinking tokens)
            parts = response.split("<|start|>assistant<|channel|>final<|message|>")
            if len(parts) > 1:
                response_text = parts[1]
            else:
                # Fallback: if marker not found, use the full content (may include thinking tokens)
                print(
                    "Warning: Could not find final channel marker in GPT-OSS response. Using full content."
                )
                response_text = response
        else:
            response_text = response

    # Strip <|end|> tags (used by GPT-OSS thinking models to mark end of thinking)
    response_text = re.sub(r"<\|end\|>", "", response_text)

    # Replace multiple spaces with single space
    response_text = re.sub(r" {2,}", " ", response_text)
    response_text = response_text.strip()

    conversation_history.append({"role": "assistant", "parts": [response_text]})

    return (
        response,
        conversation_history,
        response_text,
        num_transformer_input_tokens,
        num_transformer_generated_tokens,
    )


def process_requests(
    prompts: List[str],
    system_prompt: str,
    conversation_history: List[dict],
    whole_conversation: List[str],
    whole_conversation_metadata: List[str],
    client: ai.Client | OpenAI,
    config: types.GenerateContentConfig,
    model_choice: str,
    temperature: float,
    bedrock_runtime: boto3.Session.client,
    model_source: str,
    batch_no: int = 1,
    local_model=_pii_model,
    tokenizer=_pii_tokenizer,
    assistant_model=_pii_assistant_model,
    master: bool = False,
    assistant_prefill="",
    api_url: str = None,
) -> Tuple[List[ResponseObject], List[dict], List[str], List[str]]:
    """
    Processes a list of prompts by sending them to the model, appending the responses to the conversation history, and updating the whole conversation and metadata.

    Args:
        prompts (List[str]): A list of prompts to be processed.
        system_prompt (str): The system prompt.
        conversation_history (List[dict]): The history of the conversation.
        whole_conversation (List[str]): The complete conversation including prompts and responses.
        whole_conversation_metadata (List[str]): Metadata about the whole conversation.
        client (object): The client to use for processing the prompts, from either Gemini or OpenAI client.
        config (dict): Configuration for the model.
        model_choice (str): The choice of model to use.
        temperature (float): The temperature parameter for the model.
        model_source (str): Source of the model, whether local, AWS, Gemini, or inference-server
        batch_no (int): Batch number of the large language model request.
        local_model: Local gguf model (if loaded)
        master (bool): Is this request for the master table.
        assistant_prefill (str, optional): Is there a prefill for the assistant response. Currently only working for AWS model calls
        bedrock_runtime: The client object for boto3 Bedrock runtime
        api_url (str, optional): The API URL for inference-server calls. Required when model_source is 'inference-server'.

    Returns:
        Tuple[List[ResponseObject], List[dict], List[str], List[str]]: A tuple containing the list of responses, the updated conversation history, the updated whole conversation, and the updated whole conversation metadata.
    """
    responses = list()

    # Clear any existing progress bars
    tqdm._instances.clear()

    for prompt in prompts:

        (
            response,
            conversation_history,
            response_text,
            num_transformer_input_tokens,
            num_transformer_generated_tokens,
        ) = send_request(
            prompt,
            conversation_history,
            client=client,
            config=config,
            model_choice=model_choice,
            system_prompt=system_prompt,
            temperature=temperature,
            local_model=local_model,
            tokenizer=tokenizer,
            assistant_model=assistant_model,
            assistant_prefill=assistant_prefill,
            bedrock_runtime=bedrock_runtime,
            model_source=model_source,
            api_url=api_url,
        )

        responses.append(response)
        whole_conversation.append(system_prompt)
        whole_conversation.append(prompt)
        whole_conversation.append(response_text)

        whole_conversation_metadata.append(f"Batch {batch_no}:")

        try:
            if "AWS" in model_source:
                output_tokens = response.usage_metadata.get("outputTokens", 0)
                input_tokens = response.usage_metadata.get("inputTokens", 0)

            elif "Gemini" in model_source:
                output_tokens = response.usage_metadata.candidates_token_count
                input_tokens = response.usage_metadata.prompt_token_count

            elif "Azure/OpenAI" in model_source:
                input_tokens = response.usage_metadata.get("inputTokens", 0)
                output_tokens = response.usage_metadata.get("outputTokens", 0)

            elif "Local" in model_source:
                if USE_LLAMA_CPP == "True":
                    output_tokens = response["usage"].get("completion_tokens", 0)
                    input_tokens = response["usage"].get("prompt_tokens", 0)

                if USE_LLAMA_CPP == "False":
                    input_tokens = num_transformer_input_tokens
                    output_tokens = num_transformer_generated_tokens

            elif "inference-server" in model_source:
                # inference-server returns the same format as llama-cpp
                output_tokens = response["usage"].get("completion_tokens", 0)
                input_tokens = response["usage"].get("prompt_tokens", 0)

            else:
                input_tokens = 0
                output_tokens = 0

            whole_conversation_metadata.append(
                "input_tokens: "
                + str(input_tokens)
                + " output_tokens: "
                + str(output_tokens)
            )

        except KeyError as e:
            print(f"Key error: {e} - Check the structure of response.usage_metadata")

    return (
        responses,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
        response_text,
    )


def call_inference_server_api(
    formatted_string: str,
    system_prompt: str,
    gen_config: LlamaCPPGenerationConfig,
    api_url: str = "http://localhost:8080",
    model_name: str = None,
    use_llama_swap: bool = USE_LLAMA_SWAP,
):
    """
    Calls a inference-server API endpoint with a formatted user message and system prompt,
    using generation parameters from the LlamaCPPGenerationConfig object.

    This function provides the same interface as call_llama_cpp_chatmodel but calls
    a remote inference-server instance instead of a local model.

    Args:
        formatted_string (str): The formatted input text for the user's message.
        system_prompt (str): The system-level instructions for the model.
        gen_config (LlamaCPPGenerationConfig): An object containing generation parameters.
        api_url (str): The base URL of the inference-server API (default: "http://localhost:8080").
        model_name (str): Optional model name to use. If None, uses the default model.
        use_llama_swap (bool): Whether to use llama-swap for the model.
    Returns:
        dict: Response in the same format as call_llama_cpp_chatmodel

    Example:
        # Create generation config
        gen_config = LlamaCPPGenerationConfig(temperature=0.7, max_tokens=100)

        # Call the API
        response = call_inference_server_api(
            formatted_string="Hello, how are you?",
            system_prompt="You are a helpful assistant.",
            gen_config=gen_config,
            api_url="http://localhost:8080"
        )

        # Extract the response text
        response_text = response['choices'][0]['message']['content']

    Integration Example:
        # To use inference-server instead of local model:
        # 1. Set model_source to "inference-server"
        # 2. Provide api_url parameter
        # 3. Call your existing functions as normal

        responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(
            batch_prompts=["Your prompt here"],
            system_prompt="Your system prompt",
            conversation_history=[],
            whole_conversation=[],
            whole_conversation_metadata=[],
            client=None,  # Not used for inference-server
            client_config=None,  # Not used for inference-server
            model_choice="your-model-name",  # Model name on the server
            temperature=0.7,
            reported_batch_no=1,
            local_model=None,  # Not used for inference-server
            tokenizer=None,  # Not used for inference-server
            bedrock_runtime=None,  # Not used for inference-server
            model_source="inference-server",
            MAX_OUTPUT_VALIDATION_ATTEMPTS=3,
            api_url="http://localhost:8080"
        )
    """
    # Extract parameters from the gen_config object
    temperature = gen_config.temperature
    top_k = gen_config.top_k
    top_p = gen_config.top_p
    repeat_penalty = gen_config.repeat_penalty
    seed = gen_config.seed
    max_tokens = gen_config.max_tokens
    stream = gen_config.stream

    # Prepare the request payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_string},
    ]

    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": LLM_STOP_STRINGS if LLM_STOP_STRINGS else [],
    }
    # Add model name if specified and use llama-swap
    if model_name and use_llama_swap:
        payload["model"] = model_name

    # Determine the endpoint based on streaming preference
    if stream:
        endpoint = f"{api_url}/v1/chat/completions"
    else:
        endpoint = f"{api_url}/v1/chat/completions"

    try:
        if stream:
            # Handle streaming response
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=timeout_wait,
            )
            response.raise_for_status()

            final_tokens = []
            output_tokens = 0
            line_buffer = ""

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == "[DONE]":
                            if REPORT_LLM_OUTPUTS_TO_GUI and line_buffer.strip():
                                _report_llm_output_to_gui(line_buffer)
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    print(token, end="", flush=True)
                                    final_tokens.append(token)
                                    output_tokens += 1
                                    if REPORT_LLM_OUTPUTS_TO_GUI:
                                        line_buffer += token
                                        if "\n" in token:
                                            parts = line_buffer.split("\n")
                                            for complete_line in parts[:-1]:
                                                if complete_line.strip():
                                                    _report_llm_output_to_gui(
                                                        complete_line
                                                    )
                                            line_buffer = parts[-1] if parts else ""
                        except json.JSONDecodeError:
                            continue

            if REPORT_LLM_OUTPUTS_TO_GUI and line_buffer.strip():
                _report_llm_output_to_gui(line_buffer)
            print()  # newline after stream finishes

            text = "".join(final_tokens)

            # Estimate input tokens (rough approximation)
            input_tokens = len((system_prompt + "\n" + formatted_string).split())

            return {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        else:
            # Handle non-streaming response
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout_wait,
            )
            response.raise_for_status()

            result = response.json()

            # Ensure the response has the expected format
            if "choices" not in result:
                raise ValueError("Invalid response format from inference-server")

            return result

    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Failed to connect to inference-server at {api_url}: {str(e)}"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from inference-server: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error calling inference-server API: {str(e)}")


###
# LLM FUNCTIONS
###


def construct_gemini_generative_model(
    in_api_key: str,
    temperature: float,
    model_choice: str,
    system_prompt: str,
    max_tokens: int,
    random_seed=seed,
) -> Tuple[object, dict]:
    """
    Constructs a GenerativeModel for Gemini API calls.
    ...
    """
    # Construct a GenerativeModel
    try:
        if in_api_key:
            # print("Getting API key from textbox")
            api_key = in_api_key
            client = ai.Client(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            # print("Searching for API key in environmental variables")
            api_key = os.environ["GOOGLE_API_KEY"]
            client = ai.Client(api_key=api_key)
        else:
            print("No Gemini API key found")
            raise Warning("No Gemini API key found.")
    except Exception as e:
        print("Error constructing Gemini generative model:", e)
        raise Warning("Error constructing Gemini generative model:", e)

    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_tokens, seed=random_seed
    )

    return client, config


def construct_azure_client(in_api_key: str, endpoint: str) -> Tuple[object, dict]:
    """
    Constructs an OpenAI client for Azure/OpenAI AI Inference.
    """
    try:
        key = None
        if in_api_key:
            key = in_api_key
        elif os.environ.get("AZURE_OPENAI_API_KEY"):
            key = os.environ["AZURE_OPENAI_API_KEY"]
        if not key:
            raise Warning("No Azure/OpenAI API key found.")

        if not endpoint:
            endpoint = os.environ.get("AZURE_OPENAI_INFERENCE_ENDPOINT", "")
            if not endpoint:
                # Assume using OpenAI API
                client = OpenAI(
                    api_key=key,
                )
            else:
                # Use the provided endpoint
                client = OpenAI(
                    api_key=key,
                    base_url=f"{endpoint}",
                )

        return client, dict()
    except Exception as e:
        print("Error constructing Azure/OpenAI client:", e)
        raise


def call_aws_bedrock(
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    model_choice: str,
    bedrock_runtime: boto3.Session.client,
    assistant_prefill: str = "",
) -> ResponseObject:
    """
    This function sends a request to AWS Claude with the following parameters:
    - prompt: The user's input prompt to be processed by the model.
    - system_prompt: A system-defined prompt that provides context or instructions for the model.
    - temperature: A value that controls the randomness of the model's output, with higher values resulting in more diverse responses.
    - max_tokens: The maximum number of tokens (words or characters) in the model's response.
    - model_choice: The specific model to use for processing the request.
    - bedrock_runtime: The client object for boto3 Bedrock runtime
    - assistant_prefill: A string indicating the text that the response should start with.

    The function constructs the request configuration, invokes the model, extracts the response text, and returns a ResponseObject containing the text and metadata.
    """

    inference_config = {
        "maxTokens": max_tokens,
        "topP": 0.999,
        "temperature": temperature,
    }

    # Using an assistant prefill only works for Anthropic models.
    if assistant_prefill and "anthropic" in model_choice:
        assistant_prefill_added = True
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                ],
            },
            {
                "role": "assistant",
                # Pre-filling with '|'
                "content": [{"text": assistant_prefill}],
            },
        ]
    else:
        assistant_prefill_added = False
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                ],
            }
        ]

    system_prompt_list = [{"text": system_prompt}]

    # The converse API call.
    api_response = bedrock_runtime.converse(
        modelId=model_choice,
        messages=messages,
        system=system_prompt_list,
        inferenceConfig=inference_config,
    )

    output_message = api_response["output"]["message"]

    if "reasoningContent" in output_message["content"][0]:
        # Extract the reasoning text
        output_message["content"][0]["reasoningContent"]["reasoningText"]["text"]

        # Extract the output text
        if assistant_prefill_added:
            text = assistant_prefill + output_message["content"][1]["text"]
        else:
            text = output_message["content"][1]["text"]
    else:
        if assistant_prefill_added:
            text = assistant_prefill + output_message["content"][0]["text"]
        else:
            text = output_message["content"][0]["text"]

    # The usage statistics are neatly provided in the 'usage' key.
    usage = api_response["usage"]

    # The full API response metadata is in 'ResponseMetadata' if you still need it.
    api_response["ResponseMetadata"]

    # Create ResponseObject with the cleanly extracted data.
    response = ResponseObject(text=text, usage_metadata=usage)

    return response


def calculate_tokens_from_metadata(
    metadata_string: str, model_choice: str, model_name_map: dict
):
    """
    Calculate the number of input and output tokens for given queries based on metadata strings.

    Args:
        metadata_string (str): A string containing all relevant metadata from the string.
        model_choice (str): A string describing the model name
        model_name_map (dict): A dictionary mapping model name to source
    """

    model_name_map[model_choice]["source"]

    # Regex to find the numbers following the keys in the "Query summary metadata" section
    # This ensures we get the final, aggregated totals for the whole query.
    input_regex = r"input_tokens: (\d+)"
    output_regex = r"output_tokens: (\d+)"

    # re.findall returns a list of all matching strings (the captured groups).
    input_token_strings = re.findall(input_regex, metadata_string)
    output_token_strings = re.findall(output_regex, metadata_string)

    # Convert the lists of strings to lists of integers and sum them up
    total_input_tokens = sum([int(token) for token in input_token_strings])
    total_output_tokens = sum([int(token) for token in output_token_strings])

    number_of_calls = len(input_token_strings)

    print(f"Found {number_of_calls} LLM call entries in metadata.")
    print("-" * 20)
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")

    return total_input_tokens, total_output_tokens, number_of_calls
