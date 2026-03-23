# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.12.13-slim-trixie AS builder

# Install system dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        g++ \
        make \
        cmake \
        unzip \
        libcurl4-openssl-dev \
        git \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements_lightweight.txt .

RUN pip install --verbose --no-cache-dir --target=/install -r requirements_lightweight.txt && rm requirements_lightweight.txt

# Optionally install PaddleOCR if the INSTALL_PADDLEOCR environment variable is set to True. Note that GPU-enabled PaddleOCR is unlikely to work in the same environment as a GPU-enabled version of PyTorch, so it is recommended to install PaddleOCR as a CPU-only version if you want to use GPU-enabled PyTorch.

ARG INSTALL_PADDLEOCR=False
ENV INSTALL_PADDLEOCR=${INSTALL_PADDLEOCR}

ARG PADDLE_GPU_ENABLED=False
ENV PADDLE_GPU_ENABLED=${PADDLE_GPU_ENABLED}

RUN if [ "$INSTALL_PADDLEOCR" = "True" ] && [ "$PADDLE_GPU_ENABLED" = "False" ]; then \
    pip install --verbose --no-cache-dir --target=/install "protobuf<=7.34.0" && \
    pip install --verbose --no-cache-dir --target=/install "paddlepaddle<=3.2.1" && \
    pip install --verbose --no-cache-dir --target=/install "paddleocr<=3.3.0"; \
elif [ "$INSTALL_PADDLEOCR" = "True" ] && [ "$PADDLE_GPU_ENABLED" = "True" ]; then \
    pip install --verbose --no-cache-dir --target=/install "protobuf<=7.34.0" && \
    pip install --verbose --no-cache-dir --target=/install "paddlepaddle-gpu<=3.2.1" --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ && \
    pip install --verbose --no-cache-dir --target=/install "paddleocr<=3.3.0"; \
fi

ARG INSTALL_VLM=False
ENV INSTALL_VLM=${INSTALL_VLM}

ARG TORCH_GPU_ENABLED=False
ENV TORCH_GPU_ENABLED=${TORCH_GPU_ENABLED}

# Optionally install VLM if the INSTALL_VLM environment variable is set to True. Use e.g. --index-url https://download.pytorch.org/whl/cu129 for a specifc cuda compatible GPU version of PyTorch, otherwise the following CPU compatible versions will be installed.
RUN if [ "$INSTALL_VLM" = "True" ] && [ "$TORCH_GPU_ENABLED" = "False" ]; then \
    pip install --verbose --no-cache-dir --target=/install "torch<=2.8.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --verbose --no-cache-dir --target=/install "torchvision<=0.24.1" && \
    pip install --verbose --no-cache-dir --target=/install \
        "transformers<=5.30.0" \
        "accelerate<=1.13.0" \
        "bitsandbytes<=0.49.2" \
        "sentencepiece<=0.2.1"; \
elif [ "$INSTALL_VLM" = "True" ] && [ "$TORCH_GPU_ENABLED" = "True" ]; then \
    pip install --verbose --no-cache-dir --target=/install "torch<=2.8.0" --index-url https://download.pytorch.org/whl/cu129 && \
    pip install --verbose --no-cache-dir --target=/install "torchvision<=0.24.1" && \
    pip install --verbose --no-cache-dir --target=/install \
        "transformers<=5.30.0" \
        "accelerate<=1.13.0" \
        "bitsandbytes<=0.49.2" \
        "sentencepiece<=0.2.1" && \
    pip install --verbose --no-cache-dir --target=/install "optimum<=2.1.0" && \
    pip install --verbose --no-cache-dir --target=/install  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
    pip install --verbose --no-cache-dir --target=/install  https://github.com/ModelCloud/GPTQModel/releases/download/v5.8.0/gptqmodel-5.8.0+cu128torch2.8-cp312-cp312-linux_x86_64.whl; \
fi

# ===================================================================
# Stage 2: A common base for both Lambda and Gradio
# ===================================================================
FROM public.ecr.aws/docker/library/python:3.12.13-slim-trixie AS base

# MUST re-declare ARGs in every stage where they are used in RUN commands
ARG TORCH_GPU_ENABLED=False
ARG PADDLE_GPU_ENABLED=False

ENV TORCH_GPU_ENABLED=${TORCH_GPU_ENABLED}
ENV PADDLE_GPU_ENABLED=${PADDLE_GPU_ENABLED}

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 && \
    if [ "$TORCH_GPU_ENABLED" = "True" ] || [ "$PADDLE_GPU_ENABLED" = "True" ]; then \
        apt-get install -y --no-install-recommends libgomp1; \
    fi && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/home/user

# Set env variables for Gradio & other apps
ENV GRADIO_TEMP_DIR=/tmp/gradio_tmp/ \
    TLDEXTRACT_CACHE=/tmp/tld/ \
    MPLCONFIGDIR=/tmp/matplotlib_cache/ \
    GRADIO_OUTPUT_FOLDER=$APP_HOME/app/output/ \
    GRADIO_INPUT_FOLDER=$APP_HOME/app/input/ \
    FEEDBACK_LOGS_FOLDER=$APP_HOME/app/feedback/ \
    ACCESS_LOGS_FOLDER=$APP_HOME/app/logs/ \
    USAGE_LOGS_FOLDER=$APP_HOME/app/usage/ \
    CONFIG_FOLDER=$APP_HOME/app/config/ \
    XDG_CACHE_HOME=/tmp/xdg_cache/user_1000 \
    TESSERACT_DATA_FOLDER=/usr/share/tessdata \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PATH=$APP_HOME/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \    
    GRADIO_ANALYTICS_ENABLED=False

# Copy Python packages from the builder stage
COPY --from=builder /install /usr/local/lib/python3.12/site-packages/
COPY --from=builder /install/bin /usr/local/bin/

# Reinstall protobuf into the final site-packages. Builder uses multiple `pip install --target=/install`
# passes; that can break the `google` namespace so `google.protobuf` is missing and Paddle fails at import.
RUN pip install --no-cache-dir "protobuf<=7.34.0"

# Copy your application code and entrypoint
COPY . ${APP_HOME}/app
COPY entrypoint.sh ${APP_HOME}/app/entrypoint.sh
# Fix line endings and set execute permissions
RUN sed -i 's/\r$//' ${APP_HOME}/app/entrypoint.sh \
    && chmod +x ${APP_HOME}/app/entrypoint.sh

WORKDIR ${APP_HOME}/app

# ===================================================================
# FINAL Stage 3: The Lambda Image (runs as root for simplicity)
# ===================================================================
FROM base AS lambda
# Set runtime ENV for Lambda mode
ENV APP_MODE=lambda
ENTRYPOINT ["/home/user/app/entrypoint.sh"]
CMD ["lambda_entrypoint.lambda_handler"]

# ===================================================================
# FINAL Stage 4: The Gradio Image (runs as a secure, non-root user)
# ===================================================================
FROM base AS gradio
# Set runtime ENV for Gradio mode
ENV APP_MODE=gradio

# Create non-root user
RUN useradd -m -u 1000 user

# Create the base application directory and set its ownership
RUN mkdir -p ${APP_HOME}/app && chown user:user ${APP_HOME}/app

# Create required sub-folders within the app directory and set their permissions
# This ensures these specific directories are owned by 'user'
RUN mkdir -p \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config \
    && chown user:user \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config \
    && chmod 755 \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config 

# Now handle the /tmp and /var/tmp directories and their subdirectories, paddle, spacy, tessdata
RUN mkdir -p /tmp/gradio_tmp /tmp/tld /tmp/matplotlib_cache /tmp /var/tmp ${XDG_CACHE_HOME} \
    && chown user:user /tmp /var/tmp /tmp/gradio_tmp /tmp/tld /tmp/matplotlib_cache ${XDG_CACHE_HOME} \
    && chmod 1777 /tmp /var/tmp /tmp/gradio_tmp /tmp/tld /tmp/matplotlib_cache \
    && chmod 700 ${XDG_CACHE_HOME} \
    && mkdir -p ${APP_HOME}/.paddlex \
    && chown user:user ${APP_HOME}/.paddlex \
    && chmod 755 ${APP_HOME}/.paddlex \
    && mkdir -p ${APP_HOME}/.local/share/spacy/data \
    && chown user:user ${APP_HOME}/.local/share/spacy/data \
    && chmod 755 ${APP_HOME}/.local/share/spacy/data \
    && mkdir -p /usr/share/tessdata \
    && chown user:user /usr/share/tessdata \
    && chmod 755 /usr/share/tessdata

# Fix apply user ownership to all files in the home directory
RUN chown -R user:user /home/user

# Set permissions for Python executable
RUN chmod 755 /usr/local/bin/python

# Declare volumes (NOTE: runtime mounts will override permissions — handle with care)
VOLUME ["/tmp/matplotlib_cache"]
VOLUME ["/tmp/gradio_tmp"]
VOLUME ["/tmp/tld"]
VOLUME ["/home/user/app/output"]
VOLUME ["/home/user/app/input"]
VOLUME ["/home/user/app/logs"]
VOLUME ["/home/user/app/usage"]
VOLUME ["/home/user/app/feedback"]
VOLUME ["/home/user/app/config"]
VOLUME ["/home/user/.paddlex"]
VOLUME ["/home/user/.local/share/spacy/data"]
VOLUME ["/usr/share/tessdata"]
VOLUME ["/tmp"]
VOLUME ["/var/tmp"]

USER user

EXPOSE $GRADIO_SERVER_PORT

ENTRYPOINT ["/home/user/app/entrypoint.sh"]
CMD ["python", "app.py"]