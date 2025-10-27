# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.12.11-slim-trixie AS builder

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt .

RUN pip install --verbose --no-cache-dir --target=/install -r requirements.txt && rm requirements.txt

# Optionally install PaddleOCR if the INSTALL_PADDLEOCR environment variable is set to True. See requirements.txt for more details, including installing the GPU version of PaddleOCR.
ARG INSTALL_PADDLEOCR=False
ENV INSTALL_PADDLEOCR=${INSTALL_PADDLEOCR}

RUN if [ "$INSTALL_PADDLEOCR" = "True" ]; then \
    pip install --verbose --no-cache-dir --target=/install paddleocr==3.3.0 paddlepaddle==3.2.0; \
fi

# ===================================================================
# Stage 2: A common 'base' for both Lambda and Gradio
# ===================================================================
FROM public.ecr.aws/docker/library/python:3.12.11-slim-trixie AS base

# Set build-time and runtime environment variable for whether to run in Gradio mode or Lambda mode
ARG APP_MODE=gradio
ENV APP_MODE=${APP_MODE}

# Set build-time and runtime environment variable for whether to run in FastAPI mode
ARG RUN_FASTAPI=0
ENV RUN_FASTAPI=${RUN_FASTAPI}

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

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
    GRADIO_ANALYTICS_ENABLED=False \
    DEFAULT_CONCURRENCY_LIMIT=3

# Copy Python packages from the builder stage
COPY --from=builder /install /usr/local/lib/python3.12/site-packages/
COPY --from=builder /install/bin /usr/local/bin/

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