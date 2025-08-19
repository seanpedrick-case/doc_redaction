# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.11.11-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
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

RUN pip install --no-cache-dir --verbose --target=/install -r requirements.txt && rm requirements.txt

# Add lambda entrypoint and script
COPY lambda_entrypoint.py .
COPY entrypoint.sh .

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.11-slim-bookworm

# Set build-time and runtime environment variable
ARG APP_MODE=gradio
ENV APP_MODE=${APP_MODE}

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
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
    XDG_CACHE_HOME=/tmp/xdg_cache/user_1000

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

# Now handle the /tmp and /var/tmp directories and their subdirectories
RUN mkdir -p /tmp/gradio_tmp /tmp/tld /tmp/matplotlib_cache /tmp /var/tmp ${XDG_CACHE_HOME} \
    && chown user:user /tmp /var/tmp /tmp/gradio_tmp /tmp/tld /tmp/matplotlib_cache ${XDG_CACHE_HOME} \
    && chmod 1777 /tmp /var/tmp /tmp/gradio_tmp /tmp/tld /tmp/matplotlib_cache \
    && chmod 700 ${XDG_CACHE_HOME}

RUN mkdir -p ${APP_HOME}/.paddlex/official_models \
    && chown user:user \
    ${APP_HOME}/.paddlex/official_models \
    && chmod 755 \
    ${APP_HOME}/.paddlex/official_models

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Copy app code and entrypoint with correct ownership
COPY --chown=user . $APP_HOME/app

# Copy and chmod entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to user
USER user

# Declare working directory
WORKDIR $APP_HOME/app

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
VOLUME ["/home/user/.paddlex/official_models"]
VOLUME ["/tmp"]
VOLUME ["/var/tmp"]

# Set runtime environment
ENV PATH=$APP_HOME/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

ENTRYPOINT ["/entrypoint.sh"]

CMD ["lambda_entrypoint.lambda_handler"]