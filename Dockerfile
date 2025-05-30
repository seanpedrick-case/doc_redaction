# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.11.11-slim-bookworm AS builder

# Install system dependencies. Need to specify -y for poppler to get it to install
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

RUN pip install --no-cache-dir --target=/install -r requirements.txt

RUN rm requirements.txt

# Add lambda_entrypoint.py to the container
COPY lambda_entrypoint.py .

COPY entrypoint.sh .

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.11-slim-bookworm

# Define a build argument with a default value
ARG APP_MODE=gradio

# Echo the APP_MODE during the build to confirm its value
RUN echo "APP_MODE is set to: ${APP_MODE}"

# Set APP_MODE as an environment variable for runtime
ENV APP_MODE=${APP_MODE}

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

ENV APP_HOME=/home/user

ENV GRADIO_TEMP_DIR=/tmp/gradio_tmp/ \
    TLDEXTRACT_CACHE=/tmp/tld/ \
    MPLCONFIGDIR=/tmp/matplotlib_cache/ \
    GRADIO_OUTPUT_FOLDER=$APP_HOME/app/output/ \
    GRADIO_INPUT_FOLDER=$APP_HOME/app/input/ \
    FEEDBACK_LOGS_FOLDER=$APP_HOME/app/feedback/ \
    ACCESS_LOGS_FOLDER=$APP_HOME/app/logs/ \
    USAGE_LOGS_FOLDER=$APP_HOME/app/usage/ \
    CONFIG_FOLDER=$APP_HOME/app/config/

# Create required directories
RUN mkdir -p $APP_HOME/app/{output, input, logs, usage, feedback, config} \
    && chown -R user:user $APP_HOME/app

# For system /tmp and /var/tmp - make them world-writable with sticky bit, owned by user
RUN mkdir -p /tmp && chown user:user /tmp && chmod 1777 /tmp
RUN mkdir -p /var/tmp && chown user:user /var/tmp && chmod 1777 /var/tmp
RUN mkdir -p /tmp/matplotlib_cache && chown user:user /tmp/matplotlib_cache && chmod 1777 /tmp/matplotlib_cache
RUN mkdir -p /tmp/tld && chown user:user /tmp/tld && chmod 1777 /tmp/tld
RUN mkdir -p /tmp/gradio_tmp && chown user:user /tmp/gradio_tmp && chmod 1777 /tmp/gradio_tmp

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Entrypoint helps to switch between Gradio and Lambda mode
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure permissions are really user:user again after copying
RUN chown -R user:user $APP_HOME/app && chmod -R u+rwX $APP_HOME/app

# Switch to the "user" user
USER user

# --- ADD VOLUME DIRECTIVES ---
# If using Fargate, These paths MUST EXACTLY MATCH containerPath in your Fargate task definition mountPoints
VOLUME ["/tmp/matplotlib_cache"]
VOLUME ["/tmp/gradio_tmp"]
VOLUME ["/tmp/tld"]
VOLUME ["/home/user/app/output"]
VOLUME ["/home/user/app/input"]
VOLUME ["/home/user/app/logs"]
VOLUME ["/home/user/app/usage"]
VOLUME ["/home/user/app/feedback"]
VOLUME ["/home/user/app/config"]
VOLUME ["/tmp"]
VOLUME ["/var/tmp"]

# Set environment variables
ENV PATH=$APP_HOME/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

# Set the working directory to the user's home directory
WORKDIR $APP_HOME/app

# Copy the app code to the container
COPY --chown=user . $APP_HOME/app

ENTRYPOINT [ "/entrypoint.sh" ]

# Default command for Lambda mode
CMD [ "lambda_entrypoint.lambda_handler" ]