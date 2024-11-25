# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm AS builder

# Install system dependencies. Need to specify -y for poppler to get it to install
RUN apt-get update \
    && apt-get install -y \
        g++ \
        make \
        cmake \
        unzip \
        libcurl4-openssl-dev \
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
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Define a build argument for the mode (gradio or lambda)
ARG APP_MODE=gradio

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

# Create required directories
RUN mkdir -p /home/user/app/output \
    && mkdir -p /home/user/app/tld \
    && mkdir -p /home/user/app/logs \
    && chown -R user:user /home/user/app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Use a conditional entrypoint based on the APP_MODE argument (deprecated, now created beforehand in folder)
# RUN if [ "$APP_MODE" = "lambda" ]; then \
#         echo '#!/bin/sh\nexec python -m awslambdaric' > /entrypoint.sh; \
#     else \        
#         echo '#!/bin/sh\nexec python app.py' > /entrypoint.sh; \
#     fi && chmod +x /entrypoint.sh

# Entrypoint helps to switch between Gradio and Lambda mode
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Switch to the "user" user
USER user

# Set environmental variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_THEME=huggingface \
    TLDEXTRACT_CACHE=$HOME/app/tld/.tld_set_snapshot \
    SYSTEM=spaces

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the app code to the container
COPY --chown=user . $HOME/app



ENTRYPOINT [ "/entrypoint.sh" ]

# Default command for Lambda mode
CMD [ "lambda_entrypoint.lambda_handler" ]