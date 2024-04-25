FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt

# Set up a new user named "user" with user ID 1000
#RUN useradd -m -u 1000 user

# Change ownership of /home/user directory
#RUN chown -R user:user /home/user

# Create the temp files directory and set its permissions
#RUN mkdir -p /home/user/tmp && chown -R user:user /home/user/tmp

# Switch to the "user" user
#USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=7861 \
	GRADIO_THEME=huggingface \
	#GRADIO_TEMP_DIR=$HOME/tmp \
	#GRADIO_ROOT_PATH=/address-match \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
#COPY --chown=user . $HOME/app
COPY . $HOME/app

CMD ["python", "app.py"]