# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# FINAL PRODUCTION VERSION: Using PyTorch base image for reliability.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# --- Environment Setup ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies, including all Chrome dependencies.
RUN apt-get update && apt-get install -y \
    wget git build-essential ca-certificates fonts-liberation libasound2 libatk-bridge2.0-0 \
    libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgbm1 \
    libgcc1 libglib2.0-0 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 libpangocairo-1.0-0 \
    libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 \
    libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 lsb-release \
    xdg-utils libvulkan1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install Google Chrome for the Kaleido backend.
RUN apt-get update && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    rm -rf /var/lib/apt/lists/*
# PyTorch is already installed in the base image, so we can directly install Mamba dependencies
RUN pip install packaging ninja && \
    pip install causal-conv1d==1.4.0 --no-cache-dir && \
    pip install mamba-ssm==2.2.2 --no-cache-dir
# --- Application Setup ---
WORKDIR /opt/ml/code
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY . /opt/ml/code/
# --- Simple Diagnostics Check ---
RUN echo "--- Docker Build: COMPREHENSIVE Check ---" && \
    python -c "import torch; print('PyTorch:', torch.__version__)" && \
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" && \
    python -c "import mamba_ssm; print('Mamba SSM: OK')"
# --- SageMaker Configuration ---
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=src/train.py
ENTRYPOINT ["python", "-m", "sagemaker_training.cli.train"]
