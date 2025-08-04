# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# FINAL PRODUCTION VERSION: With all dependency and TOS fixes.
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
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
# Install Miniconda for robust package management
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
# Add conda to the system PATH
ENV PATH="/opt/conda/bin:$PATH"
# Accept Anaconda Terms of Service before creating the environment
RUN conda config --set channel_priority flexible && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels nvidia
# Create conda environment
RUN conda create -n mamba_env python=3.10 -y
# Set the shell to use the conda environment for all subsequent RUN commands
SHELL ["conda", "run", "-n", "mamba_env", "/bin/bash", "-c"]
# Install the known-good, stable combination of PyTorch and Mamba
RUN conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
RUN pip install packaging ninja && \
    pip install causal-conv1d==1.4.0 --no-cache-dir && \
    pip install mamba-ssm==2.2.2 --no-cache-dir
# --- Application Setup ---
WORKDIR /opt/ml/code
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY . /opt/ml/code/
# --- Diagnostics and Final Configuration ---
# Create a simple Python script for diagnostics to avoid escaping issues
RUN echo 'import sys, torch, mamba_ssm' > /tmp/check.py && \
    echo 'print(f"Python: {sys.version}")' >> /tmp/check.py && \
    echo 'print(f"PyTorch: {torch.__version__}")' >> /tmp/check.py && \
    echo 'print(f"CUDA Available: {torch.cuda.is_available()}")' >> /tmp/check.py && \
    echo 'print(f"Mamba SSM: {getattr(mamba_ssm, \"__version__\", \"N/A\")}")' >> /tmp/check.py
RUN echo "--- Docker Build: COMPREHENSIVE Check ---" && \
    python /tmp/check.py && \
    rm /tmp/check.py
# --- SageMaker Configuration ---
ENV CONDA_DEFAULT_ENV=mamba_env
ENV PATH="/opt/conda/envs/mamba_env/bin:$PATH"
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=src/train.py
ENTRYPOINT ["conda", "run", "-n", "mamba_env", "python", "-m", "sagemaker_training.cli.train"]
