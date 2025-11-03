# Base image with CUDA 11.8 and cuDNN 8 development tools
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies including Python, pip, AND build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory for SageMaker
WORKDIR /opt/ml/code

# --- ROBUST MULTI-STEP INSTALLATION ---

# STEP 1: Install PyTorch (the core dependency)
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# STEP 2: Install the complex, compiled packages (Mamba)
RUN pip install --no-cache-dir mamba-ssm causal-conv1d

# STEP 3: Install the remaining application-level packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 4: Finally, copy the rest of the application code
COPY . /opt/ml/code

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM src/train.py
ENV PYTHONUNBUFFERED=1
