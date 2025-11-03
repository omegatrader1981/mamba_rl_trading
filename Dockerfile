# Base image with CUDA 11.8 and cuDNN 8 development tools
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies including Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory for SageMaker
WORKDIR /opt/ml/code

# --- CORRECTED INSTALLATION ---
# STEP 1: Install PyTorch and its CUDA-specific dependencies FIRST.
# This ensures torch is available for other packages that need it during their installation.
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# STEP 2: Now, copy and install the rest of the requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 3: Finally, copy the rest of the application code.
COPY . /opt/ml/code

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM src/train.py
ENV PYTHONUNBUFFERED=1
