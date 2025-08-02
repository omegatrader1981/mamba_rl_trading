# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# PRODUCTION VERSION: Pure pip approach to avoid conda TOS issues

# FIXED: Use runtime image that includes both devel tools AND runtime libraries
FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# --- Environment Setup ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    wget \
    git \
    build-essential \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support using pip (compatible versions)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Install Mamba dependencies (pre-built wheels)
RUN pip install packaging ninja && \
    pip install causal-conv1d==1.4.0 --no-cache-dir && \
    pip install mamba-ssm==2.2.2 --no-cache-dir

# --- Application Setup ---
WORKDIR /opt/ml/code

# Copy and install the rest of our Python packages
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Copy the project code into the final image
COPY . /opt/ml/code/

# --- Enhanced Diagnostics ---
RUN echo "--- Docker Build: COMPREHENSIVE Check ..." && \
    python -c "
import sys
print(f'Python version: {sys.version}')

# Check CUDA libraries are available
print('=== CUDA Library Check ===')
import ctypes.util
for lib in ['cublas', 'cufft', 'curand']:
    path = ctypes.util.find_library(lib)
    print(f'{lib}: {\\"✅ Found\\" if path else \\"❌ Missing\\"} {path or \\"\\"}')

# Test PyTorch CUDA
print('=== PyTorch Test ===')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')

# Test mamba-ssm
print('=== Mamba-SSM Test ===')
import mamba_ssm
print(f'mamba-ssm: {getattr(mamba_ssm, \\"__version__\\", \\"N/A\\")}')

print('✅ All imports successful!')
"

# --- SageMaker Configuration ---
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=src/train.py

# Simple Python entrypoint (no conda needed)
ENTRYPOINT ["python", "-m", "sagemaker_training.cli.train"]
