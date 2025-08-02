# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential pkg-config git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/ml/code

# Install all dependencies from requirements.txt (including pinned PyTorch)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt --no-cache-dir

# Install mamba after PyTorch is locked in place
RUN pip install packaging ninja && \
    pip install causal-conv1d==1.4.0 mamba-ssm==2.2.2 --no-cache-dir

# Copy project code
COPY . .

# Test everything works
RUN python -c "import torch, mamba_ssm; print(f'✅ PyTorch: {torch.__version__}, mamba-ssm: {getattr(mamba_ssm, \"__version__\", \"N/A\")}')"

# SageMaker configuration
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=src/train.py
ENTRYPOINT ["python", "-m", "sagemaker_training.cli.train"]
