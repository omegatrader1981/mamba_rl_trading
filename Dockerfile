# Base: CUDA 11.8 (meets Mamba's CUDA >= 11.6 requirement)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch 2.1.0 + CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchaudio==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install your RL/trading stack
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install Mamba from PRE-BUILT wheels for PyTorch 2.1 + CUDA 11
RUN curl -fL "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2%2Bcu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    -o "/tmp/mamba_ssm-2.2.2+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    pip3 install --no-cache-dir "/tmp/mamba_ssm-2.2.2+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    rm "/tmp/mamba_ssm-2.2.2+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

RUN curl -fL "https://github.com/state-spaces/mamba/releases/download/v1.4.0/causal_conv1d-1.4.0%2Bcu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    -o "/tmp/causal_conv1d-1.4.0+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    pip3 install --no-cache-dir "/tmp/causal_conv1d-1.4.0+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    rm "/tmp/causal_conv1d-1.4.0+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Verify
RUN python3 -c "\
import torch; print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
import causal_conv1d; print('✅ causal-conv1d'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('Success!')"

# Copy code
COPY . /opt/ml/code
WORKDIR /opt/ml/code

# Training wrapper
RUN printf '#!/bin/bash\nset -e\nexec python3 src/train.py "$@"\n' > train_wrapper.sh
RUN chmod +x train_wrapper.sh

ENV PYTHONUNBUFFERED=1 PYTHONPATH="/opt/ml/code"
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["./train_wrapper.sh"]
