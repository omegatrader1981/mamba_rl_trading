# Use CUDA 11.8 base (true CUDA 11.8)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch 2.0.1 + CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install app dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ✅ Install mamba-ssm with SHA256 verification (real hash from your system)
RUN curl -fL -o /tmp/mamba_ssm.whl "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2%2Bcu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    echo "f2cd537a0bc57ef573b6d4a87e547afa661902ebc6fb6dbbb7c6ee9a60396b2b  /tmp/mamba_ssm.whl" | sha256sum -c - && \
    pip3 install --no-cache-dir "/tmp/mamba_ssm.whl" && \
    rm "/tmp/mamba_ssm.whl"

# Verify all dependencies
RUN python3 -c "\
import torch; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
import causal_conv1d; print('✅ causal-conv1d installed'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All critical dependencies verified!')"

# Copy code
COPY . /opt/ml/code
WORKDIR /opt/ml/code

# Training wrapper
RUN printf '#!/bin/bash\n\
set -e\n\
echo "=========================================="\n\
echo "SAGEMAKER CONTAINER STARTING"\n\
echo "=========================================="\n\
echo "PyTorch: $(python3 -c "import torch; print(torch.__version__)")"\n\
echo "CUDA available: $(python3 -c "import torch; print(torch.cuda.is_available())")"\n\
echo "Mamba: $(python3 -c "import mamba_ssm; print(mamba_ssm.__version__)")"\n\
echo "Starting training..."\n\
exec python3 src/train.py "$@"\n' > train_wrapper.sh

RUN chmod +x train_wrapper.sh

ENV PYTHONUNBUFFERED=1 PYTHONPATH="/opt/ml/code"
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["./train_wrapper.sh"]
