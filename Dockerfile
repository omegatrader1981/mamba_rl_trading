# Use official PyTorch 2.4 + CUDA 11.8 base image
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install app dependencies (no PyTorch — already installed)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install pre-built causal-conv1d wheel (for PyTorch 2.4 + CUDA 11.8)
RUN pip3 install --no-cache-dir \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu118torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
    mamba-ssm

# Verify
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
echo "PyTorch: $(python3 -c \"import torch; print(torch.__version__\") )"\n\
echo "CUDA available: $(python3 -c \"import torch; print(torch.cuda.is_available())\")"\n\
echo "Mamba: $(python3 -c \"import mamba_ssm; print(mamba_ssm.__version__)\" )"\n\
echo "Starting training..."\n\
exec python3 src/train.py \"$@\"\n' > train_wrapper.sh

RUN chmod +x train_wrapper.sh

ENV PYTHONUNBUFFERED=1 PYTHONPATH="/opt/ml/code"
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["./train_wrapper.sh"]
