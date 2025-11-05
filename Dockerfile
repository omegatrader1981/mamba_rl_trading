# Base: CUDA 11.8 + full toolkit
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git build-essential \
        libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3.10)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -O /tmp/m.sh && \
    bash /tmp/m.sh -b -p /opt/conda && \
    rm /tmp/m.sh && \
    conda clean -afy

# Create Conda env + install PyTorch 2.4.0 for CUDA 11.8
RUN conda create -n env python=3.10 -y && \
    /opt/conda/envs/env/bin/pip install --no-cache-dir \
        torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# ✅ Install mamba-ssm from **v2.2.4** (latest stable release with pre-built wheels)
RUN /opt/conda/envs/env/bin/pip install --no-cache-dir \
        "mamba-ssm==2.2.4" \
        -f https://github.com/state-spaces/mamba/releases/download/v2.2.4 \
        --only-binary=:all:

# Verify CUDA compatibility
RUN /opt/conda/envs/env/bin/python -c "\
import torch, mamba_ssm; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ Mamba: {mamba_ssm.__version__}'); \
assert torch.version.cuda == '11.8', 'CUDA version mismatch!'"

# Install app dependencies
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN /opt/conda/envs/env/bin/pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Entrypoint: Use conda run to activate env reliably
RUN printf '#!/bin/bash\n\
set -e\n\
exec /opt/conda/bin/conda run -n env --no-capture-output python src/train.py "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
