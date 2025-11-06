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

# Create Conda env
RUN conda create -n env python=3.10 -y

# Install PyTorch 2.4.0 + mamba-ssm (pre-built binary, CUDA 11.8 compatible)
RUN conda install -n env -c pytorch -c nvidia -c conda-forge \
        pytorch=2.4.0 torchvision torchaudio pytorch-cuda=11.8 \
        mamba-ssm -y

# Verify
RUN /opt/conda/envs/env/bin/python -c "\
import torch, mamba_ssm; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ Mamba: {mamba_ssm.__version__}'); \
assert torch.version.cuda == '11.8', 'CUDA mismatch!'"

# Install app dependencies
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN /opt/conda/envs/env/bin/pip install -r requirements.txt

# Copy code
COPY . .

# Entrypoint
RUN printf '#!/bin/bash\n\
set -e\n\
exec /opt/conda/bin/conda run -n env --no-capture-output python src/train.py "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
