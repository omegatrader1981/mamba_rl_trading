# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

# System packages with apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git build-essential \
        libgl1-mesa-glx libglib2.0-0

# Install Miniconda (cached layer)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -O /tmp/m.sh && \
    bash /tmp/m.sh -b -p /opt/conda && \
    rm /tmp/m.sh && \
    conda clean -afy

# Create Conda env (cached)
RUN conda create -n env python=3.10 -y

# Install PyTorch with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/env/bin/pip install --no-cache-dir \
        torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# Install mamba-ssm with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/env/bin/pip install --no-cache-dir \
        https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install causal-conv1d
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/env/bin/pip install --no-cache-dir "causal-conv1d>=1.1.0"

# Copy requirements and install with cache
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/env/bin/pip install -r requirements.txt

# Copy code (this layer changes most often, so put it last)
COPY . .

# Create entrypoint
RUN printf '#!/bin/bash\n\
set -e\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate env\n\
\n\
echo "=========================================="\n\
echo "SAGEMAKER CONTAINER STARTING"\n\
echo "=========================================="\n\
echo "Python: $(python --version)"\n\
echo "Experiment: ${SM_HP_EXPERIMENT:-not_set}"\n\
echo "=========================================="\n\
\n\
if [ "${SM_HP_EXPERIMENT}" = "baseline_test" ]; then\n\
    echo ">>> Running baseline test..."\n\
    exec python run_baseline_test.py "$@"\n\
else\n\
    echo ">>> Running main training..."\n\
    exec python src/pipeline/train.py "$@"\n\
fi\n' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
