# Base: CUDA 11.8 + full toolkit
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git build-essential \
        libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -O /tmp/m.sh && \
    bash /tmp/m.sh -b -p /opt/conda && \
    rm /tmp/m.sh && \
    conda clean -afy

# Create Conda env
RUN conda create -n env python=3.10 -y

# Install PyTorch via pip (uses system CUDA)
RUN /opt/conda/envs/env/bin/pip install --no-cache-dir \
        torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# Install mamba-ssm via GitHub wheel
RUN /opt/conda/envs/env/bin/pip install --no-cache-dir \
        https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install app deps
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN /opt/conda/envs/env/bin/pip install -r requirements.txt

# Copy code
COPY . .

# <<< NEW CONDITIONAL ENTRYPOINT SCRIPT >>>
RUN printf '#!/bin/bash\n\
set -e\n\
# Activate the conda environment\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate env\n\
\n\
echo "=========================================="\n\
echo "SAGEMAKER CONTAINER STARTING"\n\
echo "=========================================="\n\
echo "Python: $(python --version)"\n\
echo "Experiment Hyperparameter: ${SM_HP_EXPERIMENT:-not_set}"\n\
echo "=========================================="\n\
\n\
# Check which script to run based on the 'experiment' hyperparameter\n\
if [ "${SM_HP_EXPERIMENT}" = "baseline_test" ]; then\n\
    echo ">>> Running baseline test..."\n\
    exec python run_baseline_test.py "$@"\n\
else\n\
    echo ">>> Running main training (default)..."\n\
    exec python src/train.py "$@"\n\
fi\n' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
