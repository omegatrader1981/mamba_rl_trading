# syntax=docker/dockerfile:1.4
# Use the official NVIDIA PyTorch image to guarantee version alignment
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Switch to root to install packages
USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Make pip caching explicit and consistent
    PIP_CACHE_DIR=/root/.cache/pip

# System packages with apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        wget git

# PyTorch is already installed in the base image. We just install our apps.

# Install mamba-ssm from the wheel that matches PyTorch 2.4
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install causal-conv1d
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir "causal-conv1d>=1.1.0"

# Verify critical dependencies are working
RUN python -c "\
import torch, mamba_ssm, causal_conv1d; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All critical dependencies verified!')"

# Copy requirements and install with cache
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy code (changes most often - do last for better caching)
COPY . .

# Create conditional entrypoint
RUN printf '#!/bin/bash\n\
set -e\n\
echo "=========================================="\n\
echo "SAGEMAKER CONTAINER STARTING"\n\
echo "=========================================="\n\
echo "Python: $(python --version)"\n\
echo "PyTorch: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
echo "Mamba: $(python -c \"import mamba_ssm; print(mamba_ssm.__version__)\")"\n\
echo "Experiment: ${SM_HP_EXPERIMENT:-default}"\n\
echo "=========================================="\n\
\n\
if [ "${SM_HP_EXPERIMENT}" = "baseline_test" ]; then\n\
    echo ">>> Running baseline test..."\n\
    exec python run_baseline_test.py "$@"\n\
else\n\
    echo ">>> Running main training..."\n\
    exec python src/pipeline/train.py "$@"\n\
fi\n' > /entrypoint.sh && chmod +x /entrypoint.sh

# The base image has a default user, but we create one for consistency if needed.
# For SageMaker, it's often better to let it run as root or its default user.
# We will rely on the base image's user setup.

ENTRYPOINT ["/entrypoint.sh"]
