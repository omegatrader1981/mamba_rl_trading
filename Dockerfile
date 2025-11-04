# Use NVIDIA PyTorch 24.09 → PyTorch 2.4.1 (stable) + CUDA 12.5
# Confirmed compatible with causal-conv1d and mamba-ssm
FROM nvcr.io/nvidia/pytorch:24.09-py3

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (your RL/trading stack)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Mamba stack — safe on PyTorch 2.4.1
# Pip will use wheel if compatible, or build from source (now safe!)
RUN pip install --no-cache-dir "causal-conv1d>=1.5.0" "mamba-ssm>=2.2.6"

# Verify critical components
RUN python -c "\
import torch; print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
import causal_conv1d; print('✅ causal-conv1d: installed'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All critical dependencies verified!')"

# Copy code
COPY . /opt/ml/code
WORKDIR /opt/ml/code

# Training wrapper for SageMaker
RUN printf '#!/bin/bash\n\
set -e\n\
echo "=========================================="\n\
echo "SAGEMAKER CONTAINER STARTING"\n\
echo "=========================================="\n\
echo "PyTorch: $(python -c "import torch; print(torch.__version__)")"\n\
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"\n\
echo "Mamba: $(python -c "import mamba_ssm; print(mamba_ssm.__version__)")"\n\
echo "Starting training..."\n\
exec python src/train.py "$@"\n' > /opt/ml/code/train_wrapper.sh

RUN chmod +x /opt/ml/code/train_wrapper.sh

ENV PYTHONUNBUFFERED=1 PYTHONPATH="/opt/ml/code"

# Create non-root user (SageMaker best practice)
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
