# Use official PyTorch 2.4.0 image with CUDA 12.1 (ABI-compatible with public Mamba wheels)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel

USER root

# Install system dependencies (including build tools for causal-conv1d/mamba)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Mamba and causal-conv1d from source (to match CUDA 12.1 in this image)
# They are NOT in requirements.txt to control build order and avoid conflicts
RUN pip install --no-cache-dir causal-conv1d mamba-ssm

# Verify critical dependencies
RUN python -c "\
import torch; print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
import causal_conv1d; print('✅ causal-conv1d: installed'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All critical dependencies verified!')"

# Copy application code
COPY . /opt/ml/code
WORKDIR /opt/ml/code

# Create training wrapper
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

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ml/code"

# Non-root user for SageMaker
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
