# Use NVIDIA PyTorch base - proven stable
FROM nvcr.io/nvidia/pytorch:23.08-py3

USER root

# Install minimal system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install base dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Mamba from pre-built wheel + causal-conv1d (combined for efficiency)
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir \
        mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
        causal-conv1d && \
    rm mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Verify critical dependencies are working
RUN python -c "\
import torch; print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
import causal_conv1d; print('✅ causal-conv1d: installed'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All critical dependencies verified!')"

# Copy application code
COPY . /opt/ml/code

# Set working directory
WORKDIR /opt/ml/code

# Create enhanced debug wrapper with CUDA info
RUN printf '#!/bin/bash\n\
set -e\n\
echo "=========================================="\n\
echo "SAGEMAKER CONTAINER STARTING"\n\
echo "=========================================="\n\
echo "Directory: $(pwd)"\n\
echo "User: $(whoami)"\n\
echo "Python: $(python --version)"\n\
echo "PyTorch: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
echo "CUDA version: $(python -c \"import torch; print(torch.version.cuda)\")"\n\
echo "Mamba: $(python -c \"import mamba_ssm; print(mamba_ssm.__version__)\" 2>/dev/null || echo \"ERROR\")"\n\
echo ""\n\
echo "Files check:"\n\
ls -la /opt/ml/code/src/train.py && echo "✅ train.py exists" || echo "❌ train.py MISSING"\n\
echo ""\n\
echo "=========================================="\n\
echo "STARTING TRAINING"\n\
echo "=========================================="\n\
exec python src/train.py "$@"\n' > /opt/ml/code/train_wrapper.sh

RUN chmod +x /opt/ml/code/train_wrapper.sh

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

# Create non-root user for SageMaker
RUN useradd -m -u 1000 sagemaker
USER sagemaker

# Entrypoint
ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
