# Use NVIDIA PyTorch base - CUDA 12.5, PyTorch 2.4.1
FROM nvcr.io/nvidia/pytorch:24.09-py3

USER root

# Install minimal system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install base dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install causal-conv1d first
RUN pip install --no-cache-dir "causal-conv1d>=1.5.0"

# 🔑 KEY FIX: Use cu11 wheel (only one that exists) + proper filename + URL encoding
RUN curl -fL "https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3%2Bcu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    -o "/tmp/mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    pip install --no-cache-dir "/tmp/mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" && \
    rm "/tmp/mamba_ssm-2.2.6.post3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Verify dependencies
RUN python -c "\
import torch; print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
print(f'✅ ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}'); \
import causal_conv1d; print('✅ causal-conv1d: OK'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All dependencies verified!')"

# Copy application code
COPY . /opt/ml/code

WORKDIR /opt/ml/code

# Create debug wrapper
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
echo "Mamba: $(python -c \"import mamba_ssm; print(mamba_ssm.__version__)\" 2>/dev/null || echo \"ERROR\")"\n\
echo ""\n\
ls -la /opt/ml/code/src/train.py && echo "✅ train.py exists" || echo "❌ train.py MISSING"\n\
echo ""\n\
echo "=========================================="\n\
echo "STARTING TRAINING"\n\
echo "=========================================="\n\
exec python src/train.py "$@"\n' > /opt/ml/code/train_wrapper.sh

RUN chmod +x /opt/ml/code/train_wrapper.sh

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ml/code"

# Create non-root user
RUN useradd -m -u 1000 sagemaker
USER sagemaker

# Entrypoint
ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
