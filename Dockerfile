# Use NVIDIA PyTorch base - CUDA 12.5, PyTorch 2.4.1
FROM nvcr.io/nvidia/pytorch:24.09-py3

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

# Install causal-conv1d first (required by mamba-ssm)
RUN pip install --no-cache-dir "causal-conv1d>=1.5.0"

# Install Mamba from correct pre-built wheel (CUDA 12, PyTorch 2.4, old ABI)
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -O /tmp/mamba_ssm.whl && \
    pip install --no-cache-dir /tmp/mamba_ssm.whl && \
    rm /tmp/mamba_ssm.whl

# Verify dependencies (including ABI!)
RUN python -c "\
import torch; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
print(f'✅ CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}'); \
import causal_conv1d; print('✅ causal-conv1d: OK'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All dependencies verified!')"

# Copy application code
COPY . /opt/ml/code

# Set working directory
WORKDIR /opt/ml/code

# Create training wrapper for SageMaker
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

# Create non-root user (SageMaker requirement)
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
EOFcat > Dockerfile << 'EOF'
# Use NVIDIA PyTorch base - CUDA 12.5, PyTorch 2.4.1
FROM nvcr.io/nvidia/pytorch:24.09-py3

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

# Install causal-conv1d first (required by mamba-ssm)
RUN pip install --no-cache-dir "causal-conv1d>=1.5.0"

# Install Mamba from correct pre-built wheel (CUDA 12, PyTorch 2.4, old ABI)
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -O /tmp/mamba_ssm.whl && \
    pip install --no-cache-dir /tmp/mamba_ssm.whl && \
    rm /tmp/mamba_ssm.whl

# Verify dependencies (including ABI!)
RUN python -c "\
import torch; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
print(f'✅ CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}'); \
import causal_conv1d; print('✅ causal-conv1d: OK'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All dependencies verified!')"

# Copy application code
COPY . /opt/ml/code

# Set working directory
WORKDIR /opt/ml/code

# Create training wrapper for SageMaker
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

# Create non-root user (SageMaker requirement)
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
EOFcat > Dockerfile << 'EOF'
# Use NVIDIA PyTorch base - CUDA 12.5, PyTorch 2.4.1
FROM nvcr.io/nvidia/pytorch:24.09-py3

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

# Install causal-conv1d first (required by mamba-ssm)
RUN pip install --no-cache-dir "causal-conv1d>=1.5.0"

# Install Mamba from correct pre-built wheel (CUDA 12, PyTorch 2.4, old ABI)
RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -O /tmp/mamba_ssm.whl && \
    pip install --no-cache-dir /tmp/mamba_ssm.whl && \
    rm /tmp/mamba_ssm.whl

# Verify dependencies (including ABI!)
RUN python -c "\
import torch; \
print(f'✅ PyTorch: {torch.__version__}'); \
print(f'✅ CUDA: {torch.version.cuda}'); \
print(f'✅ CUDA available: {torch.cuda.is_available()}'); \
print(f'✅ CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}'); \
import causal_conv1d; print('✅ causal-conv1d: OK'); \
import mamba_ssm; print(f'✅ Mamba: {mamba_ssm.__version__}'); \
print('All dependencies verified!')"

# Copy application code
COPY . /opt/ml/code

# Set working directory
WORKDIR /opt/ml/code

# Create training wrapper for SageMaker
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

# Create non-root user (SageMaker requirement)
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
