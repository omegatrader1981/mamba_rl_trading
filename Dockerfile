# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# PROVEN WORKING VERSION: PyTorch 2.0.1 + mamba-ssm 2.0.4 with pre-built wheels

# Use Ubuntu 20.04 base with Python 3.10 - ECR Public to avoid rate limits
FROM public.ecr.aws/ubuntu/ubuntu:20.04

# --- Critical Environment Setup (prevents many common errors) ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# --- System Dependencies (addresses common build failures) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    gnupg2 \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    git \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Remove system pip to avoid conflicts, then install fresh pip
RUN apt-get purge -y python3-pip || true && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Create python symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# --- CUDA 11.8 Installation (modern CUDA for PyTorch 2.0.1) ---
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
    cuda-toolkit-11-8 \
    libcudnn8=8.6.0.*-1+cuda11.8 \
    libcudnn8-dev=8.6.0.*-1+cuda11.8 \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Python Environment Setup ---
WORKDIR /opt/ml/code

# Upgrade pip and essential tools (prevents wheel building issues)
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Install SageMaker dependencies first
RUN pip install --no-cache-dir \
    sagemaker-training==4.5.0 \
    packaging==23.1

# --- PROVEN PyTorch Installation (2.0.1 + CUDA 11.8) ---
# Install community-tested PyTorch 2.0.1 with CUDA 11.8 support
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation before proceeding (CUDA won't be available during build)
RUN python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} with CUDA support: {torch.version.cuda}')"

# --- Install Other Dependencies (copy requirements for better caching) ---
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- PROVEN MAMBA-SSM INSTALLATION (community-tested combination) ---
# Install mamba-ssm 2.0.4 which is known to work with PyTorch 2.0.1
RUN pip install mamba-ssm==2.0.4 --no-build-isolation --no-cache-dir --verbose

# --- Copy Project Code ---
COPY . /opt/ml/code/

# --- Comprehensive Verification (catches errors early) ---
RUN echo "=== SYSTEM VERIFICATION ===" && \
    python3 --version && \
    pip --version && \
    nvcc --version && \
    echo "=== PACKAGE VERIFICATION ===" && \
    python3 -c "import sys; print('Python:', sys.version)" && \
    python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA Support:', torch.version.cuda)" && \
    python3 -c "import torchvision; print('TorchVision:', torchvision.__version__)" && \
    python3 -c "import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)" && \
    python3 -c "import stable_baselines3; print('stable-baselines3:', stable_baselines3.__version__)" && \
    echo "✅ ALL CRITICAL PACKAGES LOADED SUCCESSFULLY!"

# Final directory listing for debugging
RUN echo "=== FINAL PROJECT STRUCTURE ===" && find /opt/ml/code -type f -name "*.py" | head -10

# --- SageMaker Configuration ---
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["python3", "-m", "sagemaker_training.cli.train"]
