# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# DEFINITIVE WHEEL-BASED VERSION: Uses stable, community-vetted pre-built wheels

# Use Python base instead of PyTorch base to avoid version conflicts
FROM python:3.10-slim

# --- Environment Setup ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    unzip \
    lsb-release \
    ca-certificates \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python Environment Setup ---
WORKDIR /opt/ml/code

# Upgrade pip and install base packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir sagemaker-training packaging

# --- Install PyTorch 2.0.1 + CUDA 11.8 from correct index ---
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other ML/RL dependencies (copy requirements first for better caching)
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Install mamba-ssm with --no-build-isolation ---
RUN pip install mamba-ssm==2.0.4 --no-build-isolation

# Copy the project code AFTER all dependencies are installed
COPY . /opt/ml/code/

# --- Verification ---
RUN echo "--- Verifying wheel-based installation ---" && \
    python -c "import sys; print(f'Python: {sys.version}'); \
    import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
    import mamba_ssm; print(f'Mamba SSM: {mamba_ssm.__version__}'); \
    print('✅ All components loaded successfully!')"

# --- SageMaker Configuration ---
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["python", "-m", "sagemaker_training.cli.train"]
