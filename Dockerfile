# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# FINAL PRODUCTION VERSION: Installs mamba-ssm from source for full compatibility.

# Use the PyTorch 2.3 / CUDA 12.1 platform
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# --- Environment Setup ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install system dependencies, including git for cloning.
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget unzip lsb-release git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python Environment Setup ---
WORKDIR /opt/ml/code

# This multi-step process is optimized for Docker layer caching.
# 1. Install all dependencies from requirements.txt first.
COPY requirements.txt /opt/ml/code/requirements.txt
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/conda/bin/python -m pip install --no-cache-dir sagemaker-training packaging && \
    echo "Starting pip install from requirements.txt..." && \
    /opt/conda/bin/python -m pip install --no-cache-dir --timeout=600 \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r requirements.txt && \
    echo "Finished pip install from requirements.txt."

# 2. Clone and install mamba-ssm and causal-conv1d from source.
RUN git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    pip install . && \
    pip install causal-conv1d>=1.4.1 && \
    cd .. && rm -rf mamba

# Copy the entire project context AFTER all dependencies are installed.
COPY . /opt/ml/code/

# --- Diagnostics and Final Configuration ---
# This comprehensive check verifies that all critical components are installed and compatible.
RUN echo "--- Docker Build: Verifying final directory structure in /opt/ml/code/ ---" && \
    ls -R /opt/ml/code && \
    echo "--- Docker Build: COMPREHENSIVE Check ..." && \
    /opt/conda/bin/python -c "import sys; print(f'Py version: {sys.version}'); \
    import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}'); \
    import transformers; print(f'transformers: {transformers.__version__}'); \
    import safetensors; print(f'safetensors: {safetensors.__version__}'); \
    import mamba_ssm; print(f'mamba_ssm: {getattr(mamba_ssm, \"__version__\", \"OK - Installed from source\")}')"

# --- SageMaker Configuration ---
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["/opt/conda/bin/python", "-m", "sagemaker_training.cli.train"]
