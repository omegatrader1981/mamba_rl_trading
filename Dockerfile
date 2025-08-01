# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# FINAL PRODUCTION VERSION: Uses a known-good stack with pre-built wheels for stability.
FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
# --- Environment Setup ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && apt-get install -y     wget git build-essential &&     apt-get clean && rm -rf /var/lib/apt/lists/*
# Install Miniconda for robust package management
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh &&     bash miniconda.sh -b -p /opt/conda &&     rm miniconda.sh
# Add conda to the system PATH
ENV PATH="/opt/conda/bin:$PATH"
# Create conda environment
RUN conda create -n mamba_env python=3.10 -y
# Set the shell to use the conda environment for all subsequent RUN commands
SHELL ["conda", "run", "-n", "mamba_env", "/bin/bash", "-c"]
# Install the known-good, stable combination of PyTorch and Mamba
# This uses pre-built wheels and avoids risky from-source compilation.
RUN conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
RUN pip install packaging ninja &&     pip install causal-conv1d==1.4.0 --no-cache-dir &&     pip install mamba-ssm==2.2.2 --no-cache-dir
# --- Application Setup ---
WORKDIR /opt/ml/code
# Copy and install the rest of our Python packages
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
# Copy the project code into the final image
COPY . /opt/ml/code/
# --- Diagnostics and Final Configuration ---
# Run the comprehensive check inside the correct conda environment
RUN echo "--- Docker Build: COMPREHENSIVE Check ..." &&     python -c "import sys; print(f'Py version: {sys.version}');     import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}');     import mamba_ssm; print(f'mamba_ssm: {getattr(mamba_ssm, \"__version__\", \"N/A\")}')"
# --- SageMaker Configuration ---
# Set environment variables for the conda environment to persist at runtime
ENV CONDA_DEFAULT_ENV=mamba_env
ENV CONDA_PREFIX=/opt/conda/envs/mamba_env
ENV PATH="/opt/conda/envs/mamba_env/bin:$PATH"
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=src/train.py
# Use conda run to ensure the environment is activated for the entrypoint
ENTRYPOINT ["conda", "run", "-n", "mamba_env", "python", "-m", "sagemaker_training.cli.train"]
