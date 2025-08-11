# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# DEFINITIVE FINAL VERSION: Uses the standard Docker Hub base image to avoid ECR Public auth issues.

# <<< THE PERMANENT FIX IS HERE: Use the standard, public Docker Hub image for Ubuntu >>>
FROM ubuntu:20.04

# --- 1. LEAST FREQUENTLY CHANGED: System Environment & Dependencies ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common curl wget gnupg2 ca-certificates build-essential \
    cmake ninja-build git unzip pkg-config libjpeg-dev libpng-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/python3.10 /usr/bin/python

# --- 2. INFREQUENTLY CHANGED: CUDA Installation ---
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
    cuda-toolkit-11-8 libcudnn8=8.6.0.*-1+cuda11.8 libcudnn8-dev=8.6.0.*-1+cuda11.8 \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# --- 3. OCCASIONALLY CHANGED: Python Dependencies ---
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel packaging
RUN pip install --no-cache-dir sagemaker-training==4.5.0 packaging==23.1
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mamba-ssm==2.0.4 --no-build-isolation --no-cache-dir

# --- 4. MOST FREQUENTLY CHANGED: Granular Source Code Copy ---
COPY src/ ./src/
COPY conf/ ./conf/
COPY launch_mamba_job.py .
COPY launch_configs.yaml .

# --- 5. Final Verification & Entrypoint ---
RUN echo "=== SYSTEM VERIFICATION ===" && \
    python3 --version && nvcc --version && \
    echo "=== PACKAGE VERIFICATION ===" && \
    python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA Support:', torch.version.cuda)" && \
    python3 -c "import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)" && \
    python3 -c "import stable_baselines3; print('stable-baselines3:', stable_baselines3.__version__)" && \
    echo "✅ ALL CRITICAL PACKAGES LOADED SUCCESSFULLY!"

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["python3", "-m", "sagemaker_training.cli.train"]
