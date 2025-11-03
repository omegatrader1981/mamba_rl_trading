# <<< KNOWN-GOOD STABLE VERSION >>>
FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC PYTHONUNBUFFERED=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3 -m pip install --upgrade pip setuptools wheel
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["python3", "src/train.py"]
USER root
