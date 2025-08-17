# docker/base/Dockerfile
# Custom base image with all Python dependencies pre-installed.

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

LABEL description="Custom base image for Mamba-RL project with all Python dependencies pre-installed"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

COPY requirements-base.txt requirements-project.txt ./

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements-base.txt \
    && pip install --no-cache-dir -r requirements-project.txt \
    && pip install mamba-ssm==2.0.4 --no-build-isolation --no-cache-dir

# Clean up build artifacts
RUN rm -rf /tmp/build

# Set the final working directory for the application
WORKDIR /opt/ml/code
