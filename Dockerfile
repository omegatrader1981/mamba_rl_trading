# Dockerfile for Project Phoenix
# DEFINITIVE FINAL VERSION: Based on a timestamped, immutable AWS DLC image.

# <<< THE FIX IS HERE: Using a timestamped, immutable DLC tag for maximum reproducibility >>>
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC

# Install minor system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Copy and install our harmonized, project-specific Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install mamba-ssm separately with its required build flags
RUN pip install mamba-ssm==2.0.4 --no-build-isolation --no-cache-dir

# Copy source code
COPY src/ ./src/
COPY conf/ ./conf/
COPY launch_mamba_job.py .
COPY launch_configs.yaml .

# Set SageMaker environment variables
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["python3", "-m", "sagemaker_training.cli.train"]
