# Use the official SageMaker PyTorch image as a base
FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310

USER root

# Install minimal system libraries for Kaleido (no Chrome)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your application code
COPY . /opt/ml/code

# Set the working directory
WORKDIR /opt/ml/code

# Set Python path and unbuffered output for immediate logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

# Switch back to the default SageMaker user
USER sagemaker-user

# THE FIX: Tell SageMaker what command to run
ENTRYPOINT ["python", "src/train.py"]
