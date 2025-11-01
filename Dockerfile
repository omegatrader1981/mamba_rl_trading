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

# Create a shell wrapper script for debugging
RUN printf '#!/bin/bash\n\
set -e\n\
\n\
echo "=========================================="\n\
echo "TRAIN WRAPPER STARTING"\n\
echo "=========================================="\n\
echo "Current directory: $(pwd)"\n\
echo "Current user: $(whoami)"\n\
echo "Python version: $(python --version)"\n\
echo ""\n\
echo "Contents of /opt/ml/code:"\n\
ls -la /opt/ml/code\n\
echo ""\n\
echo "Contents of /opt/ml/code/src:"\n\
ls -la /opt/ml/code/src || echo "ERROR: src/ directory not found!"\n\
echo ""\n\
echo "Checking if src/train.py exists:"\n\
test -f /opt/ml/code/src/train.py && echo "✅ src/train.py EXISTS" || echo "❌ src/train.py MISSING"\n\
echo ""\n\
echo "Python path:"\n\
python -c "import sys; print('"'"'\\n'"'"'.join(sys.path))"\n\
echo ""\n\
echo "=========================================="\n\
echo "ATTEMPTING TO RUN TRAINING SCRIPT"\n\
echo "=========================================="\n\
\n\
exec python src/train.py "$@"\n' > /opt/ml/code/train_wrapper.sh

# Make the wrapper executable
RUN chmod +x /opt/ml/code/train_wrapper.sh

# Set Python path and unbuffered output for immediate logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

# Switch back to the default SageMaker user
USER sagemaker-user

# Use the shell wrapper as entrypoint
ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh
