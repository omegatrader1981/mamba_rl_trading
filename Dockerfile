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
RUN cat > /opt/ml/code/train_wrapper.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "TRAIN WRAPPER STARTING"
echo "=========================================="
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Python version: $(python --version)"
echo ""
echo "Contents of /opt/ml/code:"
ls -la /opt/ml/code
echo ""
echo "Contents of /opt/ml/code/src:"
ls -la /opt/ml/code/src || echo "ERROR: src/ directory not found!"
echo ""
echo "Checking if src/train.py exists:"
test -f /opt/ml/code/src/train.py && echo "✅ src/train.py EXISTS" || echo "❌ src/train.py MISSING"
echo ""
echo "Python path:"
python -c "import sys; print('\n'.join(sys.path))"
echo ""
echo "=========================================="
echo "ATTEMPTING TO RUN TRAINING SCRIPT"
echo "=========================================="

# Run the actual training script
exec python src/train.py "$@"
EOF

# Make the wrapper executable
RUN chmod +x /opt/ml/code/train_wrapper.sh

# Set Python path and unbuffered output for immediate logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

# Switch back to the default SageMaker user
USER sagemaker-user

# Use the shell wrapper as entrypoint
ENTRYPOINT ["/bin/bash", "/opt/ml/code/train_wrapper.sh"]
