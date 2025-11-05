# Official PyTorch 2.4.0 + CUDA 11.8 base image (Python 3.10)
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install your app deps (no torch/mamba/causal)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Mamba + causal-conv1d from PyPI (pre-built wheels will be auto-selected)
RUN pip install --no-cache-dir "causal-conv1d>=1.5.3" "mamba-ssm"

# Verify
RUN python -c "\
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
import causal_conv1d; print('causal-conv1d: OK'); \
import mamba_ssm; print(f'mamba-ssm: {mamba_ssm.__version__}'); \
print('All dependencies verified!')"

COPY . /opt/ml/code
WORKDIR /opt/ml/code

RUN printf '#!/bin/bash\n\
set -e\n\
echo "SAGEMAKER STARTING..."\n\
echo "PyTorch: $(python -c "import torch; print(torch.__version__)")"\n\
echo "CUDA: $(python -c "import torch; print(torch.version.cuda)")"\n\
echo "Mamba: $(python -c "import mamba_ssm; print(mamba_ssm.__version__)")"\n\
exec python src/train.py "$@"\n' > train_wrapper.sh && chmod +x train_wrapper.sh

ENV PYTHONUNBUFFERED=1 PYTHONPATH="/opt/ml/code"
RUN useradd -m -u 1000 sagemaker
USER sagemaker

ENTRYPOINT ["./train_wrapper.sh"]
