# Dockerfile for Futures RL Trading Strategy (mamba_rl_trading)
# FINAL PRODUCTION VERSION: Bypasses pip for a direct, robust mamba compile.

# Use the PyTorch 2.3 / CUDA 12.1 platform
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# --- Environment Setup ---
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC
ENV LANG=C-UTF-8
ENV LC_ALL=C-UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:/home/chelsea/mamba_rl_trading/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/c/Program Files/WindowsApps/MicrosoftCorporationII.WindowsSubsystemForLinux_2.4.13.0_x64__8wekyb3d8bbwe:/mnt/c/WINDOWS/system32:/mnt/c/WINDOWS:/mnt/c/WINDOWS/System32/Wbem:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/:/mnt/c/Program Files (x86)/ATI Technologies/ATI.ACE/Core-Static:/mnt/c/WINDOWS/System32/OpenSSH/:/mnt/c/Program Files/dotnet/:/mnt/c/Program Files/Microsoft VS Code/bin:/mnt/c/Users/chelsea/AppData/Local/Programs/Python/Python313/Scripts/:/mnt/c/Users/chelsea/AppData/Local/Programs/Python/Python313/:/mnt/c/Users/chelsea/AppData/Local/Programs/Python/Launcher/:/mnt/c/Users/chelsea/AppData/Local/Microsoft/WindowsApps:/mnt/c/Users/chelsea/AppData/Local/Programs/Git/cmd:/snap/bin"

# Add the essential C++ build tools.
RUN apt-get update &&     apt-get install -y --no-install-recommends wget unzip lsb-release git build-essential ninja-build cmake &&     apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python Environment Setup ---
WORKDIR /opt/ml/code

# STAGE 1: Install dependencies from requirements.txt.
COPY requirements.txt /opt/ml/code/requirements.txt
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel &&     /opt/conda/bin/python -m pip install --no-cache-dir sagemaker-training packaging &&     echo "Starting pip install from requirements.txt..." &&     /opt/conda/bin/python -m pip install --no-cache-dir --timeout=600         --extra-index-url https://download.pytorch.org/whl/cu121         -r requirements.txt &&     echo "Finished pip install from requirements.txt."

# <<< THE FIX IS HERE: A direct build command that bypasses pip's failing backend. >>>
# This is the most robust and direct way to compile the C++ extensions.
RUN git clone https://github.com/state-spaces/mamba.git &&     cd mamba &&     /opt/conda/bin/python setup.py install &&     /opt/conda/bin/python -m pip install causal-conv1d>=1.4.1 &&     cd .. && rm -rf mamba

# STAGE 2: Copy the project code into the final image.
COPY . /opt/ml/code/

# --- Diagnostics and Final Configuration ---
# (Diagnostics are preserved)
RUN echo "--- Docker Build: Verifying final directory structure in /opt/ml/code/ ---" &&     ls -R /opt/ml/code &&     echo "--- Docker Build: COMPREHENSIVE Check ..." &&     /opt/conda/bin/python -c "import sys; print(f'Py version: {sys.version}');     import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}');     import mamba_ssm; print(f'mamba_ssm: {getattr(mamba_ssm, \"__version__\", \"N/A\")}')"

# --- SageMaker Configuration ---
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM src/train.py
ENTRYPOINT ["/opt/conda/bin/python", "-m", "sagemaker_training.cli.train"]
