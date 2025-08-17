# Dockerfile (final app image)
# This is now incredibly fast as it only copies code on top of our pre-built base image.

FROM 537124950121.dkr.ecr.eu-west-2.amazonaws.com/mamba-rl-trading-base:latest

ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=src/train.py

WORKDIR /opt/ml/code

# Copy source code (this is the only layer that changes frequently)
COPY src/ ./src/
COPY conf/ ./conf/
COPY launch_mamba_job.py .
COPY launch_configs.yaml .

ENTRYPOINT ["python3", "-m", "sagemaker_training.cli.train"]
