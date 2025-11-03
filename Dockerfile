FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /opt/ml/code
WORKDIR /opt/ml/code
ENV PYTHONUNBUFFERED=1 PYTHONPATH="/opt/ml/code:${PYTHONPATH}"
USER sagemaker-user
ENTRYPOINT ["python", "src/train.py"]
