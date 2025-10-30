FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310
USER root
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /opt/ml/code
WORKDIR /opt/ml/code
USER sagemaker-user
