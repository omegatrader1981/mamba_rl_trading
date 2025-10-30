import sagemaker
from sagemaker.pytorch import PyTorch
import logging
import os
import boto3
from datetime import datetime
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, help="S3 URI for the source.tar.gz file")
args, _ = parser.parse_known_args()

# Disable telemetry
os.environ["SAGEMAKER_TELEMETRY_OPT_OUT"] = "true"
logging.getLogger("sagemaker.telemetry").setLevel(logging.ERROR)

print("--- SCRIPT STARTED (LAUNCHING FINAL TEST) ---")

# --- Configuration ---
sagemaker_session = sagemaker.Session()
ACCOUNT_ID = sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]
REGION = sagemaker_session.boto_region_name
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
# 🔻 --- Use the NEW image tag --- 🔻
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/mamba_rl_trading:refactor-v2"
INSTANCE_TYPE = "ml.g5.xlarge"
S3_DATA_URI = f"s3://sagemaker-{REGION}-{ACCOUNT_ID}/mamba-rl-trading/data"

# --- Use the provided S3 source URI ---
S3_SOURCE_URI = args.source
print(f"Using Image: {IMAGE_URI}")
print(f"Using S3 source code: {S3_SOURCE_URI}")

# --- Estimator Definition ---
estimator = PyTorch(
    entry_point='train.py',
    source_dir=S3_SOURCE_URI,
    role=ROLE_ARN,
    image_uri=IMAGE_URI,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    max_run=3600,
    use_spot_instances=True,
    max_wait=3600,
    hyperparameters={'experiment': 'smoke_test', 'instrument': 'mnq'},
    sagemaker_session=sagemaker_session
)

print("--- ESTIMATOR CREATED. CALLING FIT() NOW... ---")

try:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"final-smoke-test-{timestamp}"
    
    estimator.fit(
        inputs={'training': S3_DATA_URI},
        wait=False,
        job_name=job_name
    )
    print("--- FIT() CALL COMPLETED SUCCESSFULLY ---")
    print(f"\n✅ Job '{job_name}' submitted! Check the SageMaker console.")
except Exception as e:
    print(f"--- FIT() CALL FAILED WITH AN ERROR ---")
    print(e)

print("--- SCRIPT FINISHED ---")
