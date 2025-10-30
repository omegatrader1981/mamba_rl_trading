import sagemaker
from sagemaker.pytorch import PyTorch
import logging
import os
import boto3
from datetime import datetime

# Disable telemetry
os.environ["SAGEMAKER_TELEMETRY_OPT_OUT"] = "true"
logging.getLogger("sagemaker.telemetry").setLevel(logging.ERROR)

print("--- SCRIPT STARTED (LAUNCHING DEBUG SNAPSHOT JOB) ---")

# --- Configuration ---
sagemaker_session = sagemaker.Session()
ACCOUNT_ID = sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]
REGION = sagemaker_session.boto_region_name
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/mamba_rl_trading:refactor-v1"
INSTANCE_TYPE = "ml.g4dn.xlarge"
S3_DATA_URI = f"s3://sagemaker-{REGION}-{ACCOUNT_ID}/mamba-rl-trading/data"

# --- Estimator Definition ---
estimator = PyTorch(
    entry_point='debug_entrypoint.py',
    source_dir='src/',
    role=ROLE_ARN,
    image_uri=IMAGE_URI,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    max_run=600, # 10 minutes is plenty
    use_spot_instances=True,
    max_wait=600,
    sagemaker_session=sagemaker_session
)

print("--- ESTIMATOR CREATED. CALLING FIT() NOW... ---")

try:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"debug-snapshot-{timestamp}"
    
    # Add the 'inputs' argument to download data
    estimator.fit(
        inputs={'training': S3_DATA_URI},
        job_name=job_name,
        wait=True, # We wait for this job to finish
        logs=True
    )
    
    print(f"--- DEBUG JOB '{job_name}' COMPLETED SUCCESSFULLY ---")
    
    bucket = sagemaker_session.default_bucket()
    s3_output_path = f"s3://{bucket}/{job_name}/output/data/debug_snapshot/"

    print("\n" + "="*70)
    print("GROUND TRUTH SNAPSHOT IS READY FOR DOWNLOAD")
    print("="*70)
    print("\nRun this command to download the snapshot:")
    print(f"aws s3 sync {s3_output_path} ./debug_snapshot")
    print("\nThen you can inspect the files inside the './debug_snapshot/code/' directory.")
    
except Exception as e:
    print(f"--- DEBUG JOB FAILED WITH AN ERROR ---")
    print(e)

print("--- SCRIPT FINISHED ---")
