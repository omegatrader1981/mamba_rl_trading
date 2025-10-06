#!/usr/bin/env python3
"""
Launch Preflight Check on SageMaker using your custom ECR image.

This script:
1. Launches a SageMaker Processing job with your container
2. Mounts your repo code
3. Runs launch_preflight_check.py inside the container
4. Streams logs to console
5. Prints PASS/FAIL summary
"""

import boto3
import time
import sys
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

# --- CONFIGURATION ---
ACCOUNT_ID = "537124950121"  # Your AWS account
REGION = "eu-west-2"         # Change if needed
IMAGE_NAME = "mamba_rl_trading"
IMAGE_TAG = "refactor-v1"
ROLE_ARN = "arn:aws:iam::537124950121:role/AmazonSageMaker-ExecutionRole-20250221T093632"  # Update if needed
INSTANCE_TYPE = "ml.m5.large"
INSTANCE_COUNT = 1

ECR_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:{IMAGE_TAG}"
JOB_NAME = f"preflight-check-{int(time.time())}"

# --- Initialize SageMaker client ---
sm_client = boto3.client("sagemaker", region_name=REGION)

# --- Use SageMaker ScriptProcessor ---
processor = ScriptProcessor(
    image_uri=ECR_URI,
    role=ROLE_ARN,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    command=["python3"]
)

print(f"Launching Preflight Check job: {JOB_NAME}")
processor.run(
    code="launch_preflight_check.py",
    job_name=JOB_NAME,
    inputs=[
        ProcessingInput(
            source=".",  # Mount your local repo directory
            destination="/opt/ml/processing/input/code"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{ACCOUNT_ID}-sagemaker-preflight/output/{JOB_NAME}"
        )
    ],
    wait=True,  # Wait until job finishes
    logs=True   # Stream logs
)

# --- Check job status ---
resp = sm_client.describe_processing_job(ProcessingJobName=JOB_NAME)
status = resp["ProcessingJobStatus"]
if status == "Completed":
    print(f"\n✅ Preflight Check PASSED: {JOB_NAME}")
    sys.exit(0)
else:
    print(f"\n❌ Preflight Check FAILED: {JOB_NAME} (Status: {status})")
    if "FailureReason" in resp:
        print("Failure Reason:", resp["FailureReason"])
    sys.exit(1)
