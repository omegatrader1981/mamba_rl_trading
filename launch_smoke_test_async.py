#!/usr/bin/env python3
"""
Launch Smoke Test on SageMaker asynchronously using a custom ECR image.

✅ Safe for Ubuntu
✅ Uses ml.g4dn.2xlarge (as requested)
✅ Prints job URL for live monitoring
"""

import boto3
import time
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# --- CONFIGURATION ---
ACCOUNT_ID = "537124950121"
REGION = "eu-west-2"
IMAGE_NAME = "mamba_rl_trading"
IMAGE_TAG = "refactor-v1"
ROLE_ARN = "arn:aws:iam::537124950121:role/AmazonSageMaker-ExecutionRole-20250221T093632"
INSTANCE_TYPE = "ml.g4dn.2xlarge"
INSTANCE_COUNT = 1
INSTRUMENT = "mnq"

ECR_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:{IMAGE_TAG}"
timestamp = time.strftime("%Y%m%d-%H%M%S")
JOB_NAME = f"mamba-smoke-test-{INSTRUMENT}-{timestamp}".replace("_", "-")

# --- Initialize client ---
sm_client = boto3.client("sagemaker", region_name=REGION)

# --- Setup processor ---
processor = ScriptProcessor(
    image_uri=ECR_URI,
    role=ROLE_ARN,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    command=["python3"],
)

print(f"🚀 Launching SageMaker Processing Job: {JOB_NAME}")
print(f"   • Image: {ECR_URI}")
print(f"   • Instance: {INSTANCE_TYPE}")

try:
    processor.run(
        code="launch_smoke_test.py",
        job_name=JOB_NAME,
        inputs=[
            ProcessingInput(source=".", destination="/opt/ml/processing/input/code")
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{ACCOUNT_ID}-sagemaker-smoke-test/output/{JOB_NAME}"
            )
        ],
        wait=False,
        logs=False,
    )
    print(f"\n✅ Job submitted successfully!")
    print(f"🌐 View logs: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/processing-jobs/{JOB_NAME}")

except Exception as e:
    print(f"\n❌ Error launching job: {e}")
