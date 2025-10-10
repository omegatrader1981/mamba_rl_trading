#!/usr/bin/env python3
"""
Stage 0 Smoke Test - Async Launcher for Ubuntu
Submits a SageMaker Training Job asynchronously (non-blocking).
"""

import boto3
import time
from datetime import datetime
from pathlib import Path
import os

# --- CONFIGURATION ---
ACCOUNT_ID = "537124950121"
REGION = "eu-west-2"
IMAGE_NAME = "mamba_rl_trading"
IMAGE_TAG = "refactor-v1"
ROLE_ARN = "arn:aws:iam::537124950121:role/AmazonSageMaker-ExecutionRole-20250221T093632"

# Use g4dn.xlarge instance (you have quota for this)
INSTANCE_TYPE = "ml.g4dn.xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 30

ECR_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:{IMAGE_TAG}"
BUCKET_NAME = f"sagemaker-{REGION}-{ACCOUNT_ID}"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
JOB_NAME = f"mamba-mnq-smoketest-{timestamp}".replace("_", "-")

print("=" * 70)
print("STAGE 0: ASYNC TRAINING JOB LAUNCHER")
print("=" * 70)
print(f"\nJob Name: {JOB_NAME}")
print(f"Instance: {INSTANCE_TYPE}")
print(f"Region: {REGION}")
print(f"Image:  {ECR_URI}\n")

sm_client = boto3.client("sagemaker", region_name=REGION)
s3_client = boto3.client("s3", region_name=REGION)

# --- Ensure S3 bucket exists ---
try:
    s3_client.head_bucket(Bucket=BUCKET_NAME)
    print(f"✅ S3 bucket exists: {BUCKET_NAME}")
except Exception:
    print(f"Creating S3 bucket: {BUCKET_NAME}")
    s3_client.create_bucket(
        Bucket=BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": REGION},
    )

# --- Upload data (if available) ---
print("\nUploading data files to S3 (if any)...")
data_prefix = "mamba-rl-trading/data"
data_dir = Path("data")
if data_dir.exists():
    for csv_file in data_dir.glob("*.csv"):
        s3_key = f"{data_prefix}/{csv_file.name}"
        print(f"  Uploading {csv_file.name}...")
        s3_client.upload_file(str(csv_file), BUCKET_NAME, s3_key)
    print("✅ Data upload complete")
else:
    print("⚠️  Warning: No data directory found (skipping upload)")

data_input_uri = f"s3://{BUCKET_NAME}/{data_prefix}"
output_uri = f"s3://{BUCKET_NAME}/mamba-rl-trading/output"
checkpoint_uri = f"s3://{BUCKET_NAME}/mamba-rl-trading/checkpoints/smoke-test"

# --- Define Training Job ---
training_job_config = {
    "TrainingJobName": JOB_NAME,
    "RoleArn": ROLE_ARN,
    "AlgorithmSpecification": {
        "TrainingImage": ECR_URI,
        "TrainingInputMode": "File",
    },
    "InputDataConfig": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": data_input_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
            "CompressionType": "None",
        }
    ],
    "OutputDataConfig": {"S3OutputPath": output_uri},
    "ResourceConfig": {
        "InstanceType": INSTANCE_TYPE,
        "InstanceCount": INSTANCE_COUNT,
        "VolumeSizeInGB": VOLUME_SIZE_GB,
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 3600,  # 1 hour — sufficient for smoke test
        # 🔻 --- FIX --- 🔻
        # MaxWaitTimeInSeconds is REQUIRED for spot training and must be >= MaxRuntimeInSeconds
        "MaxWaitTimeInSeconds": 3600
    },
    "HyperParameters": {
        "experiment": "smoke_test",
        "instrument": "mnq",
    },
    "Environment": {
        "HYDRA_FULL_ERROR": "1",
        "MAMBA_FORCE_BUILD": "1",
    },
    # 🔺 --- FIX was in StoppingCondition above --- 🔺
    "EnableManagedSpotTraining": True,  # ✅ Enables Spot; MaxWaitTimeInSeconds is required
    "CheckpointConfig": {
        "S3Uri": checkpoint_uri,
        "LocalPath": "/opt/ml/checkpoints",
    },
    "Tags": [
        {"Key": "Project", "Value": "MambaRLTrading"},
        {"Key": "Stage", "Value": "Stage0-Baseline"},
        {"Key": "Type", "Value": "SmokeTest"},
    ],
}

print("\nSubmitting training job to SageMaker (non-blocking)...\n")

try:
    response = sm_client.create_training_job(**training_job_config)
    print("=" * 70)
    print("✅ JOB SUBMITTED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nJob Name: {JOB_NAME}")
    print(f"\nMonitor at:")
    print(f"  AWS Console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{JOB_NAME}")
    print(f"\nCheck status:")
    print(f"  aws sagemaker describe-training-job --training-job-name {JOB_NAME} --region {REGION}")
    print(f"\nView logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {JOB_NAME} --region {REGION}")
    print(f"\nValidate results (after completion):")
    print(f"  python validate_smoke_test.py {JOB_NAME}\n")
    print("Expected runtime: 15–30 min | Est. cost: ~$0.20 (spot)\n")

    with open(".last_smoke_test_job", "w") as f:
        f.write(JOB_NAME)
    print(f"💾 Saved job name to .last_smoke_test_job")
    print("=" * 70)

except Exception as e:
    print("=" * 70)
    print("❌ JOB SUBMISSION FAILED")
    print("=" * 70)
    print(f"\nError: {e}\n")
    print("🔍 Troubleshooting:")
    print(f"1. Check ECR image: aws ecr describe-images --repository-name {IMAGE_NAME} --region {REGION}")
    print(f"2. Check IAM role: aws iam get-role --role-name AmazonSageMaker-ExecutionRole-20250221T093632")
    print(f"3. Check credentials: aws sts get-caller-identity")
    exit(1)
