#!/usr/bin/env python3
"""
Stage 0 Smoke Test - Async Launcher using SageMaker SDK
Submits a SageMaker Training Job asynchronously with full Spot support.
Uses PyTorch estimator for reliable parameter handling.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# --- CONFIGURATION ---
ACCOUNT_ID = "537124950121"
REGION = "eu-west-2"
IMAGE_NAME = "mamba_rl_trading"
IMAGE_TAG = "refactor-v1"
ROLE_ARN = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"

INSTANCE_TYPE = "ml.g4dn.xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 30

BUCKET_NAME = f"sagemaker-{REGION}-{ACCOUNT_ID}"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
JOB_NAME = f"mamba-mnq-smoketest-{timestamp}".replace("_", "-")

log.info("=" * 70)
log.info("STAGE 0: ASYNC TRAINING JOB LAUNCHER (SageMaker SDK)")
log.info("=" * 70)
log.info(f"\nJob Name: {JOB_NAME}")
log.info(f"Instance: {INSTANCE_TYPE} (Spot)")
log.info(f"Region: {REGION}")
log.info("")

# Initialize SageMaker session
sess = sagemaker.Session()
s3_client = boto3.client("s3", region_name=REGION)

# Ensure S3 bucket exists
try:
    s3_client.head_bucket(Bucket=BUCKET_NAME)
    log.info(f"✅ S3 bucket exists: {BUCKET_NAME}")
except Exception:
    log.info(f"Creating S3 bucket: {BUCKET_NAME}")
    s3_client.create_bucket(
        Bucket=BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": REGION},
    )

# Upload data (if available)
log.info("\nUploading data files to S3 (if any)...")
data_prefix = "mamba-rl-trading/data"
data_dir = Path("data")
if data_dir.exists():
    for csv_file in data_dir.glob("*.csv"):
        s3_key = f"{data_prefix}/{csv_file.name}"
        log.info(f"  Uploading {csv_file.name}...")
        s3_client.upload_file(str(csv_file), BUCKET_NAME, s3_key)
    log.info("✅ Data upload complete")
else:
    log.info("⚠️  Warning: No data directory found (skipping upload)")

data_input_uri = f"s3://{BUCKET_NAME}/{data_prefix}"

# Create PyTorch estimator (uses SageMaker SDK — full Spot support)
estimator = PyTorch(
    entry_point='src/train.py',
    source_dir='.',
    role=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    volume_size=VOLUME_SIZE_GB,
    framework_version='2.0.0',
    py_version='py310',
    max_run=3600,          # 1 hour
    max_wait=7200,         # 2 hours (required for Spot)
    use_spot_instances=True,
    hyperparameters={
        'experiment': 'smoke_test',
        'instrument': 'mnq',
    },
    environment={
        'HYDRA_FULL_ERROR': '1',
        'MAMBA_FORCE_BUILD': '1',
    },
    output_path=f"s3://{BUCKET_NAME}/mamba-rl-trading/output",
    checkpoint_s3_uri=f"s3://{BUCKET_NAME}/mamba-rl-trading/checkpoints/smoke-test",
    base_job_name="mamba-mnq-smoketest",
    tags=[
        {"Key": "Project", "Value": "MambaRLTrading"},
        {"Key": "Stage", "Value": "Stage0-Baseline"},
        {"Key": "Type", "Value": "SmokeTest"},
    ],
)

log.info("\nSubmitting training job to SageMaker (non-blocking)...\n")

try:
    estimator.fit(
        inputs={'training': data_input_uri},
        job_name=JOB_NAME,
        wait=False
    )
    log.info("=" * 70)
    log.info("✅ JOB SUBMITTED SUCCESSFULLY")
    log.info("=" * 70)
    log.info(f"\nJob Name: {JOB_NAME}")
    log.info(f"\nMonitor at:")
    log.info(f"  AWS Console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{JOB_NAME}")
    log.info(f"\nCheck status:")
    log.info(f"  aws sagemaker describe-training-job --training-job-name {JOB_NAME} --region {REGION}")
    log.info(f"\nView logs:")
    log.info(f"  aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {JOB_NAME} --region {REGION}")
    log.info(f"\nValidate results (after completion):")
    log.info(f"  python validate_smoke_test.py {JOB_NAME}\n")
    log.info("Expected runtime: 15–30 min | Est. cost: ~$0.20 (Spot)\n")

    with open(".last_smoke_test_job", "w") as f:
        f.write(JOB_NAME)
    log.info(f"💾 Saved job name to .last_smoke_test_job")
    log.info("=" * 70)

except Exception as e:
    log.info("=" * 70)
    log.info("❌ JOB SUBMISSION FAILED")
    log.info("=" * 70)
    log.info(f"\nError: {e}\n")
    log.info("🔍 Troubleshooting:")
    log.info(f"1. Check ECR image: aws ecr describe-images --repository-name {IMAGE_NAME} --region {REGION}")
    log.info(f"2. Check IAM role ARN includes /service-role/")
    log.info(f"3. Confirm Spot quota for ml.g4dn.xlarge is approved")
    exit(1)
