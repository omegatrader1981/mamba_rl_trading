"""
launch_smoke_test_async.py
-----------------------------------
Ubuntu-safe launcher for the Mamba RL SageMaker smoke test.
Submits the Stage 0 training job asynchronously and exits immediately.

This script:
 - Submits the smoke test job to SageMaker
 - Prints the SageMaker console URL for monitoring
 - Exits without hanging (no local blocking)
"""

import time
import boto3
import sagemaker

sm_client = boto3.client("sagemaker", region_name="eu-west-2")
job_name = f"smoke-test-{int(time.time())}"

print(f"🚀 Launching SageMaker Smoke Test job: {job_name}")

response = sm_client.create_processing_job(
    ProcessingJobName=job_name,
    RoleArn="arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20240422T194915",
    AppSpecification={
        "ImageUri": "537124950121.dkr.ecr.eu-west-2.amazonaws.com/mamba-rl-trading:refactor-v1"
    },
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.g5.2xlarge",
            "VolumeSizeInGB": 50
        }
    },
    StoppingCondition={"MaxRuntimeInSeconds": 3600},
    Tags=[{"Key": "Project", "Value": "MambaRL-SmokeTest"}],
)

print(f"✅ Smoke test job submitted successfully: {job_name}")
print(f"🔗 View progress here:")
print(f"https://eu-west-2.console.aws.amazon.com/sagemaker/home?region=eu-west-2#/processing-jobs/{job_name}")
