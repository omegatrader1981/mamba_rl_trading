#!/usr/bin/env python3
import boto3
import json
import sys
import time
from datetime import datetime

# === CONFIGURATION ===
JOB_TYPE = "smoke_test"
INSTRUMENT = "mnq"
IMAGE_TAG = "refactor-v1"
INSTANCE_TYPE = "ml.g4dn.2xlarge"  # Changed from ml.g5.2xlarge (quota-safe)
ROLE_ARN = "arn:aws:iam::537124950121:role/SageMakerExecutionRole"
REGION = "eu-west-2"

def launch_smoke_test():
    sagemaker = boto3.client("sagemaker", region_name=REGION)

    job_name = f"mamba-{JOB_TYPE}-{INSTRUMENT}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    image_uri = f"537124950121.dkr.ecr.{REGION}.amazonaws.com/mamba-rl-trading:{IMAGE_TAG}"

    print(f"🚀 Launching SageMaker Processing Job: {job_name}")
    print(f"   • Image: {image_uri}")
    print(f"   • Instance: {INSTANCE_TYPE}")

    try:
        response = sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=ROLE_ARN,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": INSTANCE_TYPE,
                    "VolumeSizeInGB": 50,
                }
            },
            AppSpecification={"ImageUri": image_uri},
            Environment={
                "JOB_TYPE": JOB_TYPE,
                "INSTRUMENT": INSTRUMENT,
                "STAGE": "stage0",
            },
            ProcessingOutputConfig={
                "Outputs": [
                    {"OutputName": "results", "S3Output": {
                        "S3Uri": f"s3://mamba-rl-jobs/{job_name}/output/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob"
                    }}
                ]
            },
            Tags=[{"Key": "Project", "Value": "MambaRL"}],
        )

        print("\n✅ Job successfully launched!")
        print(f"🔗 Job name: {job_name}")
        print("⏳ Checking job status after 30 seconds...\n")
        time.sleep(30)

        status = sagemaker.describe_processing_job(ProcessingJobName=job_name)
        print(json.dumps({
            "JobName": job_name,
            "Status": status["ProcessingJobStatus"],
            "CreationTime": str(status["CreationTime"])
        }, indent=2))

    except Exception as e:
        print(f"❌ Error launching job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_smoke_test()
