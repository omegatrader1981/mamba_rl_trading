"""
check_smoke_test_status.py
-----------------------------------
Quick status checker for the Mamba RL SageMaker smoke test.

This script:
 - Fetches and prints the current job status
 - Displays failure reasons if applicable
"""

import boto3

sm_client = boto3.client("sagemaker", region_name="eu-west-2")

def check_status(job_name: str):
    response = sm_client.describe_processing_job(ProcessingJobName=job_name)
    status = response["ProcessingJobStatus"]
    print(f"🧩 Job Name: {job_name}")
    print(f"📊 Status: {status}")
    if status == "Failed":
        print(f"❌ Failure Reason: {response.get('FailureReason', 'N/A')}")
    elif status == "Completed":
        print("✅ Smoke test completed successfully!")

if __name__ == "__main__":
    job_name = input("Enter your SageMaker smoke test job name: ").strip()
    check_status(job_name)
