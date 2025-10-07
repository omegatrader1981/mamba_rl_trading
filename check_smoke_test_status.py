#!/usr/bin/env python3
import boto3
import time
import json
from datetime import datetime, timezone

REGION = "eu-west-2"
JOB_PREFIX = "mamba-smoke_test"

def get_latest_smoke_job(sagemaker):
    jobs = sagemaker.list_processing_jobs(
        SortBy="CreationTime", SortOrder="Descending", MaxResults=10
    )["ProcessingJobSummaries"]

    for job in jobs:
        if job["ProcessingJobName"].startswith(JOB_PREFIX):
            return job["ProcessingJobName"]
    return None

def poll_smoke_test_status():
    sagemaker = boto3.client("sagemaker", region_name=REGION)

    job_name = get_latest_smoke_job(sagemaker)
    if not job_name:
        print("❌ No recent smoke test jobs found.")
        return

    print(f"🔍 Monitoring job: {job_name}\n")

    while True:
        job_desc = sagemaker.describe_processing_job(ProcessingJobName=job_name)
        status = job_desc["ProcessingJobStatus"]
        last_update = job_desc["LastModifiedTime"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"[{last_update}] Status: {status}")

        if status in ["Completed", "Failed", "Stopped"]:
            print("\n🏁 Final Job Status:")
            print(json.dumps({
                "JobName": job_name,
                "Status": status,
                "StartTime": str(job_desc["CreationTime"]),
                "EndTime": str(job_desc.get("ProcessingEndTime")),
                "S3Output": job_desc.get("ProcessingOutputConfig", {}).get("Outputs", [{}])[0].get("S3Output", {}).get("S3Uri", "N/A")
            }, indent=2))
            break

        print("⏳ Job still running... next check in 60s\n")
        time.sleep(60)

if __name__ == "__main__":
    poll_smoke_test_status()
