#!/usr/bin/env python3
"""
monitor_stage0_async.py

Auto-polling monitor for the Stage 0 async SageMaker training job.
Usage:
  # If you launched with launch_stage0_async.py (it writes .last_smoke_test_job)
  python monitor_stage0_async.py

  # Or pass a job name explicitly:
  python monitor_stage0_async.py --job-name mamba-mnq-smoketest-20251008-094358

This script polls every N seconds and prints status/secondary status,
start/end times, model artifact location and a small cost estimate.
It exits with code:
  0 -> Completed
  1 -> Failed
  2 -> Still running (if interrupted)
"""
import argparse
import boto3
import os
import sys
import time
from datetime import datetime, timezone

# Default region - will use env var if present
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-west-2"
POLL_INTERVAL = 30  # seconds

# Simple hourly rates (on-demand) for cost estimate (fallbacks)
HOURLY_RATE = {
    "ml.g4dn.xlarge": 0.526,   # approximate on-demand
    "ml.g4dn.2xlarge": 1.052,
    "ml.g5.2xlarge": 3.06
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", "-j", type=str, help="SageMaker training job name")
    p.add_argument("--region", "-r", type=str, default=DEFAULT_REGION, help="AWS region")
    p.add_argument("--interval", "-i", type=int, default=POLL_INTERVAL, help="Poll interval seconds")
    p.add_argument("--tail-errors", "-t", action="store_true", help="Show last error log lines from CloudWatch on failure")
    return p.parse_args()

def read_last_job():
    try:
        with open(".last_smoke_test_job", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def format_duration_seconds(sec):
    if sec is None:
        return "N/A"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec//60}m {sec%60}s"
    return f"{sec//3600}h {(sec%3600)//60}m"

def estimate_cost(instance_type, billable_seconds):
    if not billable_seconds:
        return "N/A"
    hours = billable_seconds / 3600.0
    rate = HOURLY_RATE.get(instance_type, list(HOURLY_RATE.values())[0])
    return f"${hours * rate:.2f}"

def describe_job(sm_client, job_name):
    try:
        return sm_client.describe_training_job(TrainingJobName=job_name)
    except Exception as e:
        print(f"Error describing job: {e}")
        return None

def print_status(job):
    status = job.get("TrainingJobStatus", "Unknown")
    secondary = job.get("SecondaryStatus", "Unknown")
    name = job.get("TrainingJobName")
    rcfg = job.get("ResourceConfig", {})
    instance_type = rcfg.get("InstanceType", "Unknown")
    start = job.get("TrainingStartTime")
    end = job.get("TrainingEndTime")
    billable = job.get("BillableTimeInSeconds") or job.get("TrainingTimeInSeconds")
    artifacts = job.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    output_path = job.get("OutputDataConfig", {}).get("S3OutputPath")
    print("\n" + "="*72)
    print(f"Job: {name}")
    print(f"Status: {status}  |  Phase: {secondary}")
    print(f"Instance: {instance_type}")
    if start:
        # start is datetime with tz; compute elapsed if in progress
        print("Started:", start.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"))
        if status == "InProgress":
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            print("Running for:", format_duration_seconds(elapsed))
    if end:
        print("Ended:", end.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"))
        if start:
            total = (end - start).total_seconds()
            print("Duration:", format_duration_seconds(total))
    if billable:
        print("Billable Time:", format_duration_seconds(billable), " | Est. cost:", estimate_cost(instance_type, billable))
    if artifacts:
        print("Model Artifacts:", artifacts)
    if output_path:
        print("Output Location (base):", output_path)
    # Failure reason
    if status == "Failed":
        fr = job.get("FailureReason")
        if fr:
            print("\n❌ Failure Reason:")
            print(fr)
    print("="*72 + "\n")

def tail_error_logs(logs_client, job_name, region, lines=30):
    # Try to find log streams matching the job name in /aws/sagemaker/TrainingJobs
    group = "/aws/sagemaker/TrainingJobs"
    try:
        streams_resp = logs_client.describe_log_streams(logGroupName=group, logStreamNamePrefix=job_name, limit=50)
        streams = streams_resp.get("logStreams", [])
        if not streams:
            print("No CloudWatch log streams found for job.")
            return
        # pick the most recent stream
        stream_name = sorted(streams, key=lambda s: s.get("lastEventTimestamp", 0))[-1]["logStreamName"]
        events = logs_client.get_log_events(logGroupName=group, logStreamName=stream_name, limit=lines, startFromHead=False)
        print(f"\n--- Last {lines} log events from {stream_name} ---")
        for e in events.get("events", []):
            ts = datetime.fromtimestamp(e["timestamp"]/1000.0).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] {e['message']}")
        print("--- end logs ---\n")
    except Exception as e:
        print("Could not fetch logs:", e)

def main():
    args = parse_args()
    job_name = args.job_name or read_last_job()
    region = args.region
    interval = args.interval

    if not job_name:
        print("Job name not provided and .last_smoke_test_job not found.")
        print("Usage: python monitor_stage0_async.py --job-name <job-name>")
        sys.exit(1)

    sm = boto3.client("sagemaker", region_name=region)
    logs_client = boto3.client("logs", region_name=region)

    print(f"Monitoring job: {job_name} in region {region}")
    print(f"Polling every {interval}s. Ctrl-C to stop (job will remain running).")

    try:
        while True:
            job = describe_job(sm, job_name)
            if job is None:
                print("Job not found / error while fetching job. Retrying in a bit...")
                time.sleep(interval)
                continue

            status = job.get("TrainingJobStatus", "Unknown")
            print_status(job)

            if status == "Completed":
                print("✅ Training job completed successfully.")
                # optionally summarize artifact keys etc.
                sys.exit(0)
            elif status == "Failed":
                print("❌ Training job FAILED.")
                if args.tail_errors:
                    tail_error_logs(logs_client, job_name, region)
                sys.exit(1)
            elif status in ("Stopped", "Stopping"):
                print(f"⚠️  Training job status: {status}.")
                if args.tail_errors:
                    tail_error_logs(logs_client, job_name, region)
                sys.exit(1)
            else:
                # InProgress, Starting, Downloading, etc.
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting (job continues running).")
        sys.exit(2)

if __name__ == "__main__":
    main()
