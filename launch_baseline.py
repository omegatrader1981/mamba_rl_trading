# launch_baseline.py
# Launches the run_baseline_test.py script as a SageMaker job.

import sagemaker
from sagemaker.estimator import Estimator
import time
import traceback
import yaml
import argparse
import sys
import boto3

def main():
    parser = argparse.ArgumentParser(description="Launch the baseline test as a SageMaker job.")
    parser.add_argument("--instrument", type=str, required=True, default="mnq")
    parser.add_argument("--image-tag", type=str, default="latest", help="The ECR image tag to use.")
    args = parser.parse_args()

    job_type = "baseline_test" # Hardcode the job type for this script

    with open("launch_configs.yaml", 'r') as f:
        launch_configs = yaml.safe_load(f)

    job_config = launch_configs.get(job_type)
    if not job_config:
        print(f"❌ ERROR: Job type '{job_type}' not found in launch_configs.yaml. Exiting.")
        sys.exit(1)
    
    region = job_config.get("region", "eu-west-2")
    print(f"--- CONFIGURING FOR JOB [{job_type}] IN REGION [{region}] ---")

    # --- AWS & SageMaker Configuration ---
    account_id = "537124950121"
    image_name = "mamba-rl-trading" # Your container that has all dependencies
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{args.image_tag}"
    role_arn = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
    
    s3_bucket_name = f"mambabot-{region}"
    s3_output_base_path = f"s3://{s3_bucket_name}/mamba-rl-trading/output/"
    s3_input_data_uri = f"s3://{s3_bucket_name}/databento_mnq_downloads/databento_mnq_downloads_5min_2020_2024/"

    # --- Job Naming ---
    base_job_name_prefix = job_config['base_job_name_prefix'].replace("{{instrument}}", args.instrument)
    unique_job_name_suffix = time.strftime("%Y%m%d-%H%MS")
    job_name_for_sagemaker = f"{base_job_name_prefix}-{unique_job_name_suffix}"[:63]

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    # --- KEY CHANGE: Use Estimator with entry_point and source_dir ---
    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        entry_point="run_baseline_test.py",
        source_dir=".",
        instance_count=job_config['instance_count'],
        instance_type=job_config['instance_type'],
        volume_size=job_config['volume_size_gb'],
        output_path=s3_output_base_path,
        sagemaker_session=sagemaker_session,
        use_spot_instances=job_config['use_spot'],
        max_run=job_config['max_run_seconds'],
        hyperparameters={}, 
        environment={"PYTHONUNBUFFERED": "1"}
    )

    print(f"\n--- Launching SageMaker Training Job ---")
    print(f"Job Name: {job_name_for_sagemaker}")
    print(f"Instance Type: {job_config['instance_type']}")
    print(f"Entry Point: {estimator.entry_point}")
    print(f"Input Data: {s3_input_data_uri}")

    try:
        estimator.fit({"training": s3_input_data_uri}, job_name=job_name_for_sagemaker, wait=False)
        actual_job_name = estimator.latest_training_job.job_name
        print(f"\n✅ Training job '{actual_job_name}' LAUNCHED successfully!")
        print(f"   View job in SageMaker console: https://{region}.console.aws.amazon.com/saker/home?region={region}#/jobs/{actual_job_name}")
    except Exception as e:
        print(f"\n❌ Error LAUNCHING SageMaker job: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
