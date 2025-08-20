# <<< DEFINITIVE MIGRATION VERSION: Points to the eu-west-1 region. >>>

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import time
import traceback
import yaml
import argparse
import sys
import boto3

def main():
    parser = argparse.ArgumentParser(description="Launch a SageMaker training job for the DRL trading strategy.")
    parser.add_argument("--job-type", type=str, required=True, help="The type of job to launch (e.g., 'phase6a_sac_broad'). Must match a key in launch_configs.yaml.")
    parser.add_argument("--instrument", type=str, required=True, help="The instrument to use (e.g., 'mnq').")
    parser.add_argument("--image-tag", type=str, default="latest", help="The Docker image tag to use.")
    args = parser.parse_args()

    with open("launch_configs.yaml", 'r') as f:
        launch_configs = yaml.safe_load(f)

    job_config = launch_configs.get(args.job_type)
    if not job_config:
        print(f"❌ ERROR: Job type '{args.job_type}' not found in launch_configs.yaml. Exiting.")
        sys.exit(1)
    
    print(f"--- CONFIGURING FOR JOB [{args.job_type}] IN REGION [eu-west-1] ---")

    account_id = "537124950121"
    region = "eu-west-1"
    image_name = "mamba-rl-trading"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{args.image_tag}"
    role_arn = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
    
    s3_bucket_name = "mambabot-eu-west-1"
    s3_output_base_path = f"s3://{s3_bucket_name}/mamba-rl-trading/output/"
    s3_input_data_uri = f"s3://{s3_bucket_name}/databento_mnq_downloads/databento_mnq_downloads_5min_2020_2024/"

    base_job_name_prefix = job_config['base_job_name_prefix'].replace("{{instrument}}", args.instrument)
    unique_job_name_suffix = time.strftime("%Y%m%d-%H%M%S")
    job_name_for_sagemaker = f"{base_job_name_prefix}-{unique_job_name_suffix}"[:63]

    hydra_overrides = {
        "instrument": args.instrument,
        "experiment": job_config['hydra_experiment']
    }

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=job_config['instance_count'],
        instance_type=job_config['instance_type'],
        volume_size=job_config['volume_size_gb'],
        output_path=s3_output_base_path,
        sagemaker_session=sagemaker_session,
        base_job_name=job_name_for_sagemaker,
        use_spot_instances=job_config['use_spot'],
        max_run=job_config['max_run_seconds'],
        max_wait=job_config.get('max_wait_seconds', job_config['max_run_seconds']),
        checkpoint_s3_uri=job_config.get('checkpoint_s3_uri'),
        hyperparameters=hydra_overrides,
        environment={"PYTHONUNBUFFERED": "1"}
    )

    print(f"\n--- Launching SageMaker Training Job ---")
    print(f"Job Name: {estimator.base_job_name}")
    print(f"Docker Image: {image_uri}")
    print(f"Instance: {job_config['instance_type']} (x{job_config['instance_count']})")
    print(f"Input Data: {s3_input_data_uri}")
    print(f"Hydra Overrides: {hydra_overrides}")

    try:
        estimator.fit({"training": s3_input_data_uri}, wait=False)
        actual_job_name = estimator.latest_training_job.job_name
        print(f"\n✅ Training job '{actual_job_name}' LAUNCHED successfully!")
        print(f"   View job in SageMaker console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{actual_job_name}")
    except Exception as e:
        print(f"\n❌ Error LAUNCHING SageMaker training job: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
