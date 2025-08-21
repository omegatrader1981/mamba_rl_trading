# <<< DEFINITIVE FINAL VERSION: Uses HyperparameterTuner for robust Spot Fleet support. >>>

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
import time
import traceback
import yaml
import argparse
import sys
import boto3

def main():
    parser = argparse.ArgumentParser(description="Launch a SageMaker training job for the DRL trading strategy.")
    parser.add_argument("--job-type", type=str, required=True)
    parser.add_argument("--instrument", type=str, required=True)
    parser.add_argument("--image-tag", type=str, default="latest")
    args = parser.parse_args()

    with open("launch_configs.yaml", 'r') as f:
        launch_configs = yaml.safe_load(f)

    job_config = launch_configs.get(args.job_type)
    if not job_config:
        print(f"❌ ERROR: Job type '{args.job_type}' not found. Exiting.")
        sys.exit(1)
    
    region = job_config.get("region", "eu-west-1")
    print(f"--- CONFIGURING FOR JOB [{args.job_type}] IN REGION [{region}] ---")

    account_id = "537124950121"
    image_name = "mamba-rl-trading"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{args.image_tag}"
    role_arn = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
    
    s3_bucket_name = f"mambabot-{region}"
    s3_output_base_path = f"s3://{s3_bucket_name}/mamba-rl-trading/output/"
    s3_input_data_uri = f"s3://{s3_bucket_name}/databento_mnq_downloads/"

    base_job_name_prefix = job_config['base_job_name_prefix'].replace("{{instrument}}", args.instrument)
    unique_job_name_suffix = time.strftime("%Y%m%d-%H%M%S")
    job_name_for_sagemaker = f"{base_job_name_prefix}-{unique_job_name_suffix}"[:63]

    hydra_overrides = {
        "instrument": args.instrument,
        "experiment": job_config['hydra_experiment']
    }

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    primary_instance = (job_config.get("instance_type") or 
                        job_config.get("instance_types", ["ml.g4dn.2xlarge"])[0])

    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=job_config['instance_count'],
        instance_type=primary_instance,
        volume_size=job_config['volume_size_gb'],
        output_path=s3_output_base_path,
        sagemaker_session=sagemaker_session,
        use_spot_instances=job_config['use_spot'],
        max_run=job_config['max_run_seconds'],
        max_wait=job_config.get('max_wait_seconds', job_config['max_run_seconds']),
        checkpoint_s3_uri=job_config.get('checkpoint_s3_uri'),
        hyperparameters=hydra_overrides,
        environment={"PYTHONUNBUFFERED": "1"}
    )

    # Use HyperparameterTuner to enable instance diversity (Spot Fleet)
    hyperparameter_ranges = {
        'dummy_param': IntegerParameter(0, 0) # Dummy param, as tuner requires one.
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='sharpe', # Not used for single job, but required.
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{'Name': 'sharpe', 'Regex': 'Final Sharpe Ratio: ([-+]?[0-9]*\.?[0-9]+)'}],
        max_jobs=1,
        max_parallel_jobs=1,
        base_tuning_job_name=job_name_for_sagemaker,
        training_instance_types=job_config.get("instance_types")
    )

    print(f"\n--- Launching SageMaker Hyperparameter Tuning Job (as a wrapper for Spot Fleet) ---")
    print(f"Job Name: {job_name_for_sagemaker}")
    print(f"Instance Fleet: {job_config.get('instance_types')}")

    try:
        tuner.fit({"training": s3_input_data_uri}, wait=False)
        actual_job_name = tuner.latest_tuning_job.job_name
        print(f"\n✅ Tuning job '{actual_job_name}' LAUNCHED successfully!")
        print(f"   View job in SageMaker console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/hyper-tuning-jobs/{actual_job_name}")
    except Exception as e:
        print(f"\n❌ Error LAUNCHING SageMaker job: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
