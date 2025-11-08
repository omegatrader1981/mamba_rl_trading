import sagemaker
from sagemaker.estimator import Estimator
import time
import yaml
import argparse
import sys
import boto3

def main():
    print("--- 1. Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, default="mnq")
    parser.add_argument("--image-tag", type=str, required=True)
    args = parser.parse_args()

    print("\n--- 2. Loading config...")
    with open("launch_configs.yaml", 'r') as f:
        launch_configs = yaml.safe_load(f)

    job_config = launch_configs.get("baseline_test")
    region = job_config.get("region", "eu-west-2")
    account_id = "537124950121"
    image_name = "mamba_rl_trading"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{args.image_tag}"
    role_arn = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
    
    s3_bucket_name = f"mambabot-{region}"
    s3_output_path = f"s3://{s3_bucket_name}/mamba-rl-trading/output/"
    s3_input_data = f"s3://{s3_bucket_name}/databento_mnq_downloads/databento_mnq_downloads_5min_2020_2024/"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"baseline-{args.instrument}-{timestamp}"[:63]

    print(f"\n--- 3. Configuration:")
    print(f"    Mode: CONTAINER")
    print(f"    Image: {image_uri}")
    print(f"    Job: {job_name}")
    print(f"    Instance: {job_config['instance_type']}")

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=job_config['instance_count'],
        instance_type=job_config['instance_type'],
        volume_size=job_config['volume_size_gb'],
        output_path=s3_output_path,
        sagemaker_session=sagemaker_session,
        use_spot_instances=job_config['use_spot'],
        max_run=job_config['max_run_seconds'],
        # <<< FIX: Pass the required max_wait parameter for spot instances >>>
        max_wait=job_config['max_wait_seconds'],
        hyperparameters={
            'experiment': 'baseline_test',
            'instrument': args.instrument
        },
        environment={"PYTHONUNBUFFERED": "1"}
    )

    print(f"\n--- 4. Launching job...")
    try:
        estimator.fit(
            inputs={"training": s3_input_data},
            job_name=job_name,
            wait=False
        )
        
        print(f"\n✅ Job submitted successfully!")
        print(f"   Name: {job_name}")
        print(f"   Console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
        print(f"\n   To monitor logs, run:")
        print(f"   aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --region {region} --follow")
        
    except Exception as e:
        print(f"\n❌ Error launching job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
