#!/usr/bin/env python3
"""
Stage 0: Repository Baseline Smoke Test Launcher
Validates the entire codebase runs without errors on SageMaker.

Success Criteria:
- Pipeline completes without exceptions
- Training runs for 10,000 steps
- Evaluation produces metrics
- All artifacts are saved correctly

Expected Runtime: 15-25 minutes on ml.g4dn.xlarge
"""

# 🔥 CRITICAL: Disable SageMaker telemetry at the environment level
import os
os.environ["SAGEMAKER_TELEMETRY_OPT_OUT"] = "true"

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_launch_config(config_path: str = "launch_configs.yaml") -> dict:
    """Load the smoke test configuration."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs['smoke_test']

def prepare_sagemaker_session():
    """Initialize SageMaker session and return hardcoded execution role ARN."""
    sess = sagemaker.Session()
    # ✅ Hardcoded ARN — required when running locally as an IAM user
    role = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
    return sess, role

def upload_data_to_s3(sess: sagemaker.Session, data_dir: str = "data") -> str:
    """Upload training data to S3."""
    bucket = sess.default_bucket()
    prefix = "mamba-rl-trading/data"
    
    log.info(f"Uploading data to s3://{bucket}/{prefix}")
    data_uri = sess.upload_data(
        path=data_dir,
        bucket=bucket,
        key_prefix=prefix
    )
    return data_uri

def launch_smoke_test(
    sess: sagemaker.Session,
    role: str,
    data_uri: str,
    config: dict
) -> PyTorch:
    """Launch the smoke test training job on SageMaker."""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{config['base_job_name_prefix'].replace('{{instrument}}', 'mnq')}-{timestamp}"
    
    log.info(f"Launching smoke test job: {job_name}")
    
    # Define the PyTorch estimator
    estimator = PyTorch(
        entry_point='src/train.py',
        source_dir='.',
        role=role,
        instance_type=config['instance_type'],
        instance_count=config['instance_count'],
        volume_size=config['volume_size_gb'],
        use_spot_instances=config['use_spot'],
        max_run=config['max_run_seconds'],
        max_wait=config.get('max_wait_seconds', config['max_run_seconds'] + 3600),
        framework_version='2.0.0',
        py_version='py310',
        hyperparameters={
            'experiment': config['hydra_experiment'],
            'instrument': 'mnq'
        },
        environment={
            'HYDRA_FULL_ERROR': '1',
            'MAMBA_FORCE_BUILD': '1'
        },
        output_path=f"s3://{sess.default_bucket()}/mamba-rl-trading/output",
        checkpoint_s3_uri=f"s3://{sess.default_bucket()}/mamba-rl-trading/checkpoints/smoke-test",
        base_job_name=config['base_job_name_prefix'].replace('{{instrument}}', 'mnq'),
        tags=[
            {'Key': 'Project', 'Value': 'MambaRLTrading'},
            {'Key': 'Stage', 'Value': 'Stage0-Baseline'},
            {'Key': 'Type', 'Value': 'SmokeTest'}
        ]
    )
    
    # Launch the training job
    estimator.fit(
        inputs={'training': data_uri},
        job_name=job_name,
        wait=False
    )
    
    log.info(f"✅ Smoke test job launched: {job_name}")
    log.info(f"Monitor at: https://{sess.boto_session.region_name}.console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    
    return estimator

def main():
    """Main execution flow for Stage 0 smoke test."""
    log.info("="*60)
    log.info("STAGE 0: REPOSITORY BASELINE VALIDATION")
    log.info("="*60)
    
    # Load configuration
    config = load_launch_config()
    
    # Setup SageMaker
    sess, role = prepare_sagemaker_session()
    
    # Upload data
    data_uri = upload_data_to_s3(sess)
    
    # Launch smoke test
    estimator = launch_smoke_test(
        sess=sess,
        role=role,
        data_uri=data_uri,
        config=config
    )
    
    log.info("\n✅ Job submitted successfully. Exiting.")
    log.info(f"Job name: {estimator.latest_training_job.name}")
    log.info("\nTo validate after completion, run:")
    log.info(f"  python validate_smoke_test.py {estimator.latest_training_job.name}")
    
    return 0

if __name__ == "__main__":
    exit(main())
