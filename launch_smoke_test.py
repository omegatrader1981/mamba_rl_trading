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

# 🔥 Disable telemetry BEFORE any sagemaker imports
import os
os.environ["SAGEMAKER_TELEMETRY_OPT_OUT"] = "true"
os.environ["AWS_TELEMETRY_OPT_OUT"] = "true"

import logging
logging.getLogger("sagemaker.telemetry").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import yaml
from datetime import datetime

# ✅ SINGLE SOURCE OF TRUTH: All resources in eu-west-2
AWS_REGION = "eu-west-2"
ACCOUNT_ID = "537124950121"
ROLE_ARN = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"

def load_launch_config(config_path: str = "launch_configs.yaml") -> dict:
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs['smoke_test']

def prepare_sagemaker_session():
    boto_session = boto3.Session(region_name=AWS_REGION)
    sess = sagemaker.Session(boto_session=boto_session)
    return sess, ROLE_ARN

def get_data_uri() -> str:
    # ✅ Now points to data in eu-west-2 (after you run the aws s3 sync above)
    bucket = f"sagemaker-{AWS_REGION}-{ACCOUNT_ID}"
    return f"s3://{bucket}/mamba-rl-trading/data"

def launch_smoke_test(sess, role, data_uri, config):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_job_name = config['base_job_name_prefix'].replace('{{instrument}}', 'mnq')
    job_name = f"{base_job_name}-{timestamp}"
    
    log.info(f"Launching job: {job_name}")
    
    estimator = PyTorch(
        entry_point='src/train.py',
        source_dir='.',
        role=role,
        instance_type=config['instance_type'],
        instance_count=config['instance_count'],
        volume_size=config['volume_size_gb'],
        use_spot_instances=config['use_spot'],
        max_run=config['max_run_seconds'],
        framework_version='2.0.0',
        py_version='py310',
        image_uri=f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/mamba_rl_trading:refactor-v1",
        hyperparameters={'experiment': config['hydra_experiment'], 'instrument': 'mnq'},
        environment={'HYDRA_FULL_ERROR': '1', 'MAMBA_FORCE_BUILD': '1'},
        output_path=f"s3://{sess.default_bucket()}/mamba-rl-trading/output",
        checkpoint_s3_uri=f"s3://{sess.default_bucket()}/mamba-rl-trading/checkpoints/smoke-test",
        tags=[
            {'Key': 'Project', 'Value': 'MambaRLTrading'},
            {'Key': 'Stage', 'Value': 'Stage0-Baseline'},
            {'Key': 'Type', 'Value': 'SmokeTest'}
        ]
    )
    
    estimator.fit(inputs={'training': data_uri}, job_name=job_name, wait=False)
    log.info(f"✅ Job launched: {job_name}")
    log.info(f"Monitor: https://{AWS_REGION}.console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    
    return job_name

def main():
    log.info("="*60)
    log.info("STAGE 0: REPOSITORY BASELINE VALIDATION")
    log.info("="*60)
    
    config = load_launch_config()
    sess, role = prepare_sagemaker_session()
    data_uri = get_data_uri()
    log.info(f"Using data: {data_uri}")
    
    job_name = launch_smoke_test(sess, role, data_uri, config)
    
    log.info("\n✅ Job submitted successfully. Exiting.")
    log.info(f"Job name: {job_name}")
    log.info(f"\nValidate after completion:")
    log.info(f"  python validate_smoke_test.py {job_name}")
    return 0

if __name__ == "__main__":
    exit(main())
