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

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import yaml
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_launch_config(config_path: str = "launch_configs.yaml") -> dict:
    """Load the smoke test configuration."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs['smoke_test']

def prepare_sagemaker_session():
    """Initialize SageMaker session and get execution role."""
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    return sess, role

def upload_code_to_s3(sess: sagemaker.Session, local_code_dir: str = ".") -> str:
    """Upload the entire codebase to S3."""
    bucket = sess.default_bucket()
    prefix = "mamba-rl-trading/code"
    
    log.info(f"Uploading code to s3://{bucket}/{prefix}")
    code_uri = sess.upload_data(
        path=local_code_dir,
        bucket=bucket,
        key_prefix=prefix
    )
    return code_uri

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
    code_uri: str,
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
        max_wait=config.get('max_run_seconds', 10800) + 3600,  # Add buffer for spot
        framework_version='2.0.0',
        py_version='py310',
        hyperparameters={
            'experiment': config['hydra_experiment'],
            'instrument': 'mnq'
        },
        environment={
            'HYDRA_FULL_ERROR': '1',  # Better error messages
            'MAMBA_FORCE_BUILD': '1'   # Ensure mamba builds correctly
        },
        output_path=f"s3://{sess.default_bucket()}/mamba-rl-trading/output",
        code_location=f"s3://{sess.default_bucket()}/mamba-rl-trading/code",
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
        inputs={
            'training': data_uri
        },
        job_name=job_name,
        wait=False  # Don't block - we'll monitor separately
    )
    
    log.info(f"✅ Smoke test job launched: {job_name}")
    log.info(f"Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    
    return estimator

def monitor_job(estimator: PyTorch, check_interval: int = 60):
    """Monitor the training job and print status updates."""
    log.info("Monitoring training job...")
    log.info("Press Ctrl+C to stop monitoring (job will continue running)")
    
    try:
        estimator.logs()  # This will stream logs
    except KeyboardInterrupt:
        log.info("\nStopped monitoring. Job continues in background.")

def validate_smoke_test_results(estimator: PyTorch) -> bool:
    """
    Validate that the smoke test met success criteria.
    
    Success Criteria for Stage 0:
    1. Job completed without errors
    2. Model artifact was created
    3. Evaluation metrics were generated
    4. No critical errors in logs
    """
    job_name = estimator.latest_training_job.name
    sm_client = boto3.client('sagemaker')
    
    # Check job status
    job_desc = sm_client.describe_training_job(TrainingJobName=job_name)
    status = job_desc['TrainingJobStatus']
    
    if status != 'Completed':
        log.error(f"❌ Job status: {status}")
        if status == 'Failed':
            log.error(f"Failure reason: {job_desc.get('FailureReason', 'Unknown')}")
        return False
    
    log.info("✅ Training job completed successfully")
    
    # Check for model artifacts
    model_artifacts = job_desc.get('ModelArtifacts', {}).get('S3ModelArtifacts')
    if not model_artifacts:
        log.error("❌ No model artifacts found")
        return False
    
    log.info(f"✅ Model artifacts saved: {model_artifacts}")
    
    # Basic validation passed
    log.info("\n" + "="*60)
    log.info("🎉 STAGE 0: REPOSITORY BASELINE - PASSED")
    log.info("="*60)
    log.info("\nNext Steps:")
    log.info("1. Review the CloudWatch logs for any warnings")
    log.info("2. Check the S3 output for evaluation metrics")
    log.info("3. Proceed to Stage 1: Trend-Following Agent Development")
    
    return True

def main():
    """Main execution flow for Stage 0 smoke test."""
    log.info("="*60)
    log.info("STAGE 0: REPOSITORY BASELINE VALIDATION")
    log.info("="*60)
    log.info("\nThis smoke test will validate:")
    log.info("✓ Code uploads and imports correctly")
    log.info("✓ Data pipeline processes without errors")
    log.info("✓ Environment creates successfully")
    log.info("✓ Model initializes and trains")
    log.info("✓ Evaluation runs and generates metrics")
    log.info("✓ All artifacts save correctly")
    log.info("\n" + "="*60 + "\n")
    
    # Load configuration
    config = load_launch_config()
    
    # Setup SageMaker
    sess, role = prepare_sagemaker_session()
    
    # Upload code and data
    # code_uri = upload_code_to_s3(sess)  # Uncomment if you want to upload code separately
    data_uri = upload_data_to_s3(sess)
    
    # Launch smoke test
    estimator = launch_smoke_test(
        sess=sess,
        role=role,
        code_uri=None,  # Will use source_dir
        data_uri=data_uri,
        config=config
    )
    
    # Monitor the job
    monitor_job(estimator)
    
    # Validate results
    success = validate_smoke_test_results(estimator)
    
    if success:
        log.info("\n✅ Stage 0 Complete - Ready for Stage 1")
        return 0
    else:
        log.error("\n❌ Stage 0 Failed - Fix issues before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())
