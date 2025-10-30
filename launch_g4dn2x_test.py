import sagemaker
from sagemaker.pytorch import PyTorch
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

print("--- SCRIPT STARTED (LAUNCHING ON ml.g4dn.2xlarge) ---")

# --- Configuration ---
sagemaker_session = sagemaker.Session()
ACCOUNT_ID = sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]
REGION = sagemaker_session.boto_region_name
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/mamba_rl_trading:refactor-v2"
STANDARD_SOURCE_PATH="s3://sagemaker-eu-west-2-537124950121/mamba-rl-trading/source/source.tar.gz"
S3_DATA_URI = f"s3://sagemaker-{REGION}-{ACCOUNT_ID}/mamba-rl-trading/data"

log.info(f"Using Image: {IMAGE_URI}")
log.info(f"Using S3 source: {STANDARD_SOURCE_PATH}")

# --- Estimator Definition with ALTERNATIVE GPU INSTANCE ---
estimator = PyTorch(
    entry_point='train.py',
    source_dir=STANDARD_SOURCE_PATH,
    role=ROLE_ARN,
    image_uri=IMAGE_URI,
    # 🔻 --- THE CHANGE --- 🔻
    instance_type='ml.g4dn.2xlarge',
    # 🔺 --- END OF CHANGE --- 🔺
    instance_count=1,
    max_run=3600,
    use_spot_instances=True, # Using spot mode as per your quota
    max_wait=3600,
    hyperparameters={'experiment': 'smoke_test', 'instrument': 'mnq'},
    sagemaker_session=sagemaker_session
)

log.info("--- ESTIMATOR CREATED. CALLING FIT() NOW... ---")

try:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"g4dn2x-smoke-test-{timestamp}"
    
    estimator.fit(
        inputs={'training': S3_DATA_URI},
        wait=False,
        job_name=job_name
    )
    log.info("--- FIT() CALL COMPLETED SUCCESSFULLY ---")
    log.info(f"\n✅ Job '{job_name}' submitted! Check the SageMaker console.")
except Exception as e:
    log.info(f"--- FIT() CALL FAILED WITH AN ERROR ---")
    log.info(e)

log.info("--- SCRIPT FINISHED ---")
