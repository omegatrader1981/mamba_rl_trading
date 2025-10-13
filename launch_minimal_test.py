import sagemaker
from sagemaker.pytorch import PyTorch
import logging
import os

# Disable telemetry
os.environ["SAGEMAKER_TELEMETRY_OPT_OUT"] = "true"
logging.getLogger("sagemaker.telemetry").setLevel(logging.ERROR)

print("--- SCRIPT STARTED ---")

# --- Hardcoded Configuration ---
REGION = "eu-west-2"
ACCOUNT_ID = "537124950121"
ROLE_ARN = "arn:aws:iam::537124950121:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/mamba_rl_trading:refactor-v1"
DATA_URI = f"s3://sagemaker-{REGION}-{ACCOUNT_ID}/mamba-rl-trading/data"
INSTANCE_TYPE = "ml.g4dn.xlarge"

# --- Estimator Definition ---
estimator = PyTorch(
    entry_point='src/train.py',
    source_dir='.',
    role=ROLE_ARN,
    image_uri=IMAGE_URI,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    max_run=3600,
    # --- CRITICAL DIAGNOSTIC STEP: Disable Spot ---
    use_spot_instances=False,
    hyperparameters={'experiment': 'sac_broad_smoke', 'instrument': 'mnq'},
    sagemaker_session=sagemaker.Session()
)

print("--- ESTIMATOR CREATED. CALLING FIT() NOW... ---")

try:
    estimator.fit(
        inputs={'training': DATA_URI},
        wait=False,
        job_name=f"minimal-smoke-test-{''.join(str(sagemaker.Session().default_bucket())[-6:])}" # Unique name
    )
    print("--- FIT() CALL COMPLETED SUCCESSFULLY ---")
except Exception as e:
    print(f"--- FIT() CALL FAILED WITH AN ERROR ---")
    print(e)

print("--- SCRIPT FINISHED ---")
