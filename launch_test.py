import sagemaker
from sagemaker.estimator import Estimator
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Argument Parsing ---
if len(sys.argv) < 2:
    log.error("ERROR: Please provide the image tag as an argument.")
    log.error("Usage: python launch_test.py <image-tag>")
    sys.exit(1)

IMAGE_TAG = sys.argv[1]
print(f"--- LAUNCHING JOB WITH IMAGE TAG: {IMAGE_TAG} ---")

# --- Configuration ---
sagemaker_session = sagemaker.Session()
ACCOUNT_ID = sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]
REGION = sagemaker_session.boto_region_name
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20250221T093632"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/mamba_rl_trading:{IMAGE_TAG}"
S3_DATA_URI = f"s3://sagemaker-{REGION}-{ACCOUNT_ID}/mamba-rl-trading/data"

log.info(f"Using Image: {IMAGE_URI}")

# --- Estimator Definition (CONTAINER MODE) ---
estimator = Estimator(
    image_uri=IMAGE_URI,
    role=ROLE_ARN,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    max_run=3600,
    use_spot_instances=True,
    max_wait=3600,
    hyperparameters={
        'experiment': 'smoke_test',
        'instrument': 'mnq'
    },
    sagemaker_session=sagemaker_session
)

# --- Job Submission ---
try:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 🔻 --- THE FIX: Use only the first 7 characters of the git hash --- 🔻
    short_hash = IMAGE_TAG[:7]
    job_name = f"cicd-{short_hash}-{timestamp}"

    log.info(f"Submitting job: {job_name}")
    estimator.fit(
        inputs={'training': S3_DATA_URI},
        wait=False,
        job_name=job_name
    )
    
    log.info(f"✅ Job '{job_name}' submitted! Check the SageMaker console.")
    log.info(f"CloudWatch logs: /aws/sagemaker/TrainingJobs (log stream: {job_name}/algo-1-*)")
    
except Exception as e:
    log.error(f"❌ FIT() CALL FAILED WITH AN ERROR: {e}")
    import traceback
    traceback.print_exc()

print("--- SCRIPT FINISHED ---")
