#!/usr/bin/env python3
"""
AWS Preflight Check for Stage 0 Smoke Test
Validates local environment, dependencies, configs, SageMaker/ECR setup.
Automatically detects AWS region and validates ECR image.
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess
import yaml
import logging
import boto3
import botocore

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ======================
# USER CONFIGURATION
# ======================
ACCOUNT_ID = "537124950121"          # ✅ Your AWS account ID
ECR_REPOSITORY = "mamba_rl_trading"  # ✅ ECR repo name
IMAGE_TAG = "refactor-v1"            # ✅ Image tag to check

# ======================
# AWS REGION DETECTION
# ======================
def get_current_aws_region():
    """Detect AWS region from environment or AWS CLI config."""
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
    if not region:
        try:
            import subprocess
            region = subprocess.check_output(
                ['aws', 'configure', 'get', 'region'], text=True
            ).strip()
        except Exception:
            region = None
    if not region:
        print("❌ Could not detect AWS region. Set AWS_REGION or AWS_DEFAULT_REGION.")
        sys.exit(1)
    print(f"🔍 Detected AWS region: {region}")
    return region

# ======================
# Preflight Checker
# ======================
class PreFlightChecker:
    """Pre-flight validation for smoke test launch."""

    def __init__(self, region: str):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        self.region = region

    def run_all_checks(self) -> bool:
        log.info("="*70)
        log.info("STAGE 0 PRE-FLIGHT CHECKLIST")
        log.info("="*70 + "\n")

        self._check_repo_structure()
        self._check_dependencies()
        self._check_config_files()
        self._check_data_files()
        self._check_aws_credentials()
        self._check_sagemaker_role()
        self._check_code_syntax()
        self._check_imports()
        self._check_smoke_config()
        self._print_results()

        return len(self.checks_failed) == 0

    def _check_repo_structure(self):
        required_paths = [
            'src/train.py',
            'src/pipeline/',
            'src/environment/',
            'src/features/',
            'src/model/',
            'conf/config.yaml',
            'conf/experiment/smoke_test.yaml',
            'launch_configs.yaml',
            'data/'
        ]
        missing = [p for p in required_paths if not Path(p).exists()]
        if not missing:
            self.checks_passed.append("✅ All required files/directories present")
        else:
            self.checks_failed.append(f"❌ Missing paths: {', '.join(missing)}")

    def _check_dependencies(self):
        critical_packages = {
            'torch': 'PyTorch',
            'stable_baselines3': 'Stable Baselines3',
            'gymnasium': 'Gymnasium',
            'hydra': 'Hydra',
            'omegaconf': 'OmegaConf',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'boto3': 'Boto3',
            'sagemaker': 'SageMaker SDK',
            'mamba_ssm': 'Mamba-SSM'
        }
        missing = []
        for package, name in critical_packages.items():
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(name)
        if not missing:
            self.checks_passed.append("✅ All critical packages installed")
        else:
            self.checks_failed.append(f"❌ Missing packages: {', '.join(missing)}")

    def _check_config_files(self):
        config_files = [
            'conf/config.yaml',
            'conf/experiment/smoke_test.yaml',
            'launch_configs.yaml'
        ]
        invalid = []
        for fpath in config_files:
            try:
                with open(fpath, 'r') as f:
                    yaml.safe_load(f)
            except Exception as e:
                invalid.append(f"{fpath}: {str(e)[:50]}")
        if not invalid:
            self.checks_passed.append("✅ All config files valid")
        else:
            self.checks_failed.append(f"❌ Invalid configs: {', '.join(invalid)}")

    def _check_data_files(self):
        data_dir = Path('data')
        if not data_dir.exists():
            self.checks_failed.append("❌ Data directory not found")
            return
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            total_size = sum(f.stat().st_size for f in csv_files) / (1024**2)
            self.checks_passed.append(f"✅ Found {len(csv_files)} data file(s), {total_size:.1f}MB total")
        else:
            self.checks_failed.append("❌ No CSV files in data directory")

    def _check_aws_credentials(self):
        try:
            sts = boto3.client('sts', region_name=self.region)
            identity = sts.get_caller_identity()
            account = identity['Account']
            self.checks_passed.append(f"✅ AWS credentials valid (Account: {account})")
        except Exception as e:
            self.checks_failed.append(f"❌ AWS credentials issue: {str(e)[:100]}")

    def _check_sagemaker_role(self):
        try:
            import sagemaker
            role = sagemaker.get_execution_role()
            self.checks_passed.append(f"✅ SageMaker role found")
        except Exception:
            try:
                iam = boto3.client('iam', region_name=self.region)
                roles = iam.list_roles()
                sm_roles = [r['RoleName'] for r in roles['Roles'] if 'SageMaker' in r['RoleName']]
                if sm_roles:
                    self.warnings.append(f"⚠️ Not in SageMaker notebook. Found roles: {', '.join(sm_roles[:3])}")
                    self.warnings.append("⚠️ You'll need to specify the role when launching")
                else:
                    self.checks_failed.append("❌ No SageMaker execution role found")
            except Exception as e:
                self.warnings.append(f"⚠️ Could not verify SageMaker role: {str(e)[:100]}")

    def _check_code_syntax(self):
        critical_files = [
            'src/train.py',
            'src/pipeline/training_pipeline.py',
            'src/environment/trading_env.py'
        ]
        errors = []
        for filepath in critical_files:
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filepath, 'exec')
            except SyntaxError as e:
                errors.append(f"{filepath}:{e.lineno}: {e.msg}")
        if not errors:
            self.checks_passed.append("✅ No syntax errors in critical files")
        else:
            self.checks_failed.append(f"❌ Syntax errors: {'; '.join(errors[:3])}")

    def _check_imports(self):
        test_imports = [
            ('src.environment', 'FuturesTradingEnv'),
            ('src.model', 'MambaFeaturesExtractor'),
            ('src/features', 'create_feature_set'),
        ]
        sys.path.insert(0, str(Path.cwd()))
        import_errors = []
        for module_name, class_name in test_imports:
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, class_name):
                    import_errors.append(f"{module_name} missing {class_name}")
            except Exception as e:
                import_errors.append(f"{module_name}: {str(e)[:50]}")
        sys.path.pop(0)
        if not import_errors:
            self.checks_passed.append("✅ All critical imports successful")
        else:
            self.checks_failed.append(f"❌ Import errors: {'; '.join(import_errors[:3])}")

    def _check_smoke_config(self):
        try:
            with open('conf/experiment/smoke_test.yaml', 'r') as f:
                config = yaml.safe_load(f)
            timesteps = config.get('training', {}).get('total_timesteps')
            if timesteps and timesteps <= 20000:
                self.checks_passed.append(f"✅ Smoke test timesteps <= {timesteps}")
            else:
                self.warnings.append(f"⚠️ Smoke test timesteps high: {timesteps}")
        except Exception as e:
            self.warnings.append(f"⚠️ Config validation error: {str(e)[:100]}")

    def _print_results(self):
        log.info("\n" + "="*70)
        log.info("PRE-FLIGHT CHECK RESULTS")
        log.info("="*70 + "\n")
        if self.checks_passed:
            log.info("✅ PASSED:")
            for check in self.checks_passed:
                log.info(f"   {check}")
        if self.warnings:
            log.info("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                log.info(f"   {warning}")
        if self.checks_failed:
            log.info("\n❌ FAILED:")
            for failure in self.checks_failed:
                log.info(f"   {failure}")
        log.info("\n" + "="*70)
        if self.checks_failed:
            sys.exit(1)

# ======================
# ECR Image Region Validation
# ======================
def check_ecr_image(region: str, image_name=ECR_REPOSITORY, image_tag=IMAGE_TAG):
    """Validate that the ECR image exists in the detected region."""
    print("🔍 Checking ECR image region and tag...")
    try:
        ecr_client = boto3.client('ecr', region_name=region)
        response = ecr_client.describe_images(
            repositoryName=image_name,
            imageIds=[{"imageTag": image_tag}]
        )
        if response.get("imageDetails"):
            uri = f"{ACCOUNT_ID}.dkr.ecr.{region}.amazonaws.com/{image_name}:{image_tag}"
            print(f"✅ Found ECR image: {uri}")
        else:
            print(f"❌ Image tag '{image_tag}' not found in repository '{image_name}'")
            sys.exit(1)
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(f"❌ ECR repository '{image_name}' not found in account {ACCOUNT_ID}.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking ECR image: {e}")
        sys.exit(1)

# ======================
# MAIN
# ======================
def main():
    print("\n=== STAGE 0: AWS PRE-FLIGHT CHECK ===\n")
    region = get_current_aws_region()
    checker = PreFlightChecker(region)
    checker.run_all_checks()
    check_ecr_image(region)
    print("\n✅ All preflight checks passed! Ready to launch Stage 0 smoke test.\n")

if __name__ == "__main__":
    main()
