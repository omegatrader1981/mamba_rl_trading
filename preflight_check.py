#!/usr/bin/env python3
"""
Pre-Flight Checklist for Stage 0 Smoke Test
Validates local environment and dependencies before launching to SageMaker.

Run this BEFORE launching the smoke test to catch issues early.
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class PreFlightChecker:
    """Pre-flight validation for smoke test launch."""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
    
    def run_all_checks(self) -> bool:
        """Execute all pre-flight checks."""
        log.info("="*70)
        log.info("STAGE 0 PRE-FLIGHT CHECKLIST")
        log.info("="*70 + "\n")
        
        # 1. Repository Structure
        log.info("📁 [1/9] Checking repository structure...")
        self._check_repo_structure()
        
        # 2. Python Dependencies
        log.info("🐍 [2/9] Validating Python dependencies...")
        self._check_dependencies()
        
        # 3. Configuration Files
        log.info("⚙️  [3/9] Checking configuration files...")
        self._check_config_files()
        
        # 4. Data Files
        log.info("💾 [4/9] Validating data files...")
        self._check_data_files()
        
        # 5. AWS Credentials
        log.info("🔐 [5/9] Checking AWS credentials...")
        self._check_aws_credentials()
        
        # 6. SageMaker Role
        log.info("👤 [6/9] Validating SageMaker execution role...")
        self._check_sagemaker_role()
        
        # 7. Code Syntax
        log.info("📝 [7/9] Checking code syntax...")
        self._check_code_syntax()
        
        # 8. Import Sanity
        log.info("📦 [8/9] Testing critical imports...")
        self._check_imports()
        
        # 9. Smoke Test Config
        log.info("🎯 [9/9] Validating smoke test configuration...")
        self._check_smoke_config()
        
        # Print results
        self._print_results()
        
        return len(self.checks_failed) == 0
    
    def _check_repo_structure(self):
        """Verify required directories and files exist."""
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
        
        missing = []
        for path in required_paths:
            if not Path(path).exists():
                missing.append(path)
        
        if not missing:
            self.checks_passed.append("✅ All required files/directories present")
        else:
            self.checks_failed.append(
                f"❌ Missing paths: {', '.join(missing)}"
            )
    
    def _check_dependencies(self):
        """Check critical Python packages are installed."""
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
        installed = []
        
        for package, name in critical_packages.items():
            try:
                importlib.import_module(package)
                installed.append(name)
            except ImportError:
                missing.append(name)
        
        if not missing:
            self.checks_passed.append(
                f"✅ All critical packages installed ({len(installed)})"
            )
        else:
            self.checks_failed.append(
                f"❌ Missing packages: {', '.join(missing)}"
            )
    
    def _check_config_files(self):
        """Validate configuration files are valid YAML."""
        config_files = [
            'conf/config.yaml',
            'conf/experiment/smoke_test.yaml',
            'launch_configs.yaml'
        ]
        
        invalid = []
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
            except Exception as e:
                invalid.append(f"{config_file}: {str(e)[:50]}")
        
        if not invalid:
            self.checks_passed.append("✅ All config files valid")
        else:
            self.checks_failed.append(
                f"❌ Invalid configs: {', '.join(invalid)}"
            )
    
    def _check_data_files(self):
        """Verify data directory contains CSV files."""
        data_dir = Path('data')
        
        if not data_dir.exists():
            self.checks_failed.append("❌ Data directory not found")
            return
        
        csv_files = list(data_dir.glob('*.csv'))
        
        if csv_files:
            total_size = sum(f.stat().st_size for f in csv_files) / (1024**2)
            self.checks_passed.append(
                f"✅ Found {len(csv_files)} data file(s), {total_size:.1f}MB total"
            )
        else:
            self.checks_failed.append("❌ No CSV files in data directory")
    
    def _check_aws_credentials(self):
        """Check AWS credentials are configured."""
        try:
            import boto3
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            account = identity['Account']
            self.checks_passed.append(
                f"✅ AWS credentials valid (Account: {account})"
            )
        except Exception as e:
            self.checks_failed.append(
                f"❌ AWS credentials issue: {str(e)[:100]}"
            )
    
    def _check_sagemaker_role(self):
        """Verify SageMaker execution role is accessible."""
        try:
            import sagemaker
            role = sagemaker.get_execution_role()
            self.checks_passed.append(f"✅ SageMaker role found")
        except Exception:
            # Try to get from environment or config
            try:
                import boto3
                iam = boto3.client('iam')
                roles = iam.list_roles()
                sm_roles = [
                    r['RoleName'] for r in roles['Roles'] 
                    if 'SageMaker' in r['RoleName']
                ]
                if sm_roles:
                    self.warnings.append(
                        f"⚠️  Not in SageMaker notebook. Found roles: {', '.join(sm_roles[:3])}"
                    )
                    self.warnings.append(
                        "⚠️  You'll need to specify the role when launching"
                    )
                else:
                    self.checks_failed.append(
                        "❌ No SageMaker execution role found"
                    )
            except Exception as e:
                self.warnings.append(
                    f"⚠️  Could not verify SageMaker role: {str(e)[:100]}"
                )
    
    def _check_code_syntax(self):
        """Run basic syntax check on critical Python files."""
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
            self.checks_failed.append(
                f"❌ Syntax errors: {'; '.join(errors[:3])}"
            )
    
    def _check_imports(self):
        """Test that critical modules can be imported."""
        test_imports = [
            ('src.environment', 'FuturesTradingEnv'),
            ('src.model', 'MambaFeaturesExtractor'),
            ('src.features', 'create_feature_set'),
        ]
        
        # Temporarily add src to path
        sys.path.insert(0, str(Path.cwd()))
        
        import_errors = []
        for module_name, class_name in test_imports:
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, class_name):
                    import_errors.append(
                        f"{module_name} missing {class_name}"
                    )
            except Exception as e:
                import_errors.append(
                    f"{module_name}: {str(e)[:50]}"
                )
        
        sys.path.pop(0)
        
        if not import_errors:
            self.checks_passed.append("✅ All critical imports successful")
        else:
            self.checks_failed.append(
                f"❌ Import errors: {'; '.join(import_errors[:3])}"
            )
    
    def _check_smoke_config(self):
        """Validate smoke test specific configuration."""
        try:
            with open('conf/experiment/smoke_test.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Check critical smoke test parameters
            checks = []
            
            # Should have minimal timesteps
            total_steps = config.get('training', {}).get('total_timesteps')
            if total_steps and total_steps <= 20000:
                checks.append("timesteps <= 20K")
            else:
                self.warnings.append(
                    f"⚠️  Smoke test timesteps high: {total_steps}"
                )
            
            # HPO should be disabled
            hpo_enabled = config.get('optimization', {}).get('enabled', True)
            if not hpo_enabled:
                checks.append("HPO disabled")
            else:
                self.warnings.append(
                    "⚠️  HPO enabled in smoke test (will be slow)"
                )
            
            # Should use minimal data
            train_data = config.get('data', {}).get('regime_definitions', {}).get('train_trending', [])
            if train_data and len(train_data) == 1:
                date_range = train_data[0]
                checks.append("minimal data range")
            else:
                self.warnings.append(
                    "⚠️  Smoke test using more than minimal data"
                )
            
            if checks:
                self.checks_passed.append(
                    f"✅ Smoke test config appropriate: {', '.join(checks)}"
                )
            
            # Check launch config
            with open('launch_configs.yaml', 'r') as f:
                launch_config = yaml.safe_load(f)
            
            smoke_launch = launch_config.get('smoke_test', {})
            instance_type = smoke_launch.get('instance_type')
            
            if instance_type == 'ml.g4dn.xlarge':
                self.checks_passed.append(
                    "✅ Smoke test using correct instance type"
                )
            else:
                self.warnings.append(
                    f"⚠️  Instance type: {instance_type} (expected ml.g4dn.xlarge)"
                )
            
        except Exception as e:
            self.warnings.append(
                f"⚠️  Config validation error: {str(e)[:100]}"
            )
    
    def _print_results(self):
        """Print comprehensive pre-flight results."""
        log.info("\n" + "="*70)
        log.info("PRE-FLIGHT CHECK RESULTS")
        log.info("="*70 + "\n")
        
        if self.checks_passed:
            log.info("✅ PASSED:")
            for check in self.checks_passed:
                log.info(f"   {check}")
            log.info("")
        
        if self.warnings:
            log.info("⚠️  WARNINGS:")
            for warning in self.warnings:
                log.info(f"   {warning}")
            log.info("")
        
        if self.checks_failed:
            log.info("❌ FAILED:")
            for failure in self.checks_failed:
                log.info(f"   {failure}")
            log.info("")
        
        log.info("="*70)
        
        if len(self.checks_failed) == 0:
            log.info("✅ PRE-FLIGHT CHECK PASSED")
            log.info("\nReady to launch smoke test!")
            log.info("\nNext step:")
            log.info("  python launch_smoke_test.py")
        else:
            log.info("❌ PRE-FLIGHT CHECK FAILED")
            log.info("\n⛔ Fix critical issues before launching:")
            for failure in self.checks_failed:
                log.info(f"   • {failure.replace('❌ ', '')}")
        
        log.info("="*70 + "\n")

def main():
    """Main execution."""
    checker = PreFlightChecker()
    success = checker.run_all_checks()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
