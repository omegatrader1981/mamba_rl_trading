#!/usr/bin/env python3
"""
Stage 0: Smoke Test Results Validator
Comprehensive validation of smoke test outputs against success criteria.

Run after the smoke test completes to ensure all systems are functioning.
"""

import boto3
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import re

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SmokeTestValidator:
    """Validates smoke test results against Stage 0 criteria."""
    
    def __init__(self, job_name: str):
        self.job_name = job_name
        self.s3_client = boto3.client('s3')
        self.sm_client = boto3.client('sagemaker')
        self.logs_client = boto3.client('logs')
        
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def run_all_checks(self) -> bool:
        """Execute all validation checks."""
        log.info("\n" + "="*70)
        log.info("STAGE 0 VALIDATION: COMPREHENSIVE SMOKE TEST CHECK")
        log.info("="*70 + "\n")
        
        # 1. Job Status Check
        log.info("📋 [1/8] Checking training job status...")
        self._check_job_status()
        
        # 2. Runtime Check
        log.info("⏱️  [2/8] Verifying runtime duration...")
        self._check_runtime()
        
        # 3. Model Artifacts Check
        log.info("💾 [3/8] Validating model artifacts...")
        self._check_model_artifacts()
        
        # 4. Output Files Check
        log.info("📁 [4/8] Checking output files...")
        self._check_output_files()
        
        # 5. Logs Analysis
        log.info("📝 [5/8] Analyzing training logs...")
        self._analyze_logs()
        
        # 6. Metrics Validation
        log.info("📊 [6/8] Validating evaluation metrics...")
        self._check_metrics()
        
        # 7. Memory/Resource Check
        log.info("💻 [7/8] Checking resource utilization...")
        self._check_resources()
        
        # 8. Data Pipeline Check
        log.info("🔄 [8/8] Validating data pipeline...")
        self._check_data_pipeline()
        
        # Print results
        self._print_results()
        
        return len(self.checks_failed) == 0
    
    def _check_job_status(self):
        """Check if the training job completed successfully."""
        try:
            response = self.sm_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            
            status = response['TrainingJobStatus']
            
            if status == 'Completed':
                self.checks_passed.append(
                    "✅ Training job completed successfully"
                )
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                self.checks_failed.append(
                    f"❌ Training job failed: {failure_reason}"
                )
            else:
                self.warnings.append(
                    f"⚠️  Job in unexpected state: {status}"
                )
                
        except Exception as e:
            self.checks_failed.append(f"❌ Could not check job status: {e}")
    
    def _check_runtime(self):
        """Verify the job ran for an appropriate duration."""
        try:
            response = self.sm_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            
            start = response.get('TrainingStartTime')
            end = response.get('TrainingEndTime')
            
            if start and end:
                duration = (end - start).total_seconds() / 60
                
                if 5 <= duration <= 30:
                    self.checks_passed.append(
                        f"✅ Runtime appropriate: {duration:.1f} minutes"
                    )
                elif duration < 5:
                    self.warnings.append(
                        f"⚠️  Suspiciously short runtime: {duration:.1f} minutes"
                    )
                else:
                    self.warnings.append(
                        f"⚠️  Long runtime for smoke test: {duration:.1f} minutes"
                    )
            else:
                self.warnings.append("⚠️  Could not determine runtime")
                
        except Exception as e:
            self.warnings.append(f"⚠️  Runtime check error: {e}")
    
    def _check_model_artifacts(self):
        """Validate model artifacts were created and saved."""
        try:
            response = self.sm_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            
            artifacts_uri = response.get('ModelArtifacts', {}).get('S3ModelArtifacts')
            
            if artifacts_uri:
                # Check if file actually exists
                bucket, key = self._parse_s3_uri(artifacts_uri)
                
                try:
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                    self.checks_passed.append(
                        f"✅ Model artifacts saved: {artifacts_uri}"
                    )
                except:
                    self.checks_failed.append(
                        f"❌ Model artifacts URI exists but file not found"
                    )
            else:
                self.checks_failed.append("❌ No model artifacts generated")
                
        except Exception as e:
            self.checks_failed.append(f"❌ Artifact check error: {e}")
    
    def _check_output_files(self):
        """Check that expected output files were created."""
        try:
            response = self.sm_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            
            output_path = response.get('OutputDataConfig', {}).get('S3OutputPath')
            
            if not output_path:
                self.warnings.append("⚠️  No output path configured")
                return
            
            bucket, prefix = self._parse_s3_uri(output_path)
            prefix = f"{prefix}/{self.job_name}/output/"
            
            expected_files = [
                'evaluation_summary.json',
                'test_equity_SmokeTest.csv',
                'equity_curve_SmokeTest.png'
            ]
            
            found_files = []
            missing_files = []
            
            for expected in expected_files:
                try:
                    key = f"{prefix}{expected}"
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                    found_files.append(expected)
                except:
                    missing_files.append(expected)
            
            if found_files:
                self.checks_passed.append(
                    f"✅ Output files created: {len(found_files)}/{len(expected_files)}"
                )
            
            if missing_files:
                self.warnings.append(
                    f"⚠️  Missing output files: {', '.join(missing_files)}"
                )
                
        except Exception as e:
            self.warnings.append(f"⚠️  Output files check error: {e}")
    
    def _analyze_logs(self):
        """Analyze CloudWatch logs for errors and warnings."""
        try:
            log_group = '/aws/sagemaker/TrainingJobs'
            log_stream = self.job_name
            
            # Get recent log events
            response = self.logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                startFromHead=False,
                limit=1000
            )
            
            events = response.get('events', [])
            
            # Count error types
            critical_errors = 0
            warnings = 0
            nan_inf_issues = 0
            
            for event in events:
                message = event['message'].lower()
                
                if 'error' in message and 'critical' in message:
                    critical_errors += 1
                elif 'warning' in message:
                    warnings += 1
                if 'nan' in message or 'inf' in message:
                    nan_inf_issues += 1
            
            if critical_errors == 0:
                self.checks_passed.append("✅ No critical errors in logs")
            else:
                self.checks_failed.append(
                    f"❌ Found {critical_errors} critical errors in logs"
                )
            
            if nan_inf_issues > 0:
                self.warnings.append(
                    f"⚠️  {nan_inf_issues} NaN/Inf warnings (may be normal during training)"
                )
            
            if warnings > 10:
                self.warnings.append(
                    f"⚠️  {warnings} warnings found (review logs)"
                )
                
        except Exception as e:
            self.warnings.append(f"⚠️  Log analysis error: {e}")
    
    def _check_metrics(self):
        """Validate evaluation metrics exist and are reasonable."""
        try:
            response = self.sm_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            
            output_path = response.get('OutputDataConfig', {}).get('S3OutputPath')
            bucket, prefix = self._parse_s3_uri(output_path)
            metrics_key = f"{prefix}/{self.job_name}/output/evaluation_summary.json"
            
            # Download metrics file
            obj = self.s3_client.get_object(Bucket=bucket, Key=metrics_key)
            metrics = json.loads(obj['Body'].read())
            
            # Check for required metrics
            required_metrics = ['sharpe', 'sortino', 'max_drawdown', 'n_trades']
            
            found_metrics = [m for m in required_metrics if m in metrics]
            
            if len(found_metrics) == len(required_metrics):
                self.checks_passed.append(
                    f"✅ All required metrics present"
                )
                
                # Check if metrics are reasonable (not NaN, not extreme)
                sharpe = metrics.get('sharpe', 0)
                sortino = metrics.get('sortino', 0)
                
                if abs(sharpe) < 100 and abs(sortino) < 100:
                    self.checks_passed.append(
                        f"✅ Metrics in reasonable range (Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f})"
                    )
                else:
                    self.warnings.append(
                        f"⚠️  Extreme metric values (may indicate issues)"
                    )
            else:
                missing = set(required_metrics) - set(found_metrics)
                self.warnings.append(
                    f"⚠️  Missing metrics: {', '.join(missing)}"
                )
                
        except Exception as e:
            self.warnings.append(f"⚠️  Metrics validation error: {e}")
    
    def _check_resources(self):
        """Check resource utilization was appropriate."""
        try:
            response = self.sm_client.describe_training_job(
                TrainingJobName=self.job_name
            )
            
            # Check billable time
            billable_seconds = response.get('BillableTimeInSeconds', 0)
            training_seconds = response.get('TrainingTimeInSeconds', 0)
            
            if billable_seconds > 0:
                efficiency = (training_seconds / billable_seconds) * 100
                
                if efficiency > 80:
                    self.checks_passed.append(
                        f"✅ Resource utilization efficient: {efficiency:.1f}%"
                    )
                else:
                    self.warnings.append(
                        f"⚠️  Low resource efficiency: {efficiency:.1f}% (spot interruptions?)"
                    )
            
            # Check instance type
            instance_type = response.get('ResourceConfig', {}).get('InstanceType')
            
            if instance_type == 'ml.g4dn.xlarge':
                self.checks_passed.append(
                    f"✅ Correct instance type: {instance_type}"
                )
            else:
                self.warnings.append(
                    f"⚠️  Unexpected instance type: {instance_type}"
                )
                
        except Exception as e:
            self.warnings.append(f"⚠️  Resource check error: {e}")
    
    def _check_data_pipeline(self):
        """Validate data pipeline executed correctly."""
        try:
            log_group = '/aws/sagemaker/TrainingJobs'
            log_stream = self.job_name
            
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                logStreamNames=[log_stream],
                filterPattern='"Data Preparation Pipeline"'
            )
            
            events = response.get('events', [])
            
            pipeline_started = any('Starting' in e['message'] for e in events)
            pipeline_completed = any('COMPLETED' in e['message'] for e in events)
            
            if pipeline_started and pipeline_completed:
                self.checks_passed.append(
                    "✅ Data pipeline executed successfully"
                )
            elif pipeline_started:
                self.warnings.append(
                    "⚠️  Data pipeline started but completion not confirmed"
                )
            else:
                self.checks_failed.append(
                    "❌ Data pipeline did not execute"
                )
                
        except Exception as e:
            self.warnings.append(f"⚠️  Data pipeline check error: {e}")
    
    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        parts = uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        return bucket, key
    
    def _print_results(self):
        """Print comprehensive validation results."""
        log.info("\n" + "="*70)
        log.info("VALIDATION RESULTS")
        log.info("="*70 + "\n")
        
        # Passed checks
        if self.checks_passed:
            log.info("✅ PASSED CHECKS:")
            for check in self.checks_passed:
                log.info(f"   {check}")
            log.info("")
        
        # Warnings
        if self.warnings:
            log.info("⚠️  WARNINGS:")
            for warning in self.warnings:
                log.info(f"   {warning}")
            log.info("")
        
        # Failed checks
        if self.checks_failed:
            log.info("❌ FAILED CHECKS:")
            for failure in self.checks_failed:
                log.info(f"   {failure}")
            log.info("")
        
        # Summary
        log.info("="*70)
        total_checks = len(self.checks_passed) + len(self.checks_failed)
        pass_rate = (len(self.checks_passed) / total_checks * 100) if total_checks > 0 else 0
        
        log.info(f"SUMMARY: {len(self.checks_passed)}/{total_checks} checks passed ({pass_rate:.1f}%)")
        log.info(f"Warnings: {len(self.warnings)}")
        
        if len(self.checks_failed) == 0:
            log.info("\n🎉 STAGE 0: REPOSITORY BASELINE - PASSED")
            log.info("✅ Codebase is functional and ready for Stage 1")
        else:
            log.info("\n❌ STAGE 0: REPOSITORY BASELINE - FAILED")
            log.info("⛔ Fix critical issues before proceeding to Stage 1")
        
        log.info("="*70 + "\n")

def main():
    """Main execution."""
    import sys
    
    if len(sys.argv) < 2:
        log.error("Usage: python validate_smoke_test.py <training-job-name>")
        log.error("\nExample:")
        log.error("  python validate_smoke_test.py mamba-mnq-smoke-test-20250106-143022")
        return 1
    
    job_name = sys.argv[1]
    
    validator = SmokeTestValidator(job_name)
    success = validator.run_all_checks()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
