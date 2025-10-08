#!/bin/bash
set -e

REGION="eu-west-2"
LOG_FILE="sagemaker_training_monitor.log"

echo "=== 🚀 SageMaker Training Job Auto Monitor ==="
echo "Region: $REGION"
echo "Log File: $LOG_FILE"
echo "============================================="

get_latest_job() {
  aws sagemaker list-training-jobs \
    --region $REGION \
    --max-results 1 \
    --sort-by CreationTime \
    --sort-order Descending \
    --query 'TrainingJobSummaries[0].TrainingJobName' \
    --output text
}

monitor_job() {
  local job_name="$1"
  echo "📡 Monitoring training job: $job_name"
  echo "---------------------------------------------" | tee -a "$LOG_FILE"

  while true; do
    STATUS=$(aws sagemaker describe-training-job \
      --region $REGION \
      --training-job-name "$job_name" \
      --query 'TrainingJobStatus' \
      --output text)

    LAST_METRIC_TIME=$(aws sagemaker describe-training-job \
      --region $REGION \
      --training-job-name "$job_name" \
      --query 'LastModifiedTime' \
      --output text)

    INSTANCE=$(aws sagemaker describe-training-job \
      --region $REGION \
      --training-job-name "$job_name" \
      --query 'ResourceConfig.InstanceType' \
      --output text)

    START_TIME=$(aws sagemaker describe-training-job \
      --region $REGION \
      --training-job-name "$job_name" \
      --query 'CreationTime' \
      --output text)

    NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    ELAPSED=$(date -u -d "0 $NOW sec - $(date -u -d "$START_TIME" +%s) sec" +"%Hh:%Mm:%Ss")

    echo "[$(date +'%H:%M:%S')] Status: $STATUS | Instance: $INSTANCE | Elapsed: $ELAPSED | LastMetric: $LAST_METRIC_TIME" | tee -a "$LOG_FILE"

    if [[ "$STATUS" == "Completed" || "$STATUS" == "Failed" || "$STATUS" == "Stopped" ]]; then
      echo "✅ Job ended with status: $STATUS"
      break
    fi

    sleep 30
  done
}

LATEST_JOB=$(get_latest_job)
if [[ "$LATEST_JOB" == "None" || -z "$LATEST_JOB" ]]; then
  echo "❌ No training jobs found in region $REGION."
  exit 1
fi

monitor_job "$LATEST_JOB"
