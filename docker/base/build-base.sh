#!/bin/bash
# docker/base/build-base.sh
# This script builds and pushes the custom base image to ECR.
# Run this manually ONCE, or whenever requirements change.

set -e

ACCOUNT_ID="537124950121"
REGION="eu-west-2"
BASE_IMAGE_NAME="mamba-rl-trading-base"
FULL_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$BASE_IMAGE_NAME:latest"

echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

echo "Building base image: $FULL_URI"
# We must run the build from the project root for the COPY commands to work
docker build -t $FULL_URI -f docker/base/Dockerfile .

echo "Pushing base image to ECR..."
docker push $FULL_URI

echo "✅ Base image pushed successfully: $FULL_URI"
