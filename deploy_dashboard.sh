#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gen-lang-client-0154760958}"
REGION="${REGION:-us-west1}"
INTERNAL_SERVICE="${INTERNAL_SERVICE:-risk-dashboard}"
EXTERNAL_SERVICE="${EXTERNAL_SERVICE:-risk-dashboard-external}"
MEMORY="${MEMORY:-1Gi}"

IMAGE_PLATFORM="${IMAGE_PLATFORM:-linux/amd64}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"
NO_CACHE="${NO_CACHE:-0}"

export PROJECT_ID REGION INTERNAL_SERVICE EXTERNAL_SERVICE MEMORY
export IMAGE_PLATFORM IMAGE_TAG NO_CACHE

bash scripts/deploy_cloud_run.sh
