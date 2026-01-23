#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gen-lang-client-0154760958}"
REGION="${REGION:-us-west1}"
INTERNAL_SERVICE="${INTERNAL_SERVICE:-risk-dashboard}"
EXTERNAL_SERVICE="${EXTERNAL_SERVICE:-risk-dashboard-external}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${APP_DIR:-${ROOT_DIR}/dashboard_app}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"
IMAGE_NAME="${IMAGE_NAME:-gcr.io/${PROJECT_ID}/risk-dashboard:${IMAGE_TAG}}"
DB_SECRET_NAME="${DB_SECRET_NAME:-DATABASE_URL}"
LLM_SECRET_NAME="${LLM_SECRET_NAME:-LLM_API_KEY}"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "APP_DIR not found: ${APP_DIR}" >&2
  exit 1
fi

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo "App dir: ${APP_DIR}"

gcloud auth configure-docker --quiet

docker build -t "${IMAGE_NAME}" "${APP_DIR}"
docker push "${IMAGE_NAME}"

INTERNAL_ENV_VARS="PUBLIC_MODE=0,ALLOW_EDITS=1,DEFAULT_VIEW=internal"
EXTERNAL_ENV_VARS="PUBLIC_MODE=1,ALLOW_EDITS=0,DEFAULT_VIEW=external"

if [[ -n "${ALLOWED_DOMAIN:-}" ]]; then
  INTERNAL_ENV_VARS="${INTERNAL_ENV_VARS},ALLOWED_DOMAIN=${ALLOWED_DOMAIN}"
fi

if [[ -n "${IAP_AUDIENCE:-}" ]]; then
  INTERNAL_ENV_VARS="${INTERNAL_ENV_VARS},IAP_AUDIENCE=${IAP_AUDIENCE}"
fi

# Internal (editable) service
gcloud run deploy "${INTERNAL_SERVICE}" \
  --image "${IMAGE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "${INTERNAL_ENV_VARS}" \
  --set-secrets "DATABASE_URL=${DB_SECRET_NAME}:latest,LLM_API_KEY=${LLM_SECRET_NAME}:latest"

# External (read-only) service
gcloud run deploy "${EXTERNAL_SERVICE}" \
  --image "${IMAGE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "${EXTERNAL_ENV_VARS}" \
  --set-secrets "DATABASE_URL=${DB_SECRET_NAME}:latest,LLM_API_KEY=${LLM_SECRET_NAME}:latest"

echo "Deployed:"
echo "  Internal: ${INTERNAL_SERVICE}"
echo "  External: ${EXTERNAL_SERVICE}"
