# Deploy Runbook (Dashboard + Jobs + Scheduler)

This runbook collects the three deploy wrappers we discussed:

- `deploy_dashboard.sh` for the dashboard app service (`risk-dashboard-database`)
- `deploy_jobs.sh` for backend processing jobs (`risk-dashboard`)
- `deploy_scheduler.sh` for Cloud Scheduler cron wiring (`risk-dashboard`)

## 1) Dashboard Deploy Wrapper

Script path: `risk-dashboard-database/deploy_dashboard.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gen-lang-client-0154760958}"
REGION="${REGION:-us-west1}"
INTERNAL_SERVICE="${INTERNAL_SERVICE:-risk-dashboard}"
EXTERNAL_SERVICE="${EXTERNAL_SERVICE:-risk-dashboard-external}"
MEMORY="${MEMORY:-1Gi}"
Í
IMAGE_PLATFORM="${IMAGE_PLATFORM:-linux/amd64}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"
NO_CACHE="${NO_CACHE:-0}"

export PROJECT_ID REGION INTERNAL_SERVICE EXTERNAL_SERVICE MEMORY
export IMAGE_PLATFORM IMAGE_TAG NO_CACHE

bash scripts/deploy_cloud_run.sh
```

Run from:

```bash
cd /Users/plane/Documents/GitHub/risk-dashboard-database
bash deploy_dashboard.sh
```

## 2) Jobs Deploy Wrapper

Script path: `risk-dashboard/deploy_jobs.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to gen-lang-client-0154760958}"
JOB_SERVICE_ACCOUNT="${JOB_SERVICE_ACCOUNT:?Set JOB_SERVICE_ACCOUNT to risk-dashboard-jobs@<PROJECT_ID>.iam.gserviceaccount.com}"

REGION="${REGION:-us-east1}"
AR_LOCATION="${AR_LOCATION:-$REGION}"
AR_REPO="${AR_REPO:-risk-dashboard-jobs}"
IMAGE_NAME="${IMAGE_NAME:-risk-dashboard-jobs}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"

SKIP_BUILD="${SKIP_BUILD:-false}"
DEPLOY_ALERT_JOBS="${DEPLOY_ALERT_JOBS:-true}"
DEPLOY_REFRESH_MVS_JOB="${DEPLOY_REFRESH_MVS_JOB:-true}"

LLM_PROVIDER="${LLM_PROVIDER:-gemini}"
LLM_MODEL="${LLM_MODEL:-gemini-2.5-flash}"
LLM_MAX_CALLS="${LLM_MAX_CALLS:-200}"
LLM_SUMMARY_MAX_CALLS="${LLM_SUMMARY_MAX_CALLS:-20}"

DASHBOARD_BASE_URL="${DASHBOARD_BASE_URL:-https://risk-dashboard-168007850529.us-west1.run.app}"

export PROJECT_ID JOB_SERVICE_ACCOUNT REGION AR_LOCATION AR_REPO IMAGE_NAME IMAGE_TAG
export SKIP_BUILD DEPLOY_ALERT_JOBS DEPLOY_REFRESH_MVS_JOB
export LLM_PROVIDER LLM_MODEL LLM_MAX_CALLS LLM_SUMMARY_MAX_CALLS DASHBOARD_BASE_URL

bash scripts/deploy_cloud_run_jobs.sh
```

Run from:

```bash
cd /Users/plane/Documents/GitHub/risk-dashboard
bash deploy_jobs.sh
```

## 3) Scheduler Deploy Wrapper

Script path: `risk-dashboard/deploy_scheduler.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to gen-lang-client-0154760958}"
SCHEDULER_SERVICE_ACCOUNT="${SCHEDULER_SERVICE_ACCOUNT:?Set SCHEDULER_SERVICE_ACCOUNT to risk-dashboard-scheduler@<PROJECT_ID>.iam.gserviceaccount.com}"

REGION="${REGION:-us-east1}"
TIME_ZONE="${TIME_ZONE:-America/New_York}"

DEPLOY_ALERT_JOBS="${DEPLOY_ALERT_JOBS:-true}"
DEPLOY_REFRESH_MVS_SCHEDULE="${DEPLOY_REFRESH_MVS_SCHEDULE:-true}"

export PROJECT_ID SCHEDULER_SERVICE_ACCOUNT REGION TIME_ZONE
export DEPLOY_ALERT_JOBS DEPLOY_REFRESH_MVS_SCHEDULE

bash scripts/deploy_cloud_scheduler_jobs.sh
```

Run from:

```bash
cd /Users/plane/Documents/GitHub/risk-dashboard
bash deploy_scheduler.sh
```

## When To Run Which

1. Dashboard code/UI/API changes (`risk-dashboard-database/dashboard_app/*`): run `deploy_dashboard.sh`
2. Job/script changes (`risk-dashboard/scripts/*`): run `deploy_jobs.sh`
3. Cron/schedule changes (`scripts/deploy_cloud_scheduler_jobs.sh`): run `deploy_scheduler.sh`
4. If you changed both jobs and schedule wiring: run in this order:
   1. `deploy_jobs.sh`
   2. `deploy_scheduler.sh`

## Notes

- Job runtime code is pulled from the jobs container image in Artifact Registry (`*.pkg.dev`), not directly from GitHub at execution time.
- Dashboard service image is currently pushed via the dashboard deploy script defaults to GCR (`gcr.io/...`) unless you override.
