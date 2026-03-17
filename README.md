# risk-dashboard-database

`risk-dashboard-database` owns the database-side and serving-side parts of the product:

- Postgres schema and materialized views
- Flask dashboard and API
- Internal versus external service behavior
- Cloud Run deployment

The active daily ingest writers now live in the sibling `risk-dashboard` repo. The `src/` directory in this repo still contains older ingest helpers, but it is no longer the primary ingest path for day-to-day operations.

## What this repo contains

- `sql/schema.sql`
  Main schema, including the current normalized tables used by the dashboard and some earlier pilot tables kept for compatibility.

- `sql/article_daily_counts_mv.sql`
- `sql/serp_daily_counts_mv.sql`
- `sql/serp_feature_daily_mv.sql`
- `sql/serp_feature_control_daily_mv.sql`
- `sql/serp_feature_daily_index_mv.sql`
- `sql/serp_feature_control_daily_index_mv.sql`
- `sql/negative_summary_mv.sql`
  Materialized views used for dashboard counts and feature rollups.

- `sql/entity_daily_metrics_v.sql`
- `sql/entity_weekly_rollup_v.sql`
- `sql/entity_anomalies_v.sql`
  Read-only analytics views used by the AI-ready insights endpoints.

- `dashboard_app/app.py`
  Flask app that serves static dashboards, DB-backed JSON APIs, CSV compatibility endpoints, internal edit routes, and aggregate refresh endpoints.

- `dashboard_app/static/internal/` and `dashboard_app/static/external/`
  Separate static dashboards for internal editable mode and external read-only mode.

- `scripts/deploy_cloud_run.sh`
  Builds one Docker image from `dashboard_app/` and deploys it to the internal and external Cloud Run services.

- `src/*.py`
  Legacy ingest copies retained for reference. Current ingest and backfill work should usually target `risk-dashboard/scripts/*.py` instead.

## Current ownership split

- `risk-dashboard`
  Active collection, scoring, DB ingest and backfills, LLM enrichment, metric generation, and alerting.

- `risk-dashboard-database`
  Schema, materialized views, dashboard/API, auth, overrides, favorites, and Cloud Run deployment.

## Setup

1. Create or point at a Postgres database.
2. Set `DATABASE_URL`.
3. Apply the schema and dashboard materialized views.
4. Populate data using the active scripts in the sibling `risk-dashboard` repo.
5. Run the Flask app locally or deploy with `scripts/deploy_cloud_run.sh`.

Example:

```bash
export DATABASE_URL="postgresql://postgres:<password>@<host>:5432/postgres"

for f in \
  sql/schema.sql \
  sql/article_daily_counts_mv.sql \
  sql/serp_daily_counts_mv.sql \
  sql/serp_feature_daily_mv.sql \
  sql/serp_feature_control_daily_mv.sql \
  sql/serp_feature_daily_index_mv.sql \
  sql/serp_feature_control_daily_index_mv.sql \
  sql/negative_summary_mv.sql \
  sql/entity_daily_metrics_v.sql \
  sql/entity_weekly_rollup_v.sql \
  sql/entity_anomalies_v.sql
do
  psql "$DATABASE_URL" -f "$f"
done
```

Optional:

- `sql/rpcs.sql` for helper RPCs
- `sql/rls.sql` if you want row-level security policies

## Loading data

Use the active ingest scripts from `../risk-dashboard/` if you have both repos checked out side by side.

Examples:

```bash
export DATABASE_URL="postgresql://..."

python ../risk-dashboard/scripts/ingest_roster_only.py \
  --roster-path gs://risk-dashboard/rosters/main-roster.csv \
  --boards-path gs://risk-dashboard/rosters/boards-roster.csv

python ../risk-dashboard/scripts/bulk_ingest.py \
  --data-dir gs://risk-dashboard/data \
  --date YYYY-MM-DD

python ../risk-dashboard/scripts/ingest_serp_features_parquet.py \
  --date YYYY-MM-DD \
  --entity-type brand

python ../risk-dashboard/scripts/llm_enrich.py --max-calls 200
```

Current normalized tables populated for the dashboard include:

- `companies`, `ceos`, `boards`
- `articles`
- `company_article_mentions`, `ceo_article_mentions`
- `company_article_mentions_daily`, `ceo_article_mentions_daily`
- `serp_runs`, `serp_results`
- `serp_feature_daily`, `serp_feature_items`, `serp_feature_summaries`
- `stock_prices_daily`, `stock_price_snapshots`
- `trends_daily`, `trends_snapshots`
- Override tables for article mentions, SERP results, and SERP feature items

## Running the dashboard locally

```bash
export DATABASE_URL="postgresql://..."
export DEFAULT_VIEW=internal
export ALLOW_UNAUTHED_INTERNAL=true

python dashboard_app/app.py
```

Useful environment variables:

- `PUBLIC_MODE`
- `ALLOW_EDITS`
- `DEFAULT_VIEW`
- `IAP_AUDIENCE`
- `ALLOWED_DOMAIN`
- `ALLOWED_EMAILS`
- `EXTERNAL_COMPANY_SCOPE`
- `ALLOW_UNAUTHED_INTERNAL`
- `LLM_API_KEY`
- `LLM_PROVIDER`
- `LLM_MODEL`

## API and dashboard behavior

The Flask app serves:

- Static dashboards from `/`, `/internal/*`, and `/external/*`
- Legacy-style CSV endpoints under `/api/data/*`
- DB-backed JSON endpoints under `/api/v1/*`
- Insights endpoints for trend summaries, anomalies, and supporting evidence under `/api/v1/insights/*`
- Internal-only endpoints for overrides, favorites, LLM SERP feature summaries, and aggregate refreshes under `/api/internal/*`

Internal mode is authenticated and editable. External mode is public and read-only.

## Deployment

`scripts/deploy_cloud_run.sh` deploys two Cloud Run services from one image:

- Internal: `risk-dashboard`
- External: `risk-dashboard-external`

The deploy script sets:

- Internal mode: `PUBLIC_MODE=0`, `ALLOW_EDITS=1`, `DEFAULT_VIEW=internal`
- External mode: `PUBLIC_MODE=1`, `ALLOW_EDITS=0`, `DEFAULT_VIEW=external`

Both services read `DATABASE_URL` and `LLM_API_KEY` from Secret Manager.

For the full end-to-end architecture, see `docs/system-overview.md`.
