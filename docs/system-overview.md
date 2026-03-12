# Risk Dashboard System Overview

This document describes the current end-to-end architecture across the two repos:

- `risk-dashboard` owns the active collection, scoring, ingest, backfill, enrichment, and alerting jobs.
- `risk-dashboard-database` owns the Postgres schema, dashboard materialized views, Flask app, and Cloud Run deployment.

The important correction to older docs is ingest ownership: the active ingest and DB-writing scripts now live in `risk-dashboard/scripts/`, while `risk-dashboard-database/src/` is a legacy copy of earlier helpers.

## 1. Repo boundaries

### `risk-dashboard`

Primary owner of:

- Google News article collection
- SERP result processing
- SERP feature ingest from raw parquet
- Roster and boards sync
- CSV backfills into Postgres
- LLM enrichment of DB rows
- Alerting
- GCS output generation

### `risk-dashboard-database`

Primary owner of:

- `sql/schema.sql`
- Dashboard materialized views in `sql/*_mv.sql`
- Flask app in `dashboard_app/app.py`
- Internal versus external auth and edit behavior
- Cloud Run deployment in `scripts/deploy_cloud_run.sh`

## 2. Data sources and pipeline stages

### 2.1 Rosters

`risk-dashboard/scripts/ingest_roster_only.py` loads:

- `gs://risk-dashboard/rosters/main-roster.csv`
- `gs://risk-dashboard/rosters/boards-roster.csv`

into:

- `companies`
- `ceos`
- `boards`

This roster sync runs inside the daily brand and CEO workflows before article or SERP processing.

### 2.2 Articles

`risk-dashboard/scripts/news_articles_brands.py` and `news_articles_ceos.py`:

- fetch Google News RSS content
- apply `risk_rules.py` heuristics plus VADER scoring
- classify control and finance-routine cases
- mark uncertain rows for later review and LLM enrichment
- write modal CSV outputs to GCS under `data/processed_articles/`
- upsert `articles` plus article mention rows directly into Postgres when `DATABASE_URL` is present

After article collection, `news_sentiment_brands.py` and `news_sentiment_ceos.py` build:

- per-date article table CSVs
- rolling daily chart CSVs under `data/daily_counts/`

The daily workflows also run `backfill_article_mentions_daily.py` to populate:

- `company_article_mentions_daily`
- `ceo_article_mentions_daily`

Those daily tables are the main source for article count APIs and dashboard article aggregates.

### 2.3 SERP results

`risk-dashboard/scripts/process_serps_brands.py` and `process_serps_ceos.py`:

- read raw SERP parquet from S3
- resolve entity matching from roster data
- classify sentiment, control, finance-routine, and uncertain rows
- write modal and table CSVs to `data/processed_serps/`
- write rolling SERP daily-count CSVs to `data/daily_counts/`
- emit mismatch files for unmapped or unresolved queries
- upsert `serp_runs` and `serp_results` directly into Postgres when `DATABASE_URL` is present

This direct DB write path is now part of the normal daily pipeline. CSV ingest remains available for backfills.

### 2.4 SERP features

`risk-dashboard/scripts/ingest_serp_features_parquet.py` is the current feature ingest path for raw SERP features. It reads the raw parquet inputs directly and writes:

- `serp_feature_daily`
- `serp_feature_items`

It also computes feature sentiment and control rollups and writes Top Stories narrative tags for negative, non-financial items.

Related jobs:

- `recompute_serp_feature_daily.py` repairs aggregate rows from DB state
- `backfill_narrative_tags.py` backfills narrative tags and gate logic

### 2.5 Metrics

`fetch_stock_data.py` and `fetch_trends_data.py` generate:

- `data/stock_prices/YYYY-MM-DD-stock-data.csv`
- `data/trends_data/YYYY-MM-DD-trends-data.csv`

These files are the metric interchange layer. `bulk_ingest.py` can load them into:

- `stock_prices_daily`
- `stock_price_snapshots`
- `trends_daily`
- `trends_snapshots`

### 2.6 Backfills and repair ingest

`risk-dashboard/scripts/bulk_ingest.py` is the current CSV backfill and repair entrypoint. It can read local files or `gs://risk-dashboard/data` and load:

- article CSVs
- optional processed SERP CSVs via `--include-serps`
- roster and boards files
- stock and trends files

This path is useful for backfills and recovery, but it is no longer the only way article and SERP rows reach Postgres.

`risk-dashboard/scripts/backfill_articles_from_processed_csv.py` is a narrower backfill path for article mention data.

### 2.7 LLM enrichment

`risk-dashboard/scripts/llm_enrich.py` is a DB-only enrichment job. It updates rows that still need LLM labels in:

- `company_article_mentions`
- `ceo_article_mentions`
- `serp_results`
- `serp_feature_items`

It writes sentiment, risk, control, severity, and reason fields directly back into Postgres and reuses prior labels as a cache when possible.

### 2.8 Alerts

Current alerting is DB-backed:

- `send_crisis_alerts.py`
- `send_targeted_alerts.py`

These jobs read Postgres summary and Top Stories data, then send Slack alerts and Salesforce review actions. Optional LLM summaries are used when `LLM_API_KEY` is configured.

`aggregate_negative_articles.py` still produces `data/daily_counts/negative-articles-summary.csv` in GCS for compatibility and reporting, but current alerting logic is not dependent on that CSV.

## 3. Database model

`risk-dashboard-database/sql/schema.sql` contains both earlier pilot tables and the current normalized schema. The dashboard and current ingest path primarily use:

- `companies`
- `ceos`
- `articles`
- `company_article_mentions`
- `ceo_article_mentions`
- `company_article_mentions_daily`
- `ceo_article_mentions_daily`
- `serp_runs`
- `serp_results`
- `serp_feature_daily`
- `serp_feature_items`
- `serp_feature_item_overrides`
- `serp_feature_summaries`
- `boards`
- `stock_prices_daily`
- `stock_price_snapshots`
- `trends_daily`
- `trends_snapshots`
- `users`
- `user_company_access`
- article and SERP override tables

Important table behavior:

- Article daily tables are range-partitioned by date.
- Overrides are stored separately and applied at query time or in materialized views.
- LLM labels live alongside rule-based labels on mention, SERP result, and SERP feature item rows.
- Narrative tags live on `serp_feature_items`.

## 4. Materialized views and aggregate reads

The dashboard relies primarily on these materialized views:

- `article_daily_counts_mv`
- `serp_daily_counts_mv`
- `serp_feature_daily_mv`
- `serp_feature_control_daily_mv`
- `serp_feature_daily_index_mv`
- `serp_feature_control_daily_index_mv`

These views fold in override data and, where applicable, LLM labels.

Refresh paths:

- `risk-dashboard/scripts/refresh_negative_summary_view.py`
- `/api/internal/refresh_aggregates` in `dashboard_app/app.py`

The refresh helpers use an advisory lock so only one aggregate refresh runs at a time.

## 5. Dashboard app

The Flask app in `risk-dashboard-database/dashboard_app/app.py` serves both the internal and external dashboards from the same codebase.

### 5.1 Static dashboard modes

- `/` serves the current default view
- `/internal/*` serves internal static assets
- `/external/*` serves public static assets

The selected mode depends on:

- `PUBLIC_MODE`
- `DEFAULT_VIEW`
- `ALLOW_EDITS`

### 5.2 Auth and access control

Internal access is controlled with:

- `X-Goog-IAP-JWT-Assertion`
- `IAP_AUDIENCE`
- `ALLOWED_DOMAIN`
- `ALLOWED_EMAILS`

Local development can bypass IAP by setting:

- `ALLOW_UNAUTHED_INTERNAL=true`

External scoping can be narrowed with:

- `EXTERNAL_COMPANY_SCOPE`

Per-user company scoping for internal traffic is driven from:

- `users`
- `user_company_access`

### 5.3 Compatibility CSV endpoints

`/api/data/<path>` still exposes legacy CSV paths, but those files are generated from Postgres queries rather than read from GCS. This includes:

- article and SERP daily-count charts
- processed article and SERP exports
- roster export
- stock and trends exports
- negative summary export

### 5.4 JSON API

Current JSON endpoints include:

- `/api/dates`
- `/api/v1/daily_counts`
- `/api/v1/processed_articles`
- `/api/v1/processed_serps`
- `/api/v1/serp_features`
- `/api/v1/serp_feature_controls`
- `/api/v1/serp_feature_items` (internal only)
- `/api/v1/narrative_tags` (internal only)
- `/api/v1/narrative_timeline` (internal only)
- `/api/v1/serp_feature_series` (internal only)
- `/api/v1/roster`
- `/api/v1/negative_summary`
- `/api/v1/boards`
- `/api/v1/stock_data`
- `/api/v1/trends_data`

The app caches JSON responses in memory and gzip-compresses larger API responses when the client supports it.

### 5.5 Internal write endpoints

Internal-only write and control routes include:

- `/api/internal/serp_feature_summary`
  Generates and stores short LLM summaries for SERP feature items.

- `/api/internal/refresh_aggregates`
- `/api/internal/refresh_aggregates/status`
  Refreshes dashboard materialized views.

- `/api/internal/overrides`
  Writes article, SERP result, and SERP feature item overrides.

- `/api/internal/favorites`
  Toggles favorite flags on companies and CEOs.

Overrides trigger targeted aggregate refreshes and API cache invalidation.

## 6. Deployment

`risk-dashboard-database/scripts/deploy_cloud_run.sh` builds one image from `dashboard_app/` and deploys two Cloud Run services:

- Internal: `risk-dashboard`
- External: `risk-dashboard-external`

Current deployment model:

- internal service is not publicly accessible
- internal service runs with `PUBLIC_MODE=0` and `ALLOW_EDITS=1`
- external service is publicly accessible
- external service runs with `PUBLIC_MODE=1` and `ALLOW_EDITS=0`
- both services receive `DATABASE_URL` and `LLM_API_KEY` from Secret Manager
- both services can receive `LLM_PROVIDER` and `LLM_MODEL`

## 7. End-to-end summary

1. `risk-dashboard` collects articles, SERPs, rosters, and metric files.
2. Daily article and SERP jobs write both GCS compatibility outputs and core Postgres rows.
3. SERP feature ingest writes directly to feature tables in Postgres from raw parquet.
4. Backfill and repair jobs can reload CSV data from GCS into Postgres as needed.
5. `risk-dashboard-database` exposes DB-backed dashboards and APIs from Flask.
6. Cloud Run hosts separate internal and external services from the same image.
