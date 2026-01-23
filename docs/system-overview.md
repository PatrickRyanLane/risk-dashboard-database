# Risk Dashboard System Overview

This document summarizes the current end-to-end flow: data collection in `risk-dashboard`, CSV outputs in GCS, ingestion into Crunchy Bridge Postgres, and the Flask dashboard in `risk-dashboard-database` deployed to Cloud Run.

## 1) Data Collection (risk-dashboard repo)

The `risk-dashboard` repo fetches and processes data daily and writes CSVs to GCS (`gs://risk-dashboard`).

Primary scripts:
- `scripts/news_articles_brands.py` and `scripts/news_articles_ceos.py`
  - Fetch Google News RSS for brands/CEOs
  - Run sentiment rules + VADER
  - Apply finance routine filter (neutralize stock/earnings coverage)
  - Force-negative rules:
    - Reddit sources
    - Legal/crisis terms (brands)
    - CEO-specific negative terms (CEOs)
  - Output: `data/processed_articles/YYYY-MM-DD-*-articles-modal.csv`
- `scripts/process_serps_brands.py` and `scripts/process_serps_ceos.py`
  - Process SERP rows (S3 raw files)
  - Apply sentiment rules + VADER + control classification
  - Apply finance routine filter (neutralize routine market coverage)
  - Force-negative rules:
    - Reddit domains
    - Legal/crisis terms
    - CEO-specific negative terms (CEOs)
  - Controlled/uncontrolled rules:
    - Controlled domains list (social + app stores)
    - Company domain roster match
    - Brand token in domain
    - YouTube channel slug match
    - Facebook/Instagram account pages are controlled; post URLs are uncontrolled
  - Output: `data/processed_serps/YYYY-MM-DD-*-serps-modal.csv`
  - Output (aggregate): `data/processed_serps/YYYY-MM-DD-*-serps-table.csv`
  - Output (daily index): `data/daily_counts/*-serps-daily-counts-chart.csv`
- `scripts/news_sentiment_brands.py` and `scripts/news_sentiment_ceos.py`
  - Aggregate daily counts from processed article CSVs
  - Output: `data/processed_articles/YYYY-MM-DD-*-articles-table.csv`
  - Output (daily index): `data/daily_counts/*-articles-daily-counts-chart.csv`
- `scripts/fetch_stock_data.py`
  - Fetch stock data (yfinance)
  - Output: `data/stock_prices/YYYY-MM-DD-stock-data.csv`
- `scripts/fetch_trends_data.py`
  - Fetch Google Trends
  - Output: `data/trends_data/YYYY-MM-DD-trends-data.csv`
- `scripts/aggregate_negative_articles.py`
  - Build negative summary
  - Output: `data/daily_counts/negative-articles-summary.csv`
- `scripts/send_crisis_alerts.py`
  - Reads negative summary, sends Slack alerts
  - Optional LLM summary in Slack alerts (gated by API key)

Finance routine filter:
- Keywords: earnings, EPS, revenue, guidance, forecast, price target, upgrade, downgrade, dividend, buyback, shares, stock, market cap, quarterly, fiscal, profit, EBITDA, 10-Q, 10-K, SEC, IPO
- Ticker pattern: `NYSE|NASDAQ|AMEX: TICKER`
- Domains: yahoo.com, marketwatch.com, fool.com, benzinga.com, seekingalpha.com, thefly.com, barrons.com, wsj.com, investorplace.com, nasdaq.com, foolcdn.com

Sentiment/control rule summary:
- Force negative if:
  - Reddit source/domain
  - Legal/crisis terms (lawsuit, regulator actions, recall, breach, etc.)
  - CEO-specific negative terms (for CEO pipelines)
- Force neutral if:
  - Finance routine filter hits
  - Neutralize-title list hits
- Control classification (SERPs):
  - Controlled if domain matches roster or controlled list
  - Controlled if YouTube channel slug matches brand token
  - Facebook/Instagram account pages controlled; post URLs uncontrolled

## 2) CSV Storage (GCS)

All outputs are written to:
- `gs://risk-dashboard/data/processed_articles/*`
- `gs://risk-dashboard/data/processed_serps/*`
- `gs://risk-dashboard/data/daily_counts/*`
- `gs://risk-dashboard/data/stock_prices/*`
- `gs://risk-dashboard/data/trends_data/*`

## 3) Ingestion (risk-dashboard-database repo)

The `risk-dashboard-database` repo ingests CSVs from GCS into Crunchy Bridge Postgres.

Key scripts:
- `src/bulk_ingest.py`
  - Reads GCS CSVs and calls:
    - `ingest_v2.py` (articles + SERPs)
    - `ingest_metrics.py` (stock + trends + boards)
- `src/ingest_v2.py`
  - Upserts companies/CEOs
  - Dedupes articles, inserts mentions and SERP results
- `src/ingest_metrics.py`
  - Inserts stock/trends daily + snapshot rows
  - Computes stock daily/7d change if missing

Schema:
- `sql/schema.sql` creates tables for companies, CEOs, articles, SERPs, mentions, overrides, users, boards, stock, trends.

## 4) Dashboard App (risk-dashboard-database repo)

The Flask app lives in:
- `risk-dashboard-database/dashboard_app/app.py`

Key behaviors:
- Internal and external dashboards served from the same app.
- Internal and external are separate Cloud Run services with different env vars.
- JSON endpoints under `/api/v1/*` read from Postgres.
- CSV endpoints under `/api/data/*` are still provided for compatibility (backed by DB, not GCS).

Relevant environment variables:
- `DATABASE_URL` (secret)
- `PUBLIC_MODE` (`0` internal, `1` external)
- `ALLOW_EDITS` (`1` internal, `0` external)
- `DEFAULT_VIEW` (`internal` or `external`)
- `ALLOWED_DOMAIN` (e.g., `terakeet.com`)
- `IAP_AUDIENCE` (internal only)
- `LLM_API_KEY` (optional; enables LLM summary in alerts)

## 5) Deployment (Cloud Run)

Two services:
- Internal: `risk-dashboard`
- External: `risk-dashboard-external`

Deploy script:
- `risk-dashboard-database/scripts/deploy_cloud_run.sh`
  - Builds image from `dashboard_app/`
  - Pushes to GCR
  - Deploys both services
  - Uses Secret Manager for `DATABASE_URL` and `LLM_API_KEY`

## 6) Current LLM Usage (Optional)

LLM is gated and optional:
- `LLM_API_KEY` not set: all LLM calls are skipped.
- Uncertain items are flagged in CSVs for later review.
- `send_crisis_alerts.py` can generate a short daily summary for Slack alerts using the top headlines.

## 7) End-to-End Flow Summary

1. `risk-dashboard` fetches raw data and writes CSVs to GCS.
2. `risk-dashboard-database` ingests those CSVs into Crunchy Bridge Postgres.
3. Flask app serves dashboards and JSON APIs from Postgres.
4. Cloud Run hosts separate internal/external services with IAP for internal access.
