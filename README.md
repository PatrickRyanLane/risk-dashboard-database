# risk-dashboard-database (pilot)

Supabase/Postgres pilot for risk-dashboard ingestion + editable overrides.

## What this includes
- SQL schema for normalized tables (companies, ceos, articles, serp_runs/results, mentions, overrides)
- CSV ingestion that populates normalized tables from existing GCS CSVs
- URL normalization + hashing helper for canonical URL handling

## Setup
1) Create a Supabase project.
2) Run the SQL in `sql/schema.sql`.
3) (Optional) Run `sql/rpcs.sql` for the override helper.
4) (Optional) Run `sql/rls.sql` if you want RLS policies.

## Environment
Set a Postgres connection string:

- `DATABASE_URL` or `SUPABASE_DB_URL`

Example:
```
export DATABASE_URL="postgresql://postgres:<password>@<host>:5432/postgres"
```

## Ingest CSVs
Use `bulk_ingest` to load daily files into normalized tables:
```
python -m src.bulk_ingest \
  --data-dir gs://risk-dashboard/data \
  --date 2025-12-19
```

## Bulk ingest daily folders
```
python -m src.bulk_ingest \
  --data-dir /path/to/news-sentiment-dashboard/data \
  --date 2025-12-19
```

Date ranges:
```
python -m src.bulk_ingest \
  --data-dir /path/to/news-sentiment-dashboard/data \
  --from 2025-12-01 \
  --to 2025-12-19
```

## GCS input
Use a gs:// path as `--data-dir` to read from Cloud Storage:
```
python -m src.bulk_ingest \
  --data-dir gs://risk-dashboard/data \
  --date 2025-12-19
```

Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a service account JSON with
`storage.objects.get` permission on the bucket.

## Roster ingestion
The bulk ingester also loads `rosters/main-roster.csv` from the bucket root
(`gs://<bucket>/rosters/main-roster.csv`). Override with `--roster-path`.

## Boards ingestion
The bulk ingester also loads `rosters/boards-roster.csv` to populate the `boards` table.

## Stock prices + trends ingestion
The bulk ingester also reads per-date files:
- `trends_data/YYYY-MM-DD-trends-data.csv`
- `stock_prices/YYYY-MM-DD-stock-data.csv`

These are expanded into daily rows in `trends_daily` and `stock_prices_daily`,
and snapshots in `trends_snapshots` and `stock_price_snapshots`.

## Overrides
Overrides live in:
- `company_article_overrides`
- `ceo_article_overrides`
- `serp_result_overrides`

## Overrides
Overrides apply to a URL hash across all sources and dates.

Option A: insert into `item_overrides` directly.
Option B: call `apply_item_override(...)` from `sql/rpcs.sql`.

Example via Supabase REST (service role key recommended for internal tools):
```
curl -X POST "$SUPABASE_URL/rest/v1/item_overrides" \
  -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url_hash": "sha256hex...",
    "risk_override": "risk",
    "controlled_override": true,
    "reason": "Reviewed by analyst",
    "user_id": "analyst@company.com"
  }'
```

## Notes
- `url_hash` is a SHA-256 of a normalized URL (see `src/url_utils.py`).
- Idempotency is enforced via `(entity_id, source_type, date, url_hash)`.
- This is designed to migrate cleanly to Cloud SQL later.
