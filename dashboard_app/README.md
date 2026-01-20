# DB-backed Dashboard App

This Flask app serves static dashboards and CSV endpoints backed by Postgres.
Deploy as two Cloud Run services:

- **Internal**: IAP enabled, editing allowed
- **External**: public, editing disabled

## Environment
- `DATABASE_URL` (required)
- `DEFAULT_VIEW` = `internal` or `external`
- `PUBLIC_MODE` = `true` for external service (disables edits)
- `ALLOW_EDITS` = `false` to disable override writes
- `IAP_AUDIENCE` (internal service only)
- `ALLOWED_DOMAIN` or `ALLOWED_EMAILS` (optional allowlist)
- `EXTERNAL_COMPANY_SCOPE` = comma-separated company names for public scoping
- `ALLOW_UNAUTHED_INTERNAL=true` for local dev only

## Routes
- `/internal/*` serves editable dashboards
- `/external/*` serves read-only dashboards
- `/api/data/...` serves CSVs from DB (same paths as legacy GCS)
- `/api/internal/overrides` inserts overrides (internal only)

## IAP notes
Set `IAP_AUDIENCE` to the IAP client ID, e.g.:
`/projects/<PROJECT_NUMBER>/apps/<CLIENT_ID>`

The override endpoint reads the IAP JWT from `X-Goog-IAP-JWT-Assertion`.
