# Migration notes: Supabase -> Cloud SQL

## Low-downtime approach
1) Enable logical replication on Supabase (if supported by your plan).
2) Create Cloud SQL Postgres instance with matching schema.
3) Use `pg_dump --schema-only` from Supabase, apply to Cloud SQL.
4) Start logical replication or run a full `pg_dump --data-only` for backfill.
5) Cut over app connection strings once replication lag is near zero.

## Simple approach (downtime)
1) Stop ingestion jobs.
2) `pg_dump` from Supabase, restore to Cloud SQL.
3) Update connection strings, restart ingestion.

## Compatibility notes
- This schema uses standard Postgres features (UUID, indexes, views).
- Avoid Supabase-only features in core tables to keep portability.
- If you enable RLS in Supabase, disable it on Cloud SQL or re-create equivalent policies.
