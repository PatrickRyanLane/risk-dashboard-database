-- Rebuild SERP feature materialized views so updated definitions are applied
-- on existing databases (where CREATE MATERIALIZED VIEW IF NOT EXISTS would
-- otherwise leave old definitions unchanged).
--
-- Run with:
--   psql "$DATABASE_URL" -f sql/rebuild_serp_feature_mvs.sql

begin;

-- Dependent views must be dropped first.
drop view if exists entity_anomalies_v;
drop view if exists entity_weekly_rollup_v;
drop view if exists entity_daily_metrics_v;

-- Recreate SERP feature materialized views from updated definitions.
drop materialized view if exists serp_feature_control_daily_index_mv;
drop materialized view if exists serp_feature_daily_index_mv;
drop materialized view if exists serp_feature_control_daily_mv;
drop materialized view if exists serp_feature_daily_mv;

commit;

-- Recreate objects in dependency order.
\ir serp_feature_daily_mv.sql
\ir serp_feature_control_daily_mv.sql
\ir serp_feature_daily_index_mv.sql
\ir serp_feature_control_daily_index_mv.sql
\ir entity_daily_metrics_v.sql
\ir entity_weekly_rollup_v.sql
\ir entity_anomalies_v.sql
