-- Core schema for Supabase/Postgres pilot
-- Uses url_hash as stable identity for overrides across sources.

create extension if not exists pgcrypto;

create table if not exists entities (
  id uuid primary key default gen_random_uuid(),
  entity_type text not null check (entity_type in ('brand','ceo')),
  name text not null,
  alias text,
  created_at timestamptz not null default now(),
  unique (entity_type, name)
);

create table if not exists items (
  id uuid primary key default gen_random_uuid(),
  entity_id uuid not null references entities(id),
  source_type text not null check (source_type in ('news','serp')),
  date date not null,
  title text not null,
  url text not null,
  url_hash text not null,
  snippet text,
  source text,
  position int,
  sentiment_raw text check (sentiment_raw in ('positive','neutral','negative')),
  risk_raw text check (risk_raw in ('risk','no_risk','unknown')),
  controlled_raw boolean,
  company text,
  rule_flags text,
  created_at timestamptz not null default now()
);

create unique index if not exists items_unique_ingest_idx
  on items (entity_id, source_type, date, url_hash);

create index if not exists items_date_idx on items (date);
create index if not exists items_entity_idx on items (entity_id);
create index if not exists items_url_hash_idx on items (url_hash);
create index if not exists items_entity_date_idx on items (entity_id, date);

create table if not exists item_overrides (
  id uuid primary key default gen_random_uuid(),
  url_hash text not null,
  risk_override text check (risk_override in ('risk','no_risk')),
  controlled_override boolean,
  reason text,
  user_id text,
  created_at timestamptz not null default now()
);

create index if not exists overrides_url_hash_idx on item_overrides (url_hash);
create index if not exists overrides_created_idx on item_overrides (created_at);

create table if not exists roster (
  id uuid primary key default gen_random_uuid(),
  ceo text,
  company text not null,
  ceo_alias text,
  websites text,
  stock text,
  sector text,
  created_at timestamptz not null default now(),
  unique (company)
);

create or replace view latest_overrides as
select distinct on (o.url_hash)
  o.url_hash,
  o.risk_override,
  o.controlled_override,
  o.reason,
  o.user_id,
  o.created_at
from item_overrides o
order by o.url_hash, o.created_at desc;

create or replace view items_effective as
select
  i.*,
  coalesce(lo.risk_override, i.risk_raw) as risk_effective,
  coalesce(lo.controlled_override, i.controlled_raw) as controlled_effective,
  lo.reason as override_reason,
  lo.user_id as override_user_id,
  lo.created_at as override_created_at
from items i
left join latest_overrides lo
  on lo.url_hash = i.url_hash;

-- Stock prices (daily expanded)
create table if not exists stock_prices_daily (
  id uuid primary key default gen_random_uuid(),
  ticker text not null,
  company text not null,
  date date not null,
  price numeric,
  created_at timestamptz not null default now(),
  unique (ticker, date)
);

create index if not exists stock_prices_company_idx on stock_prices_daily (company);
create index if not exists stock_prices_date_idx on stock_prices_daily (date);

create table if not exists stock_price_snapshots (
  id uuid primary key default gen_random_uuid(),
  ticker text not null,
  company text not null,
  as_of_date date,
  opening_price numeric,
  daily_change_pct numeric,
  seven_day_change_pct numeric,
  last_updated timestamptz,
  created_at timestamptz not null default now(),
  unique (ticker, last_updated)
);

-- Google trends (daily expanded)
create table if not exists trends_daily (
  id uuid primary key default gen_random_uuid(),
  company text not null,
  date date not null,
  interest int,
  created_at timestamptz not null default now(),
  unique (company, date)
);

create index if not exists trends_company_idx on trends_daily (company);
create index if not exists trends_date_idx on trends_daily (date);

create table if not exists trends_snapshots (
  id uuid primary key default gen_random_uuid(),
  company text not null,
  avg_interest numeric,
  last_updated timestamptz,
  created_at timestamptz not null default now(),
  unique (company, last_updated)
);

-- ==============================
-- V2 normalized schema
-- ==============================

create table if not exists companies (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  ticker text,
  sector text,
  websites text,
  created_at timestamptz not null default now()
);

create table if not exists ceos (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  company_id uuid references companies(id),
  alias text,
  created_at timestamptz not null default now(),
  unique (name, company_id)
);

create table if not exists articles (
  id uuid primary key default gen_random_uuid(),
  canonical_url text not null unique,
  title text,
  publisher text,
  snippet text,
  published_at timestamptz,
  first_seen_at timestamptz,
  last_seen_at timestamptz,
  source text
);

create index if not exists articles_source_idx on articles (source);

create table if not exists serp_runs (
  id uuid primary key default gen_random_uuid(),
  entity_type text not null check (entity_type in ('company','ceo')),
  company_id uuid references companies(id),
  ceo_id uuid references ceos(id),
  query_text text,
  provider text,
  run_at timestamptz not null
);

create index if not exists serp_runs_run_at_idx on serp_runs (run_at);
create unique index if not exists serp_runs_unique_idx
  on serp_runs (entity_type, company_id, ceo_id, run_at);

create table if not exists serp_results (
  id uuid primary key default gen_random_uuid(),
  serp_run_id uuid not null references serp_runs(id) on delete cascade,
  rank int,
  url text,
  url_hash text,
  title text,
  snippet text,
  domain text,
  sentiment_label text,
  control_class text,
  finance_routine boolean,
  uncertain boolean,
  uncertain_reason text,
  llm_label text,
  llm_sentiment_label text,
  llm_risk_label text,
  llm_control_class text,
  llm_severity text,
  llm_reason text,
  model_score numeric,
  created_at timestamptz not null default now()
);

create index if not exists serp_results_run_idx on serp_results (serp_run_id);
create index if not exists serp_results_url_hash_idx on serp_results (url_hash);
create unique index if not exists serp_results_unique_idx
  on serp_results (serp_run_id, rank, url_hash);

create table if not exists company_article_mentions (
  id uuid primary key default gen_random_uuid(),
  company_id uuid not null references companies(id),
  article_id uuid not null references articles(id),
  model_sentiment_score numeric,
  sentiment_label text,
  model_relevant boolean,
  control_class text,
  finance_routine boolean,
  uncertain boolean,
  uncertain_reason text,
  llm_label text,
  llm_sentiment_label text,
  llm_risk_label text,
  llm_control_class text,
  llm_severity text,
  llm_reason text,
  model_version text,
  run_id text,
  scored_at timestamptz,
  unique (company_id, article_id)
);

create table if not exists ceo_article_mentions (
  id uuid primary key default gen_random_uuid(),
  ceo_id uuid not null references ceos(id),
  article_id uuid not null references articles(id),
  model_sentiment_score numeric,
  sentiment_label text,
  model_relevant boolean,
  control_class text,
  finance_routine boolean,
  uncertain boolean,
  uncertain_reason text,
  llm_label text,
  llm_sentiment_label text,
  llm_risk_label text,
  llm_control_class text,
  llm_severity text,
  llm_reason text,
  model_version text,
  run_id text,
  scored_at timestamptz,
  unique (ceo_id, article_id)
);

create table if not exists company_article_overrides (
  id uuid primary key default gen_random_uuid(),
  company_id uuid not null references companies(id),
  article_id uuid not null references articles(id),
  override_sentiment_score numeric,
  override_sentiment_label text,
  override_relevant boolean,
  override_control_class text,
  note text,
  edited_by text,
  edited_at timestamptz not null default now(),
  unique (company_id, article_id)
);

create table if not exists ceo_article_overrides (
  id uuid primary key default gen_random_uuid(),
  ceo_id uuid not null references ceos(id),
  article_id uuid not null references articles(id),
  override_sentiment_score numeric,
  override_sentiment_label text,
  override_relevant boolean,
  override_control_class text,
  note text,
  edited_by text,
  edited_at timestamptz not null default now(),
  unique (ceo_id, article_id)
);

create table if not exists serp_result_overrides (
  id uuid primary key default gen_random_uuid(),
  serp_result_id uuid not null references serp_results(id),
  override_sentiment_label text,
  override_control_class text,
  note text,
  edited_by text,
  edited_at timestamptz not null default now(),
  unique (serp_result_id)
);

create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  email text not null unique,
  role text not null check (role in ('internal','external')),
  created_at timestamptz not null default now()
);

create table if not exists user_company_access (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references users(id) on delete cascade,
  company_id uuid not null references companies(id) on delete cascade,
  created_at timestamptz not null default now(),
  unique (user_id, company_id)
);

create table if not exists boards (
  id uuid primary key default gen_random_uuid(),
  ceo_id uuid not null references ceos(id),
  company_id uuid references companies(id),
  url text,
  domain text,
  source text,
  last_updated timestamptz,
  created_at timestamptz not null default now(),
  unique (ceo_id, url)
);

create index if not exists boards_ceo_idx on boards (ceo_id);
