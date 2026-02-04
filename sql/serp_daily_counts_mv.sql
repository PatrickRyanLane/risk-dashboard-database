create materialized view if not exists serp_daily_counts_mv as
select sr.run_at::date as date,
       'brand'::text as entity_type,
       c.id as entity_id,
       c.id as company_id,
       null::uuid as ceo_id,
       c.name as entity_name,
       c.name as company,
       ''::text as ceo,
       count(*) as total,
       sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'controlled' then 1 else 0 end) as controlled,
       sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative' then 1 else 0 end) as negative_serp,
       sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'neutral' then 1 else 0 end) as neutral_serp,
       sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'positive' then 1 else 0 end) as positive_serp
from serp_runs sr
join companies c on c.id = sr.company_id
join serp_results r on r.serp_run_id = sr.id
left join serp_result_overrides ov on ov.serp_result_id = r.id
where sr.entity_type = 'company'
group by sr.run_at::date, c.id, c.name
union all
select sr.run_at::date as date,
       'ceo'::text as entity_type,
       ceo.id as entity_id,
       c.id as company_id,
       ceo.id as ceo_id,
       ceo.name as entity_name,
       c.name as company,
       ceo.name as ceo,
       count(*) as total,
       sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'controlled' then 1 else 0 end) as controlled,
       sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative' then 1 else 0 end) as negative_serp,
       sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'neutral' then 1 else 0 end) as neutral_serp,
       sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'positive' then 1 else 0 end) as positive_serp
from serp_runs sr
join ceos ceo on ceo.id = sr.ceo_id
join companies c on c.id = ceo.company_id
join serp_results r on r.serp_run_id = sr.id
left join serp_result_overrides ov on ov.serp_result_id = r.id
where sr.entity_type = 'ceo'
group by sr.run_at::date, ceo.id, ceo.name, c.id, c.name;

create unique index if not exists serp_daily_counts_mv_unique_idx
    on serp_daily_counts_mv (date, entity_type, entity_id);
