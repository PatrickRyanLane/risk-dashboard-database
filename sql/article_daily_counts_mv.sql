create materialized view if not exists article_daily_counts_mv as
select cad.date as date,
       'brand'::text as entity_type,
       c.id as entity_id,
       c.id as company_id,
       null::uuid as ceo_id,
       c.name as entity_name,
       c.name as company,
       ''::text as ceo,
       ''::text as alias,
       sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'positive' then 1 else 0 end) as positive,
       sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'neutral' then 1 else 0 end) as neutral,
       sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'negative' then 1 else 0 end) as negative,
       count(*) as total,
       case when count(*) > 0
            then round((sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 6)
            else 0 end as neg_pct
from company_article_mentions_daily cad
join companies c on c.id = cad.company_id
left join company_article_overrides ov on ov.company_id = cad.company_id and ov.article_id = cad.article_id
group by cad.date, c.id, c.name
union all
select cad.date as date,
       'ceo'::text as entity_type,
       ceo.id as entity_id,
       c.id as company_id,
       ceo.id as ceo_id,
       ceo.name as entity_name,
       c.name as company,
       ceo.name as ceo,
       coalesce(ceo.alias, '') as alias,
       sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'positive' then 1 else 0 end) as positive,
       sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'neutral' then 1 else 0 end) as neutral,
       sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'negative' then 1 else 0 end) as negative,
       count(*) as total,
       case when count(*) > 0
            then round((sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 1)
            else 0 end as neg_pct
from ceo_article_mentions_daily cad
join ceos ceo on ceo.id = cad.ceo_id
join companies c on c.id = ceo.company_id
left join ceo_article_overrides ov on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
group by cad.date, ceo.id, ceo.name, ceo.alias, c.id, c.name;

create unique index if not exists article_daily_counts_mv_unique_idx
    on article_daily_counts_mv (date, entity_type, entity_id);
