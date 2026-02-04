create materialized view if not exists serp_feature_daily_mv as
select sfi.date,
       sfi.entity_type,
       sfi.entity_id,
       sfi.entity_name,
       sfi.feature_type,
       count(*) as total_count,
       sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'positive' then 1 else 0 end) as positive_count,
       sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'neutral' then 1 else 0 end) as neutral_count,
       sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative' then 1 else 0 end) as negative_count
from serp_feature_items sfi
left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
group by sfi.date, sfi.entity_type, sfi.entity_id, sfi.entity_name, sfi.feature_type;

create unique index if not exists serp_feature_daily_mv_unique_idx
    on serp_feature_daily_mv (date, entity_type, entity_id, feature_type);
