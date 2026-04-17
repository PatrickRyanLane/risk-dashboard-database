create materialized view if not exists serp_feature_control_daily_index_mv as
select sfi.date,
       sfi.entity_type,
       sfi.feature_type,
       count(*) filter (
           where coalesce(ov.override_control_class, uov.override_control_class, sfi.llm_control_class, sfi.control_class) is not null
       ) as total_count,
       sum(
           case when coalesce(ov.override_control_class, uov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'controlled'
                then 1 else 0 end
       ) as controlled_count
from serp_feature_items sfi
left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
left join serp_feature_url_overrides uov on uov.entity_type = sfi.entity_type and uov.entity_id = sfi.entity_id and uov.feature_type = sfi.feature_type and uov.url_hash = sfi.url_hash
group by sfi.date, sfi.entity_type, sfi.feature_type;

create unique index if not exists serp_feature_control_daily_index_mv_unique_idx
    on serp_feature_control_daily_index_mv (date, entity_type, feature_type);

create index if not exists serp_feature_control_daily_index_mv_date_idx
    on serp_feature_control_daily_index_mv (date);
