create materialized view if not exists serp_feature_control_daily_mv as
select sfi.date,
       sfi.entity_type,
       sfi.entity_id,
       sfi.entity_name,
       sfi.feature_type,
       count(*) filter (
           where coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) is not null
       ) as total_count,
       sum(
           case when coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'controlled'
                then 1 else 0 end
       ) as controlled_count
from serp_feature_items sfi
left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
group by sfi.date, sfi.entity_type, sfi.entity_id, sfi.entity_name, sfi.feature_type;

create unique index if not exists serp_feature_control_daily_mv_unique_idx
    on serp_feature_control_daily_mv (date, entity_type, entity_id, feature_type);

create index if not exists serp_feature_control_daily_mv_entity_name_date_idx
    on serp_feature_control_daily_mv (entity_type, entity_name, date);

create index if not exists serp_feature_control_daily_mv_date_idx
    on serp_feature_control_daily_mv (date);

create index if not exists serp_feature_control_daily_mv_entity_feature_date_idx
    on serp_feature_control_daily_mv (entity_type, feature_type, date);
