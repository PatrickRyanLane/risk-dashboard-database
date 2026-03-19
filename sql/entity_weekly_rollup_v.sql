create or replace view entity_weekly_rollup_v as
select min(edm.date) over w as week_start,
       edm.entity_type,
       edm.entity_id,
       edm.company_id,
       edm.ceo_id,
       edm.entity_name,
       edm.company,
       edm.ceo,
       sum(edm.article_negative_count) over w as article_negative_7d,
       sum(edm.article_total_count) over w as article_total_7d,
       avg(edm.article_negative_pct) over w as article_negative_pct_avg_7d,
       sum(edm.serp_negative_count) over w as serp_negative_7d,
       sum(edm.serp_total_count) over w as serp_total_7d,
       sum(edm.serp_controlled_count) over w as serp_controlled_7d,
       sum(edm.serp_uncontrolled_count) over w as serp_uncontrolled_7d,
       sum(edm.top_stories_total_count) over w as top_stories_total_7d,
       sum(edm.top_stories_negative_count) over w as top_stories_negative_7d,
       sum(edm.top_stories_controlled_count) over w as top_stories_controlled_7d,
       sum(edm.top_stories_uncontrolled_count) over w as top_stories_uncontrolled_7d,
       sum(case when edm.top_stories_negative_count >= 4 then 1 else 0 end) over w as top_stories_crisis_days_7d,
       sum(edm.crisis_risk_count) over w as crisis_risk_7d,
       edm.date as window_end
from entity_daily_metrics_v edm
window w as (
    partition by edm.entity_type, edm.entity_id
    order by edm.date
    rows between 6 preceding and current row
);
