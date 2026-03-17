create or replace view entity_anomalies_v as
with baseline as (
    select edm.*,
           count(*) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 30 preceding and 1 preceding
           ) as prior_observation_days_30d,
           avg(edm.article_negative_count::numeric) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 30 preceding and 1 preceding
           ) as article_negative_baseline_30d,
           avg(edm.serp_uncontrolled_count::numeric) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 30 preceding and 1 preceding
           ) as serp_uncontrolled_baseline_30d,
           avg(edm.top_stories_negative_count::numeric) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 30 preceding and 1 preceding
           ) as top_stories_negative_baseline_30d,
           max(edm.top_stories_negative_count) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 7 preceding and 1 preceding
           ) as top_stories_prior_7d_max,
           max(edm.top_stories_negative_count) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 30 preceding and 8 preceding
           ) as top_stories_prior_30d_max,
           sum(case when edm.top_stories_negative_count >= 4 then 1 else 0 end) over (
               partition by edm.entity_type, edm.entity_id
               order by edm.date
               rows between 2 preceding and current row
           ) as top_stories_crisis_days_3d
    from entity_daily_metrics_v edm
)
select b.date,
       b.entity_type,
       b.entity_id,
       b.company_id,
       b.ceo_id,
       b.entity_name,
       b.company,
       b.ceo,
       'article_spike'::text as anomaly_type,
       greatest(b.article_negative_count - coalesce(b.article_negative_baseline_30d, 0), 0)::numeric as severity_score,
       b.article_negative_count::numeric as observed_value,
       coalesce(b.article_negative_baseline_30d, 0) as baseline_value,
       b.article_negative_count,
       b.serp_uncontrolled_count,
       b.top_stories_negative_count,
       'Negative article coverage is materially above the trailing 30-day baseline.'::text as summary
from baseline b
where b.article_negative_count >= 4
  and coalesce(b.prior_observation_days_30d, 0) >= 7
  and b.article_negative_count >= coalesce(b.article_negative_baseline_30d, 0) + 2
  and b.article_negative_count >= greatest(4, coalesce(b.article_negative_baseline_30d, 0) * 2)

union all

select b.date,
       b.entity_type,
       b.entity_id,
       b.company_id,
       b.ceo_id,
       b.entity_name,
       b.company,
       b.ceo,
       'serp_uncontrolled_spike'::text as anomaly_type,
       greatest(b.serp_uncontrolled_count - coalesce(b.serp_uncontrolled_baseline_30d, 0), 0)::numeric as severity_score,
       b.serp_uncontrolled_count::numeric as observed_value,
       coalesce(b.serp_uncontrolled_baseline_30d, 0) as baseline_value,
       b.article_negative_count,
       b.serp_uncontrolled_count,
       b.top_stories_negative_count,
       'Uncontrolled negative SERP results are materially above the trailing 30-day baseline.'::text as summary
from baseline b
where b.serp_uncontrolled_count >= 3
  and coalesce(b.prior_observation_days_30d, 0) >= 7
  and b.serp_uncontrolled_count >= coalesce(b.serp_uncontrolled_baseline_30d, 0) + 2
  and b.serp_uncontrolled_count >= greatest(3, coalesce(b.serp_uncontrolled_baseline_30d, 0) * 2)

union all

select b.date,
       b.entity_type,
       b.entity_id,
       b.company_id,
       b.ceo_id,
       b.entity_name,
       b.company,
       b.ceo,
       'top_stories_surge'::text as anomaly_type,
       greatest(b.top_stories_negative_count - coalesce(b.top_stories_negative_baseline_30d, 0), 0)::numeric as severity_score,
       b.top_stories_negative_count::numeric as observed_value,
       coalesce(b.top_stories_negative_baseline_30d, 0) as baseline_value,
       b.article_negative_count,
       b.serp_uncontrolled_count,
       b.top_stories_negative_count,
       'Negative Top Stories volume is materially above the trailing 30-day baseline.'::text as summary
from baseline b
where b.top_stories_negative_count >= 4
  and coalesce(b.prior_observation_days_30d, 0) >= 7
  and b.top_stories_negative_count >= coalesce(b.top_stories_negative_baseline_30d, 0) + 2
  and b.top_stories_negative_count >= greatest(4, coalesce(b.top_stories_negative_baseline_30d, 0) * 2)

union all

select b.date,
       b.entity_type,
       b.entity_id,
       b.company_id,
       b.ceo_id,
       b.entity_name,
       b.company,
       b.ceo,
       'sustained_top_stories'::text as anomaly_type,
       (b.top_stories_negative_count + b.top_stories_crisis_days_3d)::numeric as severity_score,
       b.top_stories_negative_count::numeric as observed_value,
       coalesce(b.top_stories_negative_baseline_30d, 0) as baseline_value,
       b.article_negative_count,
       b.serp_uncontrolled_count,
       b.top_stories_negative_count,
       'Negative Top Stories have persisted at crisis-level volume for multiple consecutive days.'::text as summary
from baseline b
where b.top_stories_negative_count >= 4
  and b.top_stories_crisis_days_3d >= 3

union all

select b.date,
       b.entity_type,
       b.entity_id,
       b.company_id,
       b.ceo_id,
       b.entity_name,
       b.company,
       b.ceo,
       'search_spillover'::text as anomaly_type,
       (b.top_stories_negative_count + b.serp_uncontrolled_count)::numeric as severity_score,
       (b.top_stories_negative_count + b.serp_uncontrolled_count)::numeric as observed_value,
       0::numeric as baseline_value,
       b.article_negative_count,
       b.serp_uncontrolled_count,
       b.top_stories_negative_count,
       'Negative coverage is now showing up in both Top Stories and broader search results.'::text as summary
from baseline b
where b.article_negative_count >= 3
  and b.top_stories_negative_count >= 4
  and b.serp_uncontrolled_count >= 2

union all

select b.date,
       b.entity_type,
       b.entity_id,
       b.company_id,
       b.ceo_id,
       b.entity_name,
       b.company,
       b.ceo,
       'resurfacing_top_stories'::text as anomaly_type,
       (b.top_stories_negative_count + 2)::numeric as severity_score,
       b.top_stories_negative_count::numeric as observed_value,
       0::numeric as baseline_value,
       b.article_negative_count,
       b.serp_uncontrolled_count,
       b.top_stories_negative_count,
       'Top Stories returned after at least a week of relative quiet.'::text as summary
from baseline b
where b.top_stories_negative_count >= 4
  and coalesce(b.top_stories_prior_7d_max, 0) = 0
  and coalesce(b.top_stories_prior_30d_max, 0) >= 4;
