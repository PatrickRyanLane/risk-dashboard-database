create or replace view entity_daily_metrics_v as
with article as (
    select mv.date,
           mv.entity_type,
           mv.entity_id,
           mv.company_id,
           mv.ceo_id,
           mv.entity_name,
           mv.company,
           mv.ceo,
           mv.positive as article_positive_count,
           mv.neutral as article_neutral_count,
           mv.negative as article_negative_count,
           mv.total as article_total_count,
           mv.neg_pct as article_negative_pct
    from article_daily_counts_mv mv
),
serp as (
    select mv.date,
           mv.entity_type,
           mv.entity_id,
           mv.company_id,
           mv.ceo_id,
           mv.entity_name,
           mv.company,
           mv.ceo,
           mv.positive_serp as serp_positive_count,
           mv.neutral_serp as serp_neutral_count,
           mv.negative_serp as serp_negative_count,
           mv.total as serp_total_count,
           mv.controlled as serp_controlled_count,
           greatest(mv.total - mv.controlled, 0) as serp_uncontrolled_count
    from serp_daily_counts_mv mv
),
article_crisis as (
    select cad.date,
           'brand'::text as entity_type,
           cad.company_id as entity_id,
           cad.company_id,
           null::uuid as ceo_id,
           count(*) filter (where cam.llm_risk_label = 'crisis_risk') as crisis_risk_count
    from company_article_mentions_daily cad
    left join company_article_mentions cam
      on cam.company_id = cad.company_id and cam.article_id = cad.article_id
    group by cad.date, cad.company_id

    union all

    select cad.date,
           'ceo'::text as entity_type,
           cad.ceo_id as entity_id,
           ce.company_id,
           cad.ceo_id,
           count(*) filter (where cem.llm_risk_label = 'crisis_risk') as crisis_risk_count
    from ceo_article_mentions_daily cad
    join ceos ce on ce.id = cad.ceo_id
    left join ceo_article_mentions cem
      on cem.ceo_id = cad.ceo_id and cem.article_id = cad.article_id
    group by cad.date, ce.company_id, cad.ceo_id
),
top_stories_sentiment as (
    select fd.date,
           case when fd.entity_type in ('brand', 'company') then 'brand' else 'ceo' end as entity_type,
           fd.entity_id,
           case
               when fd.entity_type in ('brand', 'company') then fd.entity_id
               else ce.company_id
           end as company_id,
           case
               when fd.entity_type = 'ceo' then fd.entity_id
               else null::uuid
           end as ceo_id,
           coalesce(company_brand.name, company_ceo.name, fd.entity_name) as company,
           case when fd.entity_type = 'ceo' then coalesce(ce.name, fd.entity_name) else '' end as ceo,
           coalesce(
               case when fd.entity_type = 'ceo' then ce.name else company_brand.name end,
               fd.entity_name
           ) as entity_name,
           sum(fd.total_count) as top_stories_total_count,
           sum(fd.positive_count) as top_stories_positive_count,
           sum(fd.neutral_count) as top_stories_neutral_count,
           sum(fd.negative_count) as top_stories_negative_count
    from serp_feature_daily_mv fd
    left join companies company_brand
      on fd.entity_type in ('brand', 'company')
     and company_brand.id = fd.entity_id
    left join ceos ce
      on fd.entity_type = 'ceo'
     and ce.id = fd.entity_id
    left join companies company_ceo
      on company_ceo.id = ce.company_id
    where fd.feature_type = 'top_stories_items'
    group by fd.date,
             case when fd.entity_type in ('brand', 'company') then 'brand' else 'ceo' end,
             fd.entity_id,
             case
                 when fd.entity_type in ('brand', 'company') then fd.entity_id
                 else ce.company_id
             end,
             case
                 when fd.entity_type = 'ceo' then fd.entity_id
                 else null::uuid
             end,
             coalesce(company_brand.name, company_ceo.name, fd.entity_name),
             case when fd.entity_type = 'ceo' then coalesce(ce.name, fd.entity_name) else '' end,
             coalesce(
                 case when fd.entity_type = 'ceo' then ce.name else company_brand.name end,
                 fd.entity_name
             )
),
top_stories_control as (
    select fc.date,
           case when fc.entity_type in ('brand', 'company') then 'brand' else 'ceo' end as entity_type,
           fc.entity_id,
           sum(fc.controlled_count) as top_stories_controlled_count
    from serp_feature_control_daily_mv fc
    where fc.feature_type = 'top_stories_items'
    group by fc.date,
             case when fc.entity_type in ('brand', 'company') then 'brand' else 'ceo' end,
             fc.entity_id
),
entity_keys as (
    select date, entity_type, entity_id from article
    union
    select date, entity_type, entity_id from serp
    union
    select date, entity_type, entity_id from article_crisis
    union
    select date, entity_type, entity_id from top_stories_sentiment
)
select k.date,
       k.entity_type,
       k.entity_id,
       coalesce(a.company_id, s.company_id, ac.company_id, ts.company_id) as company_id,
       coalesce(a.ceo_id, s.ceo_id, ac.ceo_id, ts.ceo_id) as ceo_id,
       coalesce(nullif(a.entity_name, ''), nullif(s.entity_name, ''), nullif(ts.entity_name, ''), '') as entity_name,
       coalesce(nullif(a.company, ''), nullif(s.company, ''), nullif(ts.company, ''), '') as company,
       coalesce(nullif(a.ceo, ''), nullif(s.ceo, ''), nullif(ts.ceo, ''), '') as ceo,
       coalesce(a.article_positive_count, 0) as article_positive_count,
       coalesce(a.article_neutral_count, 0) as article_neutral_count,
       coalesce(a.article_negative_count, 0) as article_negative_count,
       coalesce(a.article_total_count, 0) as article_total_count,
       coalesce(a.article_negative_pct, 0) as article_negative_pct,
       coalesce(s.serp_positive_count, 0) as serp_positive_count,
       coalesce(s.serp_neutral_count, 0) as serp_neutral_count,
       coalesce(s.serp_negative_count, 0) as serp_negative_count,
       coalesce(s.serp_total_count, 0) as serp_total_count,
       coalesce(s.serp_controlled_count, 0) as serp_controlled_count,
       coalesce(s.serp_uncontrolled_count, 0) as serp_uncontrolled_count,
       coalesce(ts.top_stories_total_count, 0) as top_stories_total_count,
       coalesce(ts.top_stories_positive_count, 0) as top_stories_positive_count,
       coalesce(ts.top_stories_neutral_count, 0) as top_stories_neutral_count,
       coalesce(ts.top_stories_negative_count, 0) as top_stories_negative_count,
       coalesce(tc.top_stories_controlled_count, 0) as top_stories_controlled_count,
       greatest(coalesce(ts.top_stories_total_count, 0) - coalesce(tc.top_stories_controlled_count, 0), 0) as top_stories_uncontrolled_count,
       coalesce(ac.crisis_risk_count, 0) as crisis_risk_count
from entity_keys k
left join article a
  on a.date = k.date and a.entity_type = k.entity_type and a.entity_id = k.entity_id
left join serp s
  on s.date = k.date and s.entity_type = k.entity_type and s.entity_id = k.entity_id
left join article_crisis ac
  on ac.date = k.date and ac.entity_type = k.entity_type and ac.entity_id = k.entity_id
left join top_stories_sentiment ts
  on ts.date = k.date and ts.entity_type = k.entity_type and ts.entity_id = k.entity_id
left join top_stories_control tc
  on tc.date = k.date and tc.entity_type = k.entity_type and tc.entity_id = k.entity_id;
