create materialized view if not exists negative_articles_summary_mv as
with base as (
    select cad.date as date,
           c.id as company_id,
           c.name as company,
           coalesce(ceo.name, '') as ceo,
           coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
           a.title as title,
           m.llm_risk_label as llm_risk_label,
           'brand'::text as article_type
    from company_article_mentions_daily cad
    join company_article_mentions m
      on m.company_id = cad.company_id and m.article_id = cad.article_id
    join companies c on c.id = cad.company_id
    join articles a on a.id = cad.article_id
    left join company_article_overrides ov
      on ov.company_id = cad.company_id and ov.article_id = cad.article_id
    left join ceos ceo on ceo.company_id = c.id
    union all
    select cad.date as date,
           c.id as company_id,
           c.name as company,
           coalesce(ceo.name, '') as ceo,
           coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
           a.title as title,
           m.llm_risk_label as llm_risk_label,
           'ceo'::text as article_type
    from ceo_article_mentions_daily cad
    join ceo_article_mentions m
      on m.ceo_id = cad.ceo_id and m.article_id = cad.article_id
    join ceos ceo on ceo.id = cad.ceo_id
    join companies c on c.id = ceo.company_id
    join articles a on a.id = cad.article_id
    left join ceo_article_overrides ov
      on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
)
select date,
       company_id,
       company,
       ceo,
       article_type,
       count(*) filter (where sentiment = 'negative') as negative_count,
       count(*) filter (where llm_risk_label = 'crisis_risk') as crisis_risk_count,
       array_to_string(
           (array_agg(title order by title) filter (where sentiment = 'negative'))[1:3],
           ' | '
       ) as top_headlines
from base
group by date, company_id, company, ceo, article_type;

create unique index if not exists negative_articles_summary_mv_unique_idx
    on negative_articles_summary_mv (date, company_id, article_type, ceo);
