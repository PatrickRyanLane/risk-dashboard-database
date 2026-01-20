import csv
from datetime import datetime, timezone

from psycopg2.extras import execute_values

from src.url_utils import normalize_url, url_hash


def upsert_companies_ceos(conn, roster_file_obj):
    reader = csv.DictReader(roster_file_obj)
    companies_map = {}
    ceos_map = {}

    for row in reader:
        company = (row.get('Company') or row.get('company') or '').strip()
        if not company:
            continue
        ticker = (row.get('Stock') or row.get('stock') or '').strip()
        sector = (row.get('Sector') or row.get('sector') or '').strip()
        websites = (row.get('Websites') or row.get('websites') or row.get('Website') or row.get('website') or '').strip()
        companies_map[company] = (company, ticker, sector, websites)

    companies = list(companies_map.values())
    if companies:
        sql = """
            insert into companies (name, ticker, sector, websites)
            values %s
            on conflict (name) do update set
              ticker = excluded.ticker,
              sector = excluded.sector,
              websites = excluded.websites
        """
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, companies, page_size=1000)

    roster_file_obj.seek(0)
    reader = csv.DictReader(roster_file_obj)

    for row in reader:
        company = (row.get('Company') or row.get('company') or '').strip()
        ceo = (row.get('CEO') or row.get('ceo') or '').strip()
        alias = (row.get('CEO Alias') or row.get('ceo alias') or row.get('alias') or '').strip()
        if not company or not ceo:
            continue
        ceos_map[(ceo, company)] = (ceo, company, alias)

    ceos = list(ceos_map.values())
    if ceos:
        sql = """
            insert into ceos (name, company_id, alias)
            select v.ceo, c.id, v.alias
            from (values %s) as v(ceo, company, alias)
            join companies c on c.name = v.company
            on conflict (name, company_id) do update set
              alias = excluded.alias
        """
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, ceos, page_size=1000)


def fetch_company_map(conn):
    with conn.cursor() as cur:
        cur.execute("select name, id from companies")
        return {name: cid for name, cid in cur.fetchall()}


def fetch_ceo_map(conn):
    with conn.cursor() as cur:
        cur.execute("select name, company_id, id from ceos")
        return {(name, company_id): cid for name, company_id, cid in cur.fetchall()}


def upsert_articles(conn, rows):
    if not rows:
        return
    sql = """
        insert into articles (canonical_url, title, publisher, snippet, published_at, first_seen_at, last_seen_at, source)
        values %s
        on conflict (canonical_url) do update set
          title = coalesce(excluded.title, articles.title),
          publisher = coalesce(excluded.publisher, articles.publisher),
          snippet = coalesce(excluded.snippet, articles.snippet),
          published_at = coalesce(excluded.published_at, articles.published_at),
          last_seen_at = excluded.last_seen_at
    """
    with conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)


def fetch_article_map(conn, urls):
    if not urls:
        return {}
    sql = "select canonical_url, id from articles where canonical_url = any(%s)"
    with conn.cursor() as cur:
        cur.execute(sql, (list(urls),))
        return {u: aid for u, aid in cur.fetchall()}


def ingest_article_mentions(conn, file_obj, entity_type, date_str):
    reader = csv.DictReader(file_obj)
    now = datetime.now(timezone.utc)

    articles = {}
    mentions = []
    article_urls = []

    company_map = fetch_company_map(conn)
    ceo_map = fetch_ceo_map(conn)

    for row in reader:
        title = (row.get('title') or '').strip()
        url = (row.get('url') or '').strip()
        if not title or not url:
            continue
        canonical = normalize_url(url)
        if not canonical:
            continue

        publisher = (row.get('source') or '').strip()
        sentiment = (row.get('sentiment') or '').strip().lower() or None

        articles[canonical] = (canonical, title, publisher, None, None, now, now, 'google_rss')
        article_urls.append(canonical)

        if entity_type == 'company':
            company = (row.get('company') or '').strip()
            company_id = company_map.get(company)
            if not company_id:
                continue
            mentions.append((company_id, canonical, sentiment, date_str))
        else:
            ceo = (row.get('ceo') or '').strip()
            company = (row.get('company') or '').strip()
            company_id = company_map.get(company)
            if not company_id:
                continue
            ceo_id = ceo_map.get((ceo, company_id))
            if not ceo_id:
                continue
            mentions.append((ceo_id, canonical, sentiment, date_str))

    upsert_articles(conn, list(articles.values()))
    article_map = fetch_article_map(conn, article_urls)

    scored_at = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    if entity_type == 'company':
        insert_rows = []
        for company_id, canonical, sentiment, dstr in mentions:
            article_id = article_map.get(canonical)
            if not article_id:
                continue
            insert_rows.append((company_id, article_id, sentiment, scored_at, 'vader'))
        if insert_rows:
            sql = """
                insert into company_article_mentions (company_id, article_id, sentiment_label, scored_at, model_version)
                values %s
                on conflict (company_id, article_id) do update set
                  sentiment_label = excluded.sentiment_label,
                  scored_at = excluded.scored_at,
                  model_version = excluded.model_version
            """
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, insert_rows, page_size=1000)
    else:
        insert_rows = []
        for ceo_id, canonical, sentiment, dstr in mentions:
            article_id = article_map.get(canonical)
            if not article_id:
                continue
            insert_rows.append((ceo_id, article_id, sentiment, scored_at, 'vader'))
        if insert_rows:
            sql = """
                insert into ceo_article_mentions (ceo_id, article_id, sentiment_label, scored_at, model_version)
                values %s
                on conflict (ceo_id, article_id) do update set
                  sentiment_label = excluded.sentiment_label,
                  scored_at = excluded.scored_at,
                  model_version = excluded.model_version
            """
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, insert_rows, page_size=1000)


def ingest_serp_results(conn, file_obj, entity_type, date_str):
    reader = csv.DictReader(file_obj)
    now = datetime.now(timezone.utc)

    company_map = fetch_company_map(conn)
    ceo_map = fetch_ceo_map(conn)

    run_rows = {}
    result_rows = []

    for row in reader:
        company = (row.get('company') or '').strip()
        ceo = (row.get('ceo') or '').strip()
        title = (row.get('title') or '').strip()
        url = (row.get('url') or row.get('link') or '').strip()
        if not title or not url:
            continue

        rank = row.get('position')
        try:
            rank = int(float(rank)) if rank not in (None, '') else None
        except Exception:
            rank = None

        canonical = normalize_url(url)
        domain = ''
        try:
            from urllib.parse import urlparse
            domain = (urlparse(url).hostname or '').replace('www.', '')
        except Exception:
            domain = ''

        sentiment = (row.get('sentiment') or '').strip().lower() or None
        controlled = (row.get('controlled') or '').strip().lower()
        control_class = None
        if controlled in {'true', '1', 'controlled'}:
            control_class = 'controlled'
        elif controlled in {'false', '0', 'uncontrolled'}:
            control_class = 'uncontrolled'

        if entity_type == 'company':
            company_id = company_map.get(company)
            if not company_id:
                continue
            run_key = ('company', company_id)
        else:
            company_id = company_map.get(company)
            if not company_id:
                continue
            ceo_id = ceo_map.get((ceo, company_id))
            if not ceo_id:
                continue
            run_key = ('ceo', ceo_id)

        if run_key not in run_rows:
            run_rows[run_key] = {
                'entity_type': run_key[0],
                'company_id': company_id if run_key[0] == 'company' else None,
                'ceo_id': run_key[1] if run_key[0] == 'ceo' else None,
                'query_text': company if run_key[0] == 'company' else f"{ceo} {company}",
                'provider': 'google_serp',
                'run_at': datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            }

        result_rows.append((run_key, rank, url, url_hash(canonical), title, row.get('snippet') or '', domain, sentiment, control_class))

    # Insert runs
    if run_rows:
        run_values = []
        for key, data in run_rows.items():
            run_values.append((data['entity_type'], data['company_id'], data['ceo_id'], data['query_text'], data['provider'], data['run_at']))
        sql = """
            insert into serp_runs (entity_type, company_id, ceo_id, query_text, provider, run_at)
            values %s
            on conflict (entity_type, company_id, ceo_id, run_at) do update set
              query_text = excluded.query_text,
              provider = excluded.provider
            returning id, entity_type, company_id, ceo_id
        """
        run_id_map = {}
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, run_values, page_size=1000)
                rows = cur.fetchall()
                for run_id, etype, company_id, ceo_id in rows:
                    if etype == 'company':
                        run_id_map[('company', company_id)] = run_id
                    else:
                        run_id_map[('ceo', ceo_id)] = run_id
    else:
        run_id_map = {}

    if result_rows:
        insert_rows = []
        for run_key, rank, url, uhash, title, snippet, domain, sentiment, control_class in result_rows:
            run_id = run_id_map.get(run_key)
            if not run_id:
                continue
            insert_rows.append((run_id, rank, url, uhash, title, snippet, domain, sentiment, control_class))

        if insert_rows:
            sql = """
                insert into serp_results (serp_run_id, rank, url, url_hash, title, snippet, domain, sentiment_label, control_class)
                values %s
                on conflict (serp_run_id, rank, url_hash) do update set
                  url = excluded.url,
                  title = excluded.title,
                  snippet = excluded.snippet,
                  domain = excluded.domain,
                  sentiment_label = excluded.sentiment_label,
                  control_class = excluded.control_class
            """
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, insert_rows, page_size=1000)
