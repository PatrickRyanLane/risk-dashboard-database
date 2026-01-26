#!/usr/bin/env python3
"""
DB-backed dashboard API + static assets.
Internal service should be protected by IAP; external service read-only.
"""

import csv
import io
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import psycopg2
from flask import Flask, Response, jsonify, request, send_from_directory, abort
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

app = Flask(__name__, static_folder='static', static_url_path='/static')

PORT = int(os.environ.get('PORT', 8080))
DB_DSN = os.environ.get('DATABASE_URL')
DEFAULT_VIEW = os.environ.get('DEFAULT_VIEW', 'external')
PUBLIC_MODE = os.environ.get('PUBLIC_MODE', 'false').lower() in {'1', 'true', 'yes'}
ALLOW_EDITS = os.environ.get('ALLOW_EDITS', 'true').lower() in {'1', 'true', 'yes'}
IAP_AUDIENCE = os.environ.get('IAP_AUDIENCE', '')
ALLOWED_DOMAIN = os.environ.get('ALLOWED_DOMAIN', '')
ALLOWED_EMAILS = {e.strip().lower() for e in os.environ.get('ALLOWED_EMAILS', '').split(',') if e.strip()}
ALLOW_UNAUTHED_INTERNAL = os.environ.get('ALLOW_UNAUTHED_INTERNAL', 'false').lower() in {'1', 'true', 'yes'}
EXTERNAL_COMPANY_SCOPE = [c.strip() for c in os.environ.get('EXTERNAL_COMPANY_SCOPE', '').split(',') if c.strip()]
IAP_CERTS_URL = os.environ.get('IAP_CERTS_URL', 'https://www.gstatic.com/iap/verify/public_key')

_api_cache = {}
_api_cache_ttl = int(os.environ.get('API_CACHE_TTL', '120'))

DATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-(brand|ceo)-(articles|serps)-(modal|table)\.csv$')
STOCK_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-stock-data\.csv$')
TRENDS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-trends-data\.csv$')


# --------------------------- Auth helpers ---------------------------

def get_iap_email() -> str:
    if ALLOW_UNAUTHED_INTERNAL:
        return 'local-dev'
    token = request.headers.get('X-Goog-IAP-JWT-Assertion')
    if not token:
        return ''
    try:
        req = google_requests.Request()
        payload = id_token.verify_token(token, req, audience=IAP_AUDIENCE, certs_url=IAP_CERTS_URL)
        return payload.get('email', '')
    except Exception:
        return ''


def require_internal_user() -> str:
    email = get_iap_email()
    if not email:
        return ''
    email = email.lower()
    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        return ''
    if ALLOWED_DOMAIN and not email.endswith('@' + ALLOWED_DOMAIN.lower()):
        return ''
    return email


def get_request_email() -> str:
    if PUBLIC_MODE:
        return ''
    email = get_iap_email()
    return (email or '').lower()


def current_view() -> str:
    if PUBLIC_MODE:
        return 'external'
    return DEFAULT_VIEW if DEFAULT_VIEW in {'internal', 'external'} else 'internal'


def require_internal_access():
    if PUBLIC_MODE or ALLOW_UNAUTHED_INTERNAL:
        return
    if not require_internal_user():
        abort(403)


def get_company_scope_ids() -> List[str]:
    if PUBLIC_MODE and EXTERNAL_COMPANY_SCOPE:
        rows = query_rows(
            "select id from companies where name = any(%s)",
            (EXTERNAL_COMPANY_SCOPE,)
        )
        return [r[0] for r in rows]

    email = get_request_email()
    if not email:
        return []

    rows = query_rows(
        """
        select u.role, c.id
        from users u
        left join user_company_access uca on u.id = uca.user_id
        left join companies c on c.id = uca.company_id
        where lower(u.email) = %s
        """,
        (email,)
    )
    if not rows:
        return []
    roles = {r[0] for r in rows if r[0]}
    if 'internal' in roles:
        return []
    return [r[1] for r in rows if r[1]]


def scope_clause(column: str, params: List):
    scope_ids = get_company_scope_ids()
    if not scope_ids:
        return "", params
    params.append(scope_ids)
    return f" and {column} = any(%s)", params


# --------------------------- DB helpers ---------------------------

def get_conn():
    if not DB_DSN:
        raise RuntimeError('DATABASE_URL is required')
    return psycopg2.connect(DB_DSN)


def rows_to_csv(headers: List[str], rows: Iterable[Tuple]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def query_rows(sql: str, params: Tuple = ()) -> List[Tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def query_dict(sql: str, params: Tuple = ()) -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def serialize_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (datetime,)):
                clean[k] = v.isoformat()
            elif hasattr(v, 'isoformat'):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        out.append(clean)
    return out


def get_cached_json(key: str):
    now = datetime.utcnow().timestamp()
    cached = _api_cache.get(key)
    if not cached:
        return None
    data, ts = cached
    if now - ts > _api_cache_ttl:
        _api_cache.pop(key, None)
        return None
    return data


def set_cached_json(key: str, data):
    _api_cache[key] = (data, datetime.utcnow().timestamp())


# --------------------------- Static routes ---------------------------

@app.route('/')
def root():
    view = current_view()
    if view == 'internal':
        require_internal_access()
    return send_from_directory(f'static/{view}', 'dashboard.html')


@app.route('/brand-dashboard.html')
@app.route('/ceo-dashboard.html')
@app.route('/sectors.html')
def top_level_dashboards():
    view = current_view()
    if view == 'internal':
        require_internal_access()
    filename = request.path.lstrip('/')
    return send_from_directory(f'static/{view}', filename)


@app.route('/images/<path:path>')
def top_level_images(path):
    view = current_view()
    if view == 'internal':
        require_internal_access()
    return send_from_directory(f'static/{view}/images', path)


@app.route('/internal/')
@app.route('/internal/<path:path>')
def internal_static(path='dashboard.html'):
    if PUBLIC_MODE:
        abort(404)
    require_internal_access()
    return send_from_directory('static/internal', path)


@app.route('/external/')
@app.route('/external/<path:path>')
def external_static(path='dashboard.html'):
    return send_from_directory('static/external', path)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})



# --------------------------- Data endpoints ---------------------------

@app.route('/api/data/<path:filepath>')
def api_data(filepath):
    # Daily counts
    if filepath == 'daily_counts/brand-articles-daily-counts-chart.csv':
        return brand_articles_daily_counts()
    if filepath == 'daily_counts/ceo-articles-daily-counts-chart.csv':
        return ceo_articles_daily_counts()
    if filepath == 'daily_counts/brand-serps-daily-counts-chart.csv':
        return brand_serps_daily_counts()
    if filepath == 'daily_counts/ceo-serps-daily-counts-chart.csv':
        return ceo_serps_daily_counts()
    if filepath == 'daily_counts/negative-articles-summary.csv':
        return negative_articles_summary()

    # Roster
    if filepath == 'rosters/main-roster.csv':
        return roster_csv()

    # Processed articles / serps
    if filepath.startswith('processed_articles/'):
        name = filepath.split('/', 1)[1]
        return processed_articles_csv(name)
    if filepath.startswith('processed_serps/'):
        name = filepath.split('/', 1)[1]
        return processed_serps_csv(name)

    # Stock / trends
    if filepath.startswith('stock_prices/'):
        name = filepath.split('/', 1)[1]
        return stock_data_csv(name)
    if filepath.startswith('trends_data/'):
        name = filepath.split('/', 1)[1]
        return trends_data_csv(name)

    return jsonify({'error': f'Unknown path: {filepath}'}), 404


@app.route('/api/dates')
def available_dates():
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_rows(
        f"""
        select distinct cam.scored_at::date as date
        from company_article_mentions cam
        join companies c on c.id = cam.company_id
        where cam.scored_at is not null {scope_sql}
        order by date desc
        """,
        tuple(params),
    )
    dates = [r[0].isoformat() for r in rows]
    return jsonify({'dates': dates, 'count': len(dates), 'latest': dates[0] if dates else None})


# --------------------------- JSON endpoints (v1) ---------------------------

@app.route('/api/v1/daily_counts')
def daily_counts_json():
    kind = request.args.get('kind', '')
    cache_key = f"daily_counts:{kind}:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    if kind == 'brand_articles':
        params = []
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select cam.scored_at::date as date, c.name as company,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='positive' then 1 else 0 end) as positive,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='neutral' then 1 else 0 end) as neutral,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end) as negative,
              count(*) as total,
              case when count(*) > 0
                then round((sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 6)
                else 0 end as neg_pct
            from company_article_mentions cam
            join companies c on c.id = cam.company_id
            left join company_article_overrides ov on ov.company_id = cam.company_id and ov.article_id = cam.article_id
            where cam.scored_at is not null {scope_sql}
            group by cam.scored_at::date, c.name
            order by cam.scored_at::date, c.name
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    if kind == 'ceo_articles':
        params = []
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select cam.scored_at::date as date, ceo.name as ceo, c.name as company,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='positive' then 1 else 0 end) as positive,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='neutral' then 1 else 0 end) as neutral,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end) as negative,
              count(*) as total,
              case when count(*) > 0
                then round((sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 1)
                else 0 end as neg_pct,
              '' as theme,
              coalesce(ceo.alias, '') as alias
            from ceo_article_mentions cam
            join ceos ceo on ceo.id = cam.ceo_id
            join companies c on c.id = ceo.company_id
            left join ceo_article_overrides ov on ov.ceo_id = cam.ceo_id and ov.article_id = cam.article_id
            where cam.scored_at is not null {scope_sql}
            group by cam.scored_at::date, ceo.name, c.name, ceo.alias
            order by cam.scored_at::date, ceo.name
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    if kind == 'brand_serps':
        params = []
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select sr.run_at::date as date, c.name as company,
              count(*) as total,
              sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class)='controlled' then 1 else 0 end) as controlled,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='negative' then 1 else 0 end) as negative_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='neutral' then 1 else 0 end) as neutral_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='positive' then 1 else 0 end) as positive_serp
            from serp_runs sr
            join companies c on c.id = sr.company_id
            join serp_results r on r.serp_run_id = sr.id
            left join serp_result_overrides ov on ov.serp_result_id = r.id
            where sr.entity_type = 'company' {scope_sql}
            group by sr.run_at::date, c.name
            order by sr.run_at::date, c.name
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    if kind == 'ceo_serps':
        params = []
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select sr.run_at::date as date, ceo.name as ceo, c.name as company,
              count(*) as total,
              sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class)='controlled' then 1 else 0 end) as controlled,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='negative' then 1 else 0 end) as negative_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='neutral' then 1 else 0 end) as neutral_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='positive' then 1 else 0 end) as positive_serp
            from serp_runs sr
            join ceos ceo on ceo.id = sr.ceo_id
            join companies c on c.id = ceo.company_id
            join serp_results r on r.serp_run_id = sr.id
            left join serp_result_overrides ov on ov.serp_result_id = r.id
            where sr.entity_type = 'ceo' {scope_sql}
            group by sr.run_at::date, ceo.name, c.name
            order by sr.run_at::date, ceo.name
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    return jsonify({'error': 'invalid kind'}), 400


@app.route('/api/v1/processed_articles')
def processed_articles_json():
    dstr = request.args.get('date', '')
    entity = request.args.get('entity', 'brand')
    kind = request.args.get('kind', 'modal')
    filename = f"{dstr}-{entity}-articles-{kind}.csv"
    resp = processed_articles_csv(filename)
    if resp.status_code != 200:
        return resp
    rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
    return jsonify(rows)


@app.route('/api/v1/processed_serps')
def processed_serps_json():
    dstr = request.args.get('date', '')
    entity = request.args.get('entity', 'brand')
    kind = request.args.get('kind', 'modal')
    filename = f"{dstr}-{entity}-serps-{kind}.csv"
    resp = processed_serps_csv(filename)
    if resp.status_code != 200:
        return resp
    rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
    return jsonify(rows)


@app.route('/api/v1/serp_features')
def serp_features_json():
    entity = request.args.get('entity', 'brand')
    days = int(request.args.get('days', '90') or 90)
    entity_type = 'company' if entity == 'brand' else 'ceo'

    params = [entity_type, days]
    if entity_type == 'company':
        scope_sql, params = scope_clause("c.id", params)
        sql = f"""
            select s.date, s.entity_name, s.feature_type,
                   s.total_count, s.positive_count, s.neutral_count, s.negative_count
            from serp_feature_daily s
            join companies c on c.id = s.entity_id
            where s.entity_type = %s
              and s.date >= (current_date - (%s || ' days')::interval)
              {scope_sql}
            order by s.date, s.entity_name, s.feature_type
        """
    else:
        scope_sql, params = scope_clause("c.id", params)
        sql = f"""
            select s.date, s.entity_name, s.feature_type,
                   s.total_count, s.positive_count, s.neutral_count, s.negative_count
            from serp_feature_daily s
            join ceos ceo on ceo.id = s.entity_id
            join companies c on c.id = ceo.company_id
            where s.entity_type = %s
              and s.date >= (current_date - (%s || ' days')::interval)
              {scope_sql}
            order by s.date, s.entity_name, s.feature_type
        """
    rows = query_rows(sql, tuple(params))
    return jsonify(rows)


@app.route('/api/v1/roster')
def roster_json():
    cache_key = "roster"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    resp = roster_csv()
    if resp.status_code != 200:
        return resp
    rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
    set_cached_json(cache_key, rows)
    return jsonify(rows)


@app.route('/api/v1/negative_summary')
def negative_summary_json():
    cache_key = f"negative_summary:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    resp = negative_articles_summary()
    if resp.status_code != 200:
        return resp
    rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
    set_cached_json(cache_key, rows)
    return jsonify(rows)


@app.route('/api/v1/boards')
def boards_json():
    cache_key = "boards"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_dict(
        f"""
        select ceo.name as ceo, c.name as company, b.url, b.domain, b.source, b.last_updated
        from boards b
        join ceos ceo on ceo.id = b.ceo_id
        join companies c on c.id = ceo.company_id
        where 1=1 {scope_sql}
        order by ceo.name, b.domain
        """,
        tuple(params),
    )
    data = serialize_rows(rows)
    set_cached_json(cache_key, data)
    return jsonify(data)


@app.route('/api/v1/stock_data')
def stock_data_json():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'date is required (YYYY-MM-DD)'}), 400
    cache_key = f"stock_data:{date_str}:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    target = datetime.strptime(date_str, '%Y-%m-%d').date()
    rows = build_stock_rows(target)
    set_cached_json(cache_key, rows)
    return jsonify(rows)


@app.route('/api/v1/trends_data')
def trends_data_json():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'date is required (YYYY-MM-DD)'}), 400
    cache_key = f"trends_data:{date_str}:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    target = datetime.strptime(date_str, '%Y-%m-%d').date()
    rows = build_trends_rows(target)
    set_cached_json(cache_key, rows)
    return jsonify(rows)


@app.after_request
def add_cache_headers(response):
    if request.method == 'GET' and request.path.startswith('/api/') and not request.path.startswith('/api/internal/'):
        response.headers.setdefault('Cache-Control', f'public, max-age={_api_cache_ttl}')
    return response


# --------------------------- Internal edit endpoint ---------------------------

@app.route('/api/internal/overrides', methods=['POST'])
def apply_override():
    if PUBLIC_MODE or not ALLOW_EDITS:
        return jsonify({'error': 'editing disabled'}), 403
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401

    payload = request.get_json(force=True, silent=True) or {}
    mention_type = (payload.get('mention_type') or '').strip()
    mention_id = (payload.get('mention_id') or '').strip()
    serp_result_id = (payload.get('serp_result_id') or '').strip()
    sentiment_override = payload.get('sentiment_override')
    control_override = payload.get('control_override')
    relevant_override = payload.get('relevant_override')
    note = payload.get('note') or 'dashboard edit'

    if mention_type in {'company_article', 'ceo_article'} and not mention_id:
        return jsonify({'error': 'mention_id is required'}), 400
    if mention_type == 'serp_result' and not serp_result_id:
        return jsonify({'error': 'serp_result_id is required'}), 400

    if sentiment_override not in (None, 'positive', 'neutral', 'negative', 'risk', 'no_risk'):
        return jsonify({'error': 'invalid sentiment_override'}), 400
    if control_override not in (None, 'controlled', 'uncontrolled'):
        return jsonify({'error': 'invalid control_override'}), 400

    with get_conn() as conn:
        with conn.cursor() as cur:
            if mention_type == 'company_article':
                cur.execute(
                    """
                    insert into company_article_overrides
                      (company_id, article_id, override_sentiment_label, override_relevant, override_control_class, note, edited_by)
                    select company_id, article_id, %s, %s, %s, %s, %s
                    from company_article_mentions
                    where id = %s
                    on conflict (company_id, article_id) do update set
                      override_sentiment_label = excluded.override_sentiment_label,
                      override_relevant = excluded.override_relevant,
                      override_control_class = excluded.override_control_class,
                      note = excluded.note,
                      edited_by = excluded.edited_by,
                      edited_at = now()
                    returning id
                    """,
                    (sentiment_override, relevant_override, control_override, note, user_email, mention_id),
                )
            elif mention_type == 'ceo_article':
                cur.execute(
                    """
                    insert into ceo_article_overrides
                      (ceo_id, article_id, override_sentiment_label, override_relevant, override_control_class, note, edited_by)
                    select ceo_id, article_id, %s, %s, %s, %s, %s
                    from ceo_article_mentions
                    where id = %s
                    on conflict (ceo_id, article_id) do update set
                      override_sentiment_label = excluded.override_sentiment_label,
                      override_relevant = excluded.override_relevant,
                      override_control_class = excluded.override_control_class,
                      note = excluded.note,
                      edited_by = excluded.edited_by,
                      edited_at = now()
                    returning id
                    """,
                    (sentiment_override, relevant_override, control_override, note, user_email, mention_id),
                )
            elif mention_type == 'serp_result':
                cur.execute(
                    """
                    insert into serp_result_overrides
                      (serp_result_id, override_sentiment_label, override_control_class, note, edited_by)
                    values (%s, %s, %s, %s, %s)
                    on conflict (serp_result_id) do update set
                      override_sentiment_label = excluded.override_sentiment_label,
                      override_control_class = excluded.override_control_class,
                      note = excluded.note,
                      edited_by = excluded.edited_by,
                      edited_at = now()
                    returning id
                    """,
                    (serp_result_id, sentiment_override, control_override, note, user_email),
                )
            else:
                return jsonify({'error': 'invalid mention_type'}), 400
            row = cur.fetchone()

    return jsonify({'status': 'ok', 'id': row[0] if row else None})


# --------------------------- CSV builders ---------------------------


def brand_articles_daily_counts():
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_rows(
        f"""
        select cam.scored_at::date as date, c.name as company,
          sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='positive' then 1 else 0 end) as positive,
          sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='neutral' then 1 else 0 end) as neutral,
          sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end) as negative,
          count(*) as total,
          case when count(*) > 0
            then round((sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 6)
            else 0 end as neg_pct
        from company_article_mentions cam
        join companies c on c.id = cam.company_id
        left join company_article_overrides ov on ov.company_id = cam.company_id and ov.article_id = cam.article_id
        where cam.scored_at is not null {scope_sql}
        group by cam.scored_at::date, c.name
        order by cam.scored_at::date, c.name
        """,
        tuple(params),
    )
    csv_text = rows_to_csv(['date','company','positive','neutral','negative','total','neg_pct'], rows)
    return Response(csv_text, content_type='text/csv')


def ceo_articles_daily_counts():
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_rows(
        f"""
        select cam.scored_at::date as date, ceo.name as ceo, c.name as company,
          sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='positive' then 1 else 0 end) as positive,
          sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='neutral' then 1 else 0 end) as neutral,
          sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end) as negative,
          count(*) as total,
          case when count(*) > 0
            then round((sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 1)
            else 0 end as neg_pct,
          '' as theme,
          coalesce(ceo.alias, '') as alias
        from ceo_article_mentions cam
        join ceos ceo on ceo.id = cam.ceo_id
        join companies c on c.id = ceo.company_id
        left join ceo_article_overrides ov on ov.ceo_id = cam.ceo_id and ov.article_id = cam.article_id
        where cam.scored_at is not null {scope_sql}
        group by cam.scored_at::date, ceo.name, c.name, ceo.alias
        order by cam.scored_at::date, ceo.name
        """,
        tuple(params),
    )
    csv_text = rows_to_csv(['date','ceo','company','positive','neutral','negative','total','neg_pct','theme','alias'], rows)
    return Response(csv_text, content_type='text/csv')


def brand_serps_daily_counts():
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_rows(
        f"""
        select sr.run_at::date as date, c.name as company,
          count(*) as total,
          sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class)='controlled' then 1 else 0 end) as controlled,
          sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='negative' then 1 else 0 end) as negative_serp,
          sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='neutral' then 1 else 0 end) as neutral_serp,
          sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='positive' then 1 else 0 end) as positive_serp
        from serp_runs sr
        join companies c on c.id = sr.company_id
        join serp_results r on r.serp_run_id = sr.id
        left join serp_result_overrides ov on ov.serp_result_id = r.id
        where sr.entity_type = 'company' {scope_sql}
        group by sr.run_at::date, c.name
        order by sr.run_at::date, c.name
        """,
        tuple(params),
    )
    csv_text = rows_to_csv(['date','company','total','controlled','negative_serp','neutral_serp','positive_serp'], rows)
    return Response(csv_text, content_type='text/csv')


def ceo_serps_daily_counts():
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_rows(
        f"""
        select sr.run_at::date as date, ceo.name as ceo, c.name as company,
          count(*) as total,
          sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class)='controlled' then 1 else 0 end) as controlled,
          sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='negative' then 1 else 0 end) as negative_serp,
          sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='neutral' then 1 else 0 end) as neutral_serp,
          sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='positive' then 1 else 0 end) as positive_serp
        from serp_runs sr
        join ceos ceo on ceo.id = sr.ceo_id
        join companies c on c.id = ceo.company_id
        join serp_results r on r.serp_run_id = sr.id
        left join serp_result_overrides ov on ov.serp_result_id = r.id
        where sr.entity_type = 'ceo' {scope_sql}
        group by sr.run_at::date, ceo.name, c.name
        order by sr.run_at::date, ceo.name
        """,
        tuple(params),
    )
    csv_text = rows_to_csv(['date','ceo','company','total','controlled','negative_serp','neutral_serp','positive_serp'], rows)
    return Response(csv_text, content_type='text/csv')


def processed_articles_csv(filename: str):
    m = DATE_RE.match(filename)
    if not m:
        return jsonify({'error': 'invalid filename'}), 400
    dstr, entity, _, kind = m.groups()

    if kind == 'modal':
        if entity == 'brand':
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select c.name as company, a.title, a.canonical_url as url, a.publisher as source,
                       coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label) as sentiment,
                       ov.override_sentiment_label as sentiment_override,
                       ov.override_control_class as control_override,
                       coalesce(cam.llm_sentiment_label, cam.llm_risk_label) as llm_label,
                       cam.id as mention_id
                from company_article_mentions cam
                join companies c on c.id = cam.company_id
                join articles a on a.id = cam.article_id
                left join company_article_overrides ov on ov.company_id = cam.company_id and ov.article_id = cam.article_id
                where cam.scored_at::date = %s {scope_sql}
                order by c.name, a.title
                """,
                tuple(params)
            )
            headers = ['company','title','url','source','sentiment','sentiment_override','control_override','llm_label','mention_id']
        else:
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select ceo.name as ceo, c.name as company, a.title, a.canonical_url as url, a.publisher as source,
                       coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label) as sentiment,
                       ov.override_sentiment_label as sentiment_override,
                       ov.override_control_class as control_override,
                       coalesce(cam.llm_sentiment_label, cam.llm_risk_label) as llm_label,
                       cam.id as mention_id
                from ceo_article_mentions cam
                join ceos ceo on ceo.id = cam.ceo_id
                join companies c on c.id = ceo.company_id
                join articles a on a.id = cam.article_id
                left join ceo_article_overrides ov on ov.ceo_id = cam.ceo_id and ov.article_id = cam.article_id
                where cam.scored_at::date = %s {scope_sql}
                order by ceo.name, a.title
                """,
                tuple(params)
            )
            headers = ['ceo','company','title','url','source','sentiment','sentiment_override','control_override','llm_label','mention_id']
        csv_text = rows_to_csv(headers, rows)
        return Response(csv_text, content_type='text/csv')

    # table
    if entity == 'brand':
        params = [dstr]
        scope_sql, params = scope_clause("c.id", params)
        rows = query_rows(
            f"""
            select cam.scored_at::date as date, c.name as company,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='positive' then 1 else 0 end) as positive,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='neutral' then 1 else 0 end) as neutral,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end) as negative,
              count(*) as total,
              case when count(*) > 0
                then round((sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 6)
                else 0 end as neg_pct
            from company_article_mentions cam
            join companies c on c.id = cam.company_id
            left join company_article_overrides ov on ov.company_id = cam.company_id and ov.article_id = cam.article_id
            where cam.scored_at::date = %s {scope_sql}
            group by cam.scored_at::date, c.name
            order by c.name
            """,
            tuple(params)
        )
        headers = ['date','company','positive','neutral','negative','total','neg_pct']
    else:
        params = [dstr]
        scope_sql, params = scope_clause("c.id", params)
        rows = query_rows(
            f"""
            select cam.scored_at::date as date, ceo.name as ceo, c.name as company,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='positive' then 1 else 0 end) as positive,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='neutral' then 1 else 0 end) as neutral,
              sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end) as negative,
              count(*) as total,
              case when count(*) > 0
                then round((sum(case when coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 1)
                else 0 end as neg_pct,
              '' as theme,
              coalesce(ceo.alias, '') as alias
            from ceo_article_mentions cam
            join ceos ceo on ceo.id = cam.ceo_id
            join companies c on c.id = ceo.company_id
            left join ceo_article_overrides ov on ov.ceo_id = cam.ceo_id and ov.article_id = cam.article_id
            where cam.scored_at::date = %s {scope_sql}
            group by cam.scored_at::date, ceo.name, c.name, ceo.alias
            order by ceo.name
            """,
            tuple(params)
        )
        headers = ['date','ceo','company','positive','neutral','negative','total','neg_pct','theme','alias']
    csv_text = rows_to_csv(headers, rows)
    return Response(csv_text, content_type='text/csv')


def processed_serps_csv(filename: str):
    m = DATE_RE.match(filename)
    if not m:
        return jsonify({'error': 'invalid filename'}), 400
    dstr, entity, _, kind = m.groups()

    if kind == 'modal':
        if entity == 'brand':
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select c.name as company, r.title, r.url, r.rank as position, r.snippet,
                       coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) as sentiment,
                       coalesce(ov.override_control_class, r.llm_control_class, r.control_class) as controlled,
                       ov.override_sentiment_label as sentiment_override,
                       ov.override_control_class as control_override,
                       coalesce(r.llm_sentiment_label, r.llm_risk_label) as llm_label,
                       r.id as serp_result_id
                from serp_runs sr
                join companies c on c.id = sr.company_id
                join serp_results r on r.serp_run_id = sr.id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where sr.entity_type='company' and sr.run_at::date = %s {scope_sql}
                order by c.name, r.rank
                """,
                tuple(params)
            )
            headers = ['company','title','url','position','snippet','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
        else:
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select ceo.name as ceo, c.name as company, r.title, r.url, r.rank as position, r.snippet,
                       coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) as sentiment,
                       coalesce(ov.override_control_class, r.llm_control_class, r.control_class) as controlled,
                       ov.override_sentiment_label as sentiment_override,
                       ov.override_control_class as control_override,
                       coalesce(r.llm_sentiment_label, r.llm_risk_label) as llm_label,
                       r.id as serp_result_id
                from serp_runs sr
                join ceos ceo on ceo.id = sr.ceo_id
                join companies c on c.id = ceo.company_id
                join serp_results r on r.serp_run_id = sr.id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where sr.entity_type='ceo' and sr.run_at::date = %s {scope_sql}
                order by ceo.name, r.rank
                """,
                tuple(params)
            )
            headers = ['ceo','company','title','url','position','snippet','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
        csv_text = rows_to_csv(headers, rows)
        return Response(csv_text, content_type='text/csv')

    if entity == 'brand':
        params = [dstr]
        scope_sql, params = scope_clause("c.id", params)
        rows = query_rows(
            f"""
            select sr.run_at::date as date, c.name as company,
              count(*) as total,
              sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class)='controlled' then 1 else 0 end) as controlled,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='negative' then 1 else 0 end) as negative_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='neutral' then 1 else 0 end) as neutral_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='positive' then 1 else 0 end) as positive_serp
            from serp_runs sr
            join companies c on c.id = sr.company_id
            join serp_results r on r.serp_run_id = sr.id
            left join serp_result_overrides ov on ov.serp_result_id = r.id
            where sr.entity_type='company' and sr.run_at::date = %s {scope_sql}
            group by sr.run_at::date, c.name
            order by c.name
            """,
            tuple(params)
        )
        headers = ['date','company','total','controlled','negative_serp','neutral_serp','positive_serp']
    else:
        params = [dstr]
        scope_sql, params = scope_clause("c.id", params)
        rows = query_rows(
            f"""
            select sr.run_at::date as date, ceo.name as ceo, c.name as company,
              count(*) as total,
              sum(case when coalesce(ov.override_control_class, r.llm_control_class, r.control_class)='controlled' then 1 else 0 end) as controlled,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='negative' then 1 else 0 end) as negative_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='neutral' then 1 else 0 end) as neutral_serp,
              sum(case when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label)='positive' then 1 else 0 end) as positive_serp
            from serp_runs sr
            join ceos ceo on ceo.id = sr.ceo_id
            join companies c on c.id = ceo.company_id
            join serp_results r on r.serp_run_id = sr.id
            left join serp_result_overrides ov on ov.serp_result_id = r.id
            where sr.entity_type='ceo' and sr.run_at::date = %s {scope_sql}
            group by sr.run_at::date, ceo.name, c.name
            order by ceo.name
            """,
            tuple(params)
        )
        headers = ['date','ceo','company','total','controlled','negative_serp','neutral_serp','positive_serp']

    csv_text = rows_to_csv(headers, rows)
    return Response(csv_text, content_type='text/csv')


def roster_csv():
    rows = query_rows(
        """
        select ceo.name as ceo, c.name as company, coalesce(ceo.alias, '') as ceo_alias,
               coalesce(c.websites, '') as websites, coalesce(c.ticker, '') as stock, coalesce(c.sector, '') as sector
        from companies c
        left join ceos ceo on ceo.company_id = c.id
        order by c.name, ceo.name
        """
    )
    csv_text = rows_to_csv(['CEO','Company','CEO Alias','Websites','Stock','Sector'], rows)
    return Response(csv_text, content_type='text/csv')


def stock_data_csv(filename: str):
    m = STOCK_RE.match(filename)
    if not m:
        return jsonify({'error': 'invalid filename'}), 400
    dstr = m.group(1)
    target = datetime.strptime(dstr, '%Y-%m-%d').date()
    rows = build_stock_rows(target)
    csv_rows = []
    for row in rows:
        prices = '|'.join(str(p) for p in row['price_history'])
        dates = '|'.join(row['date_history'])
        csv_rows.append((
            row['ticker'],
            row['company'],
            row['opening_price'],
            row['daily_change_pct'],
            row['seven_day_change_pct'],
            prices,
            dates,
            row['last_updated'] or ''
        ))

    headers = ['ticker','company','opening_price','daily_change_pct','seven_day_change_pct','price_history','date_history','last_updated']
    csv_text = rows_to_csv(headers, csv_rows)
    return Response(csv_text, content_type='text/csv')


def trends_data_csv(filename: str):
    m = TRENDS_RE.match(filename)
    if not m:
        return jsonify({'error': 'invalid filename'}), 400
    dstr = m.group(1)
    target = datetime.strptime(dstr, '%Y-%m-%d').date()
    rows = build_trends_rows(target)
    csv_rows = []
    for row in rows:
        values = '|'.join(str(v) for v in row['trends_history'])
        dates = '|'.join(row['date_history'])
        csv_rows.append((
            row['company'],
            values,
            dates,
            row['last_updated'] or '',
            row['avg_interest'],
        ))

    headers = ['company','trends_history','date_history','last_updated','avg_interest']
    csv_text = rows_to_csv(headers, csv_rows)
    return Response(csv_text, content_type='text/csv')


def build_stock_rows(target: datetime.date) -> List[Dict]:
    scope_ids = get_company_scope_ids()
    params = [target]
    scope_sql = ""
    if scope_ids:
        scope_sql = " and company in (select name from companies where id = any(%s))"
        params.append(scope_ids)
    snapshots = query_dict(
        f"""
        select ticker, company, opening_price, daily_change_pct, seven_day_change_pct, last_updated, as_of_date
        from stock_price_snapshots
        where as_of_date = %s {scope_sql}
        """,
        tuple(params)
    )
    if not snapshots:
        params = [target]
        scope_sql = ""
        if scope_ids:
            scope_sql = " and company in (select name from companies where id = any(%s))"
            params.append(scope_ids)
        latest_rows = query_rows(
            f"""
            select max(as_of_date)
            from stock_price_snapshots
            where as_of_date <= %s {scope_sql}
            """,
            tuple(params)
        )
        latest = latest_rows[0][0] if latest_rows and latest_rows[0] else None
        if latest:
            params = [latest]
            scope_sql = ""
            if scope_ids:
                scope_sql = " and company in (select name from companies where id = any(%s))"
                params.append(scope_ids)
            snapshots = query_dict(
                f"""
                select ticker, company, opening_price, daily_change_pct, seven_day_change_pct, last_updated, as_of_date
                from stock_price_snapshots
                where as_of_date = %s {scope_sql}
                """,
                tuple(params)
            )
            target = latest

    start = target - timedelta(days=120)
    params = [start, target]
    scope_sql = ""
    if scope_ids:
        scope_sql = " and company in (select name from companies where id = any(%s))"
        params.append(scope_ids)
    history = query_dict(
        f"""
        select ticker, company, date, price
        from stock_prices_daily
        where date between %s and %s {scope_sql}
        order by ticker, date
        """,
        tuple(params)
    )

    hist_map: Dict[str, List[Tuple[str, float]]] = {}
    for row in history:
        key = row['ticker'] or row['company']
        hist_map.setdefault(key, []).append((row['date'].isoformat(), row['price']))

    rows = []
    for snap in snapshots:
        key = snap['ticker'] or snap['company']
        series = hist_map.get(key, [])
        prices_only = [p for _, p in series]
        daily_change = snap['daily_change_pct']
        seven_day_change = snap['seven_day_change_pct']
        if daily_change is None and len(prices_only) >= 2:
            prev = prices_only[-2]
            last = prices_only[-1]
            if prev:
                daily_change = ((last - prev) / prev) * 100
        if seven_day_change is None and len(prices_only) >= 8:
            prev7 = prices_only[-8]
            last = prices_only[-1]
            if prev7:
                seven_day_change = ((last - prev7) / prev7) * 100
        rows.append({
            'ticker': snap['ticker'],
            'company': snap['company'],
            'opening_price': snap['opening_price'],
            'daily_change_pct': daily_change,
            'seven_day_change_pct': seven_day_change,
            'price_history': [p for _, p in series],
            'date_history': [d for d, _ in series],
            'volume_history': [],
            'last_updated': snap['last_updated'].isoformat() if snap['last_updated'] else ''
        })
    return rows


def build_trends_rows(target: datetime.date) -> List[Dict]:
    scope_ids = get_company_scope_ids()
    params = [target]
    scope_sql = ""
    if scope_ids:
        scope_sql = " and company in (select name from companies where id = any(%s))"
        params.append(scope_ids)
    snapshots = query_dict(
        f"""
        select company, avg_interest, last_updated
        from trends_snapshots
        where last_updated::date = %s {scope_sql}
        """,
        tuple(params)
    )
    if not snapshots:
        params = [target]
        scope_sql = ""
        if scope_ids:
            scope_sql = " and company in (select name from companies where id = any(%s))"
            params.append(scope_ids)
        latest_rows = query_rows(
            f"""
            select max(last_updated::date)
            from trends_snapshots
            where last_updated::date <= %s {scope_sql}
            """,
            tuple(params)
        )
        latest = latest_rows[0][0] if latest_rows and latest_rows[0] else None
        if latest:
            params = [latest]
            scope_sql = ""
            if scope_ids:
                scope_sql = " and company in (select name from companies where id = any(%s))"
                params.append(scope_ids)
            snapshots = query_dict(
                f"""
                select company, avg_interest, last_updated
                from trends_snapshots
                where last_updated::date = %s {scope_sql}
                """,
                tuple(params)
            )
            target = latest

    start = target - timedelta(days=60)
    params = [start, target]
    scope_sql = ""
    if scope_ids:
        scope_sql = " and company in (select name from companies where id = any(%s))"
        params.append(scope_ids)
    history = query_dict(
        f"""
        select company, date, interest
        from trends_daily
        where date between %s and %s {scope_sql}
        order by company, date
        """,
        tuple(params)
    )

    hist_map: Dict[str, List[Tuple[str, int]]] = {}
    for row in history:
        hist_map.setdefault(row['company'], []).append((row['date'].isoformat(), row['interest']))

    rows = []
    for snap in snapshots:
        series = hist_map.get(snap['company'], [])
        rows.append({
            'company': snap['company'],
            'trends_history': [v for _, v in series],
            'date_history': [d for d, _ in series],
            'last_updated': snap['last_updated'].isoformat() if snap['last_updated'] else '',
            'avg_interest': snap['avg_interest'],
        })
    return rows


def negative_articles_summary():
    lookback_days = int(os.environ.get('NEGATIVE_SUMMARY_LOOKBACK_DAYS', '90'))
    start_date = datetime.utcnow().date() - timedelta(days=lookback_days)

    params = [start_date]
    scope_sql, params = scope_clause("c.id", params)
    rows = query_dict(
        f"""
        select cam.scored_at::date as date, c.name as company, ceo.name as ceo,
               coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label) as sentiment,
               a.title, 'brand' as article_type
        from company_article_mentions cam
        join companies c on c.id = cam.company_id
        join articles a on a.id = cam.article_id
        left join company_article_overrides ov on ov.company_id = cam.company_id and ov.article_id = cam.article_id
        left join ceos ceo on ceo.company_id = c.id
        where cam.scored_at::date >= %s {scope_sql}
        union all
        select cam.scored_at::date as date, c.name as company, ceo.name as ceo,
               coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label) as sentiment,
               a.title, 'ceo' as article_type
        from ceo_article_mentions cam
        join ceos ceo on ceo.id = cam.ceo_id
        join companies c on c.id = ceo.company_id
        join articles a on a.id = cam.article_id
        left join ceo_article_overrides ov on ov.ceo_id = cam.ceo_id and ov.article_id = cam.article_id
        where cam.scored_at::date >= %s {scope_sql}
        """,
        tuple(params + params),
    )

    grouped: Dict[Tuple[str, str, str, str], List[str]] = {}
    for r in rows:
        if r.get('sentiment') != 'negative':
            continue
        date_str = r['date'].isoformat()
        company = (r.get('company') or '').strip()
        if not company:
            continue
        ceo = r.get('ceo') or ''
        article_type = r.get('article_type') or 'brand'
        key = (date_str, company, ceo, article_type)
        grouped.setdefault(key, []).append(r.get('title') or '')

    out_rows = []
    for (date_str, company, ceo, article_type), titles in grouped.items():
        clean = [t.strip() for t in titles if t.strip()]
        top = '|'.join(clean[:3])
        out_rows.append((date_str, company, ceo, len(clean), top, article_type))

    headers = ['date','company','ceo','negative_count','top_headlines','article_type']
    csv_text = rows_to_csv(headers, out_rows)
    return Response(csv_text, content_type='text/csv')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
