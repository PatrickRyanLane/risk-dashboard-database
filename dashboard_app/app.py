#!/usr/bin/env python3
"""
DB-backed dashboard API + static assets.
Internal service should be protected by IAP; external service read-only.
"""

import csv
import gzip
import time
import io
import logging
import os
import re
import threading
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import psycopg2
from psycopg2.pool import PoolError, ThreadedConnectionPool
import requests
from flask import Flask, Response, jsonify, request, send_from_directory, abort
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

app = Flask(__name__, static_folder='static', static_url_path='/static')
logging.basicConfig(level=logging.INFO)

PORT = int(os.environ.get('PORT', 8080))
DB_DSN = os.environ.get('DATABASE_URL')
DB_POOL_MIN = int(os.environ.get('DB_POOL_MIN', '1'))
DB_POOL_MAX = int(os.environ.get('DB_POOL_MAX', '5'))
DEFAULT_VIEW = os.environ.get('DEFAULT_VIEW', 'external')
PUBLIC_MODE = os.environ.get('PUBLIC_MODE', 'false').lower() in {'1', 'true', 'yes'}
ALLOW_EDITS = os.environ.get('ALLOW_EDITS', 'true').lower() in {'1', 'true', 'yes'}
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai').lower()
LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-4o-mini')
LLM_SUMMARY_ITEMS = int(os.environ.get('LLM_SUMMARY_ITEMS', '12'))
IAP_AUDIENCE = os.environ.get('IAP_AUDIENCE', '')
ALLOWED_DOMAIN = os.environ.get('ALLOWED_DOMAIN', '')
ALLOWED_EMAILS = {e.strip().lower() for e in os.environ.get('ALLOWED_EMAILS', '').split(',') if e.strip()}
ALLOW_UNAUTHED_INTERNAL = os.environ.get('ALLOW_UNAUTHED_INTERNAL', 'false').lower() in {'1', 'true', 'yes'}
EXTERNAL_COMPANY_SCOPE = [c.strip() for c in os.environ.get('EXTERNAL_COMPANY_SCOPE', '').split(',') if c.strip()]
IAP_CERTS_URL = os.environ.get('IAP_CERTS_URL', 'https://www.gstatic.com/iap/verify/public_key')

_api_cache = {}
_api_cache_ttl = int(os.environ.get('API_CACHE_TTL', '300'))
_db_pool = None
_db_pool_lock = threading.Lock()
_db_fallback_ids = set()
_db_fallback_lock = threading.Lock()
_refresh_lock = threading.Lock()
_refresh_in_progress = False
_refresh_last_status = {
    'status': 'idle',
    'started_at': None,
    'finished_at': None,
    'duration_ms': None,
    'error': None,
}

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


def build_serp_feature_summary_prompt(entity_type: str, entity_name: str,
                                      feature_type: str, items: List[Dict]) -> Dict[str, str]:
    system = (
        "You summarize SERP feature results for internal users. "
        "Write exactly one concise sentence. "
        "No preamble, no leading entity name."
    )
    lines = []
    for item in items:
        title = (item.get("title") or "").strip()
        source = (item.get("source") or "").strip()
        url = (item.get("url") or "").strip()
        if title and source:
            lines.append(f"- {title} ({source})")
        elif title:
            lines.append(f"- {title}")
        elif url:
            lines.append(f"- {url}")
    joined = "\n".join(lines)
    user = (
        f"Entity: {entity_type} = {entity_name}\n"
        f"Feature: {feature_type}\n"
        f"Items:\n{joined}\n"
        "Return summary only."
    )
    return {"system": system, "user": user}


def call_llm_text(prompt: Dict[str, str]) -> Tuple[str, str, str]:
    if not LLM_API_KEY:
        return "", "llm_not_configured", ""
    if LLM_PROVIDER == "gemini":
        model = LLM_MODEL or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": f"{prompt['system']}\n\n{prompt['user']}"}]}],
            "generationConfig": {"temperature": 0.2},
        }
        resp = requests.post(url, params={"key": LLM_API_KEY}, json=payload, timeout=20)
        if resp.status_code != 200:
            detail = resp.text or ""
            if len(detail) > 1000:
                detail = detail[:1000] + "…"
            return "", f"gemini_http_{resp.status_code}", detail
        data = resp.json()
        try:
            return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip(), "", ""
        except Exception:
            return "", "gemini_parse_error", ""
    model = LLM_MODEL or "gpt-4o-mini"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if resp.status_code != 200:
        detail = resp.text or ""
        if len(detail) > 1000:
            detail = detail[:1000] + "…"
        return "", f"openai_http_{resp.status_code}", detail
    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip(), "", ""
    except Exception:
        return "", "openai_parse_error", ""


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
    global _db_pool
    if DB_POOL_MAX < 1:
        return psycopg2.connect(DB_DSN)
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_pool = ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, dsn=DB_DSN)
    try:
        return _db_pool.getconn()
    except PoolError:
        app.logger.warning("db_pool_exhausted_fallback")
        conn = psycopg2.connect(DB_DSN)
        with _db_fallback_lock:
            _db_fallback_ids.add(id(conn))
        return conn


def put_conn(conn) -> None:
    if conn is None:
        return
    with _db_fallback_lock:
        if id(conn) in _db_fallback_ids:
            _db_fallback_ids.remove(id(conn))
            conn.close()
            return
    if _db_pool is None:
        conn.close()
        return
    try:
        conn.autocommit = False
    except Exception:
        pass
    _db_pool.putconn(conn)


def rows_to_csv(headers: List[str], rows: Iterable[Tuple]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def query_rows(sql: str, params: Tuple = ()) -> List[Tuple]:
    start = time.perf_counter()
    conn = get_conn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        put_conn(conn)
    elapsed = (time.perf_counter() - start) * 1000
    if os.environ.get('PERF_LOG', '0') == '1' or elapsed > 500:
        head = (sql.strip().splitlines()[0] if sql else '').strip()
        app.logger.info("query_rows_ms=%.1f sql=%s", elapsed, head[:200])
    return rows


def query_dict(sql: str, params: Tuple = ()) -> List[Dict]:
    start = time.perf_counter()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        put_conn(conn)
    elapsed = (time.perf_counter() - start) * 1000
    if os.environ.get('PERF_LOG', '0') == '1' or elapsed > 500:
        head = (sql.strip().splitlines()[0] if sql else '').strip()
        app.logger.info("query_dict_ms=%.1f sql=%s", elapsed, head[:200])
    return rows


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
        select distinct cad.date as date
        from company_article_mentions_daily cad
        join companies c on c.id = cad.company_id
        where cad.date is not null {scope_sql}
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
    days_raw = (request.args.get('days') or '').strip()
    days = None
    if days_raw:
        try:
            days = max(1, int(days_raw))
        except ValueError:
            days = None
    if kind == 'brand_articles':
        params = []
        scope_sql, params = scope_clause("mv.company_id", params)
        date_sql = ""
        if days:
            params.append(days)
            date_sql = "and mv.date >= (current_date - (%s || ' days')::interval)"
        rows = query_dict(
            f"""
            select mv.date as date, mv.company as company,
                   mv.positive, mv.neutral, mv.negative, mv.total, mv.neg_pct
            from article_daily_counts_mv mv
            where mv.entity_type = 'brand' {scope_sql} {date_sql}
            order by mv.date, mv.company
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    if kind == 'ceo_articles':
        params = []
        scope_sql, params = scope_clause("mv.company_id", params)
        date_sql = ""
        if days:
            params.append(days)
            date_sql = "and mv.date >= (current_date - (%s || ' days')::interval)"
        rows = query_dict(
            f"""
            select mv.date as date, mv.ceo as ceo, mv.company as company,
                   mv.positive, mv.neutral, mv.negative, mv.total, mv.neg_pct,
                   '' as theme, mv.alias
            from article_daily_counts_mv mv
            where mv.entity_type = 'ceo' {scope_sql} {date_sql}
            order by mv.date, mv.ceo
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    if kind == 'brand_serps':
        params = []
        scope_sql, params = scope_clause("mv.company_id", params)
        date_sql = ""
        if days:
            params.append(days)
            date_sql = "and mv.date >= (current_date - (%s || ' days')::interval)"
        rows = query_dict(
            f"""
            select mv.date as date, mv.company as company,
                   mv.total, mv.controlled, mv.negative_serp, mv.neutral_serp, mv.positive_serp
            from serp_daily_counts_mv mv
            where mv.entity_type = 'brand' {scope_sql} {date_sql}
            order by mv.date, mv.company
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    if kind == 'ceo_serps':
        params = []
        scope_sql, params = scope_clause("mv.company_id", params)
        date_sql = ""
        if days:
            params.append(days)
            date_sql = "and mv.date >= (current_date - (%s || ' days')::interval)"
        rows = query_dict(
            f"""
            select mv.date as date, mv.ceo as ceo, mv.company as company,
                   mv.total, mv.controlled, mv.negative_serp, mv.neutral_serp, mv.positive_serp
            from serp_daily_counts_mv mv
            where mv.entity_type = 'ceo' {scope_sql} {date_sql}
            order by mv.date, mv.ceo
            """,
            tuple(params),
        )
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    return jsonify({'error': 'invalid kind'}), 400


@app.route('/api/v1/processed_articles')
def processed_articles_json():
    debug = (request.args.get('debug') or '').strip().lower() in {'1', 'true', 'yes'}
    dstr = request.args.get('date', '')
    entity = request.args.get('entity', 'brand')
    kind = request.args.get('kind', 'modal')
    entity_name = request.args.get('entity_name', '').strip()
    if not dstr:
        return jsonify({'error': 'date is required (YYYY-MM-DD)'}), 400
    try:
        limit = max(1, int(request.args.get('limit', '200')))
    except ValueError:
        limit = 200
    try:
        offset = max(0, int(request.args.get('offset', '0')))
    except ValueError:
        offset = 0
    if limit > 1000:
        limit = 1000

    filename = f"{dstr}-{entity}-articles-{kind}.csv"
    if kind == 'modal':
        try:
            if entity == 'brand':
                params = [dstr]
                scope_sql, params = scope_clause("c.id", params)
                name_sql = ""
                if entity_name:
                    params.append(entity_name)
                    name_sql = "and c.name = %s"
                params.extend([limit, offset])
                total_params = params[:-2]
                rows = query_rows(
                    f"""
                    select c.name as company, a.title, a.canonical_url as url, a.publisher as source,
                           coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
                           coalesce(ov.override_control_class, cm.control_class) as control_class,
                           ov.override_sentiment_label as sentiment_override,
                           ov.override_control_class as control_override,
                           coalesce(cm.llm_sentiment_label, cm.llm_risk_label) as llm_label,
                           cm.id as mention_id
                    from company_article_mentions_daily cad
                    join companies c on c.id = cad.company_id
                    join articles a on a.id = cad.article_id
                    left join company_article_mentions cm on cm.company_id = cad.company_id and cm.article_id = cad.article_id
                    left join company_article_overrides ov on ov.company_id = cad.company_id and ov.article_id = cad.article_id
                    where cad.date = %s {scope_sql} {name_sql}
                    order by c.name, a.title
                    limit %s offset %s
                    """,
                    tuple(params)
                )
                total_rows = query_rows(
                    f"""
                    select count(*)
                    from company_article_mentions_daily cad
                    join companies c on c.id = cad.company_id
                    join articles a on a.id = cad.article_id
                    left join company_article_mentions cm on cm.company_id = cad.company_id and cm.article_id = cad.article_id
                    left join company_article_overrides ov on ov.company_id = cad.company_id and ov.article_id = cad.article_id
                    where cad.date = %s {scope_sql} {name_sql}
                    """,
                    tuple(total_params)
                )
                total = int(total_rows[0][0]) if total_rows else 0
                headers = ['company','title','url','source','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
            else:
                params = [dstr]
                scope_sql, params = scope_clause("c.id", params)
                name_sql = ""
                if entity_name:
                    params.append(entity_name)
                    name_sql = "and ceo.name = %s"
                params.extend([limit, offset])
                total_params = params[:-2]
                rows = query_rows(
                    f"""
                    select ceo.name as ceo, c.name as company, a.title, a.canonical_url as url, a.publisher as source,
                           coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
                           coalesce(ov.override_control_class, cm.control_class) as control_class,
                           ov.override_sentiment_label as sentiment_override,
                           ov.override_control_class as control_override,
                           coalesce(cm.llm_sentiment_label, cm.llm_risk_label) as llm_label,
                           cm.id as mention_id
                    from ceo_article_mentions_daily cad
                    join ceos ceo on ceo.id = cad.ceo_id
                    join companies c on c.id = ceo.company_id
                    join articles a on a.id = cad.article_id
                    left join ceo_article_mentions cm on cm.ceo_id = cad.ceo_id and cm.article_id = cad.article_id
                    left join ceo_article_overrides ov on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
                    where cad.date = %s {scope_sql} {name_sql}
                    order by ceo.name, a.title
                    limit %s offset %s
                    """,
                    tuple(params)
                )
                total_rows = query_rows(
                    f"""
                    select count(*)
                    from ceo_article_mentions_daily cad
                    join ceos ceo on ceo.id = cad.ceo_id
                    join companies c on c.id = ceo.company_id
                    join articles a on a.id = cad.article_id
                    left join ceo_article_mentions cm on cm.ceo_id = cad.ceo_id and cm.article_id = cad.article_id
                    left join ceo_article_overrides ov on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
                    where cad.date = %s {scope_sql} {name_sql}
                    """,
                    tuple(total_params)
                )
                total = int(total_rows[0][0]) if total_rows else 0
                headers = ['ceo','company','title','url','source','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
            return jsonify({"rows": [dict(zip(headers, row)) for row in rows], "total": total})
        except Exception as exc:
            app.logger.exception("processed_articles failed")
            if debug:
                return jsonify({'error': 'processed_articles_failed', 'detail': str(exc)}), 500
            raise
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
    entity_name = request.args.get('entity_name', '').strip()
    try:
        limit = max(1, int(request.args.get('limit', '200')))
    except ValueError:
        limit = 200
    try:
        offset = max(0, int(request.args.get('offset', '0')))
    except ValueError:
        offset = 0
    if limit > 1000:
        limit = 1000
    filename = f"{dstr}-{entity}-serps-{kind}.csv"
    if kind == 'modal':
        if entity == 'brand':
            params = [dstr, dstr]
            scope_sql, params = scope_clause("c.id", params)
            name_sql = ""
            if entity_name:
                params.append(entity_name)
                name_sql = "and c.name = %s"
            params.extend([limit, offset])
            total_params = params[:-2]
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
                where sr.entity_type='company'
                  and sr.run_at >= %s::date
                  and sr.run_at < (%s::date + interval '1 day')
                  {scope_sql} {name_sql}
                order by c.name, r.rank
                limit %s offset %s
                """,
                tuple(params)
            )
            total_rows = query_rows(
                f"""
                select count(*)
                from serp_runs sr
                join companies c on c.id = sr.company_id
                join serp_results r on r.serp_run_id = sr.id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where sr.entity_type='company'
                  and sr.run_at >= %s::date
                  and sr.run_at < (%s::date + interval '1 day')
                  {scope_sql} {name_sql}
                """,
                tuple(total_params)
            )
            total = int(total_rows[0][0]) if total_rows else 0
            headers = ['company','title','url','position','snippet','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
        else:
            params = [dstr, dstr]
            scope_sql, params = scope_clause("c.id", params)
            name_sql = ""
            if entity_name:
                params.append(entity_name)
                name_sql = "and ceo.name = %s"
            params.extend([limit, offset])
            total_params = params[:-2]
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
                where sr.entity_type='ceo'
                  and sr.run_at >= %s::date
                  and sr.run_at < (%s::date + interval '1 day')
                  {scope_sql} {name_sql}
                order by ceo.name, r.rank
                limit %s offset %s
                """,
                tuple(params)
            )
            total_rows = query_rows(
                f"""
                select count(*)
                from serp_runs sr
                join ceos ceo on ceo.id = sr.ceo_id
                join companies c on c.id = ceo.company_id
                join serp_results r on r.serp_run_id = sr.id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where sr.entity_type='ceo'
                  and sr.run_at >= %s::date
                  and sr.run_at < (%s::date + interval '1 day')
                  {scope_sql} {name_sql}
                """,
                tuple(total_params)
            )
            total = int(total_rows[0][0]) if total_rows else 0
            headers = ['ceo','company','title','url','position','snippet','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
        return jsonify({"rows": [dict(zip(headers, row)) for row in rows], "total": total})
    resp = processed_serps_csv(filename)
    if resp.status_code != 200:
        return resp
    rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
    return jsonify(rows)


@app.route('/api/v1/serp_features')
def serp_features_json():
    entity = request.args.get('entity', 'brand')
    days = int(request.args.get('days', '90') or 90)
    date_str = (request.args.get('date') or '').strip()
    entity_name = request.args.get('entity_name', '').strip()
    mode = request.args.get('mode', '').strip()
    debug = (request.args.get('debug') or '').strip().lower() in {'1', 'true', 'yes'}
    cache_key = f"serp_features:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    entity_type = 'company' if entity == 'brand' else 'ceo'
    entity_types = ['brand', 'company'] if entity_type == 'company' else [entity_type]

    if entity_type == 'company':
        if mode == "index":
            if date_str:
                params = [entity_types, date_str]
                date_sql = "and date = %s"
            else:
                params = [entity_types, days]
                date_sql = "and date >= (current_date - (%s || ' days')::interval) and date <= current_date"
            sql = f"""
                select date, 'Index' as entity_name, feature_type,
                       sum(total_count) as total_count,
                       sum(positive_count) as positive_count,
                       sum(neutral_count) as neutral_count,
                       sum(negative_count) as negative_count
                from serp_feature_daily_index_mv
                where entity_type = any(%s)
                  {date_sql}
                group by date, feature_type
                order by date, feature_type
            """
        else:
            name_sql_base = "and s.entity_name = %s" if entity_name else ""
            feat_val = request.args.get('feature_type')
            feat_sql_base = "and s.feature_type = %s" if feat_val else ""

        if mode == "index":
            params = params
        else:
            params_base = [entity_types]
            scope_sql_base, params_base = scope_clause("c.id", params_base)
            if date_str:
                params_base.append(date_str)
                date_sql = "and s.date = %s"
            else:
                params_base.append(days)
                date_sql = "and s.date >= (current_date - (%s || ' days')::interval) and s.date <= current_date"

            if entity_name:
                params_base.append(entity_name)
            if request.args.get('feature_type'):
                params_base.append(request.args.get('feature_type'))

            join_sql = "join companies c on c.id = s.entity_id" if scope_sql_base else ""
            sql = f"""
                select s.date, s.entity_name, s.feature_type,
                       s.total_count, s.positive_count, s.neutral_count, s.negative_count
                from serp_feature_daily_mv s
                {join_sql}
                where s.entity_type = any(%s)
                  {date_sql}
                  {scope_sql_base}
                  {name_sql_base}
                  {feat_sql_base}
                order by date, feature_type
            """
            params = params_base
    else:
        if mode == "index":
            if date_str:
                params = [entity_type, date_str]
                date_sql = "and date = %s"
            else:
                params = [entity_type, days]
                date_sql = "and date >= (current_date - (%s || ' days')::interval) and date <= current_date"
            sql = f"""
                select date, 'Index' as entity_name, feature_type,
                       sum(total_count) as total_count,
                       sum(positive_count) as positive_count,
                       sum(neutral_count) as neutral_count,
                       sum(negative_count) as negative_count
                from serp_feature_daily_index_mv
                where entity_type = %s
                  {date_sql}
                group by date, feature_type
                order by date, feature_type
            """
        else:
            name_sql_base = "and s.entity_name = %s" if entity_name else ""
            feat_val = request.args.get('feature_type')
            feat_sql_base = "and s.feature_type = %s" if feat_val else ""

            params_base = [entity_type]
            scope_sql_base, params_base = scope_clause("c.id", params_base)
            if date_str:
                params_base.append(date_str)
                date_sql = "and s.date = %s"
            else:
                params_base.append(days)
                date_sql = "and s.date >= (current_date - (%s || ' days')::interval) and s.date <= current_date"

            if entity_name:
                params_base.append(entity_name)
            if request.args.get('feature_type'):
                params_base.append(request.args.get('feature_type'))

            join_sql = "join ceos ceo on ceo.id = s.entity_id join companies c on c.id = ceo.company_id" if scope_sql_base else ""
            sql = f"""
                select s.date, s.entity_name, s.feature_type,
                       s.total_count, s.positive_count, s.neutral_count, s.negative_count
                from serp_feature_daily_mv s
                {join_sql}
                where s.entity_type = %s
                  {date_sql}
                  {scope_sql_base}
                  {name_sql_base}
                  {feat_sql_base}
                order by date, feature_type
            """
            params = params_base
    try:
        rows = query_dict(sql, tuple(params))
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    except Exception as exc:
        app.logger.exception("serp_features failed")
        if debug:
            return jsonify({'error': 'serp_features_failed', 'detail': str(exc)}), 500
        raise


@app.route('/api/v1/serp_feature_controls')
def serp_feature_controls_json():
    entity = request.args.get('entity', 'brand')
    days = int(request.args.get('days', '90') or 90)
    date_str = (request.args.get('date') or '').strip()
    entity_name = request.args.get('entity_name', '').strip()
    mode = request.args.get('mode', '').strip()
    debug = (request.args.get('debug') or '').strip().lower() in {'1', 'true', 'yes'}
    cache_key = f"serp_feature_controls:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)
    entity_type = 'company' if entity == 'brand' else 'ceo'
    entity_types = ['brand', 'company'] if entity_type == 'company' else [entity_type]

    if entity_type == 'company':
        if mode == "index":
            if date_str:
                params = [entity_types, date_str]
                date_sql = "and date = %s"
            else:
                params = [entity_types, days]
                date_sql = "and date >= (current_date - (%s || ' days')::interval) and date <= current_date"
            sql = f"""
                select date, 'Index' as entity_name, feature_type,
                       sum(total_count) as total_count,
                       sum(controlled_count) as controlled_count
                from serp_feature_control_daily_index_mv
                where entity_type = any(%s)
                  {date_sql}
                group by date, feature_type
                order by date, feature_type
            """
        else:
            name_sql = "and s.entity_name = %s" if entity_name else ""
            params = [entity_types]
            scope_sql, params = scope_clause("c.id", params)
            if date_str:
                params.append(date_str)
                date_sql = "and s.date = %s"
            else:
                params.append(days)
                date_sql = "and s.date >= (current_date - (%s || ' days')::interval) and s.date <= current_date"
            if entity_name:
                params.append(entity_name)
            join_sql = "join companies c on c.id = s.entity_id" if scope_sql else ""
            sql = f"""
                select s.date, s.entity_name, s.feature_type,
                       s.total_count, s.controlled_count
                from serp_feature_control_daily_mv s
                {join_sql}
                where s.entity_type = any(%s)
                  {date_sql}
                  {scope_sql}
                  {name_sql}
                order by date, feature_type
            """
    else:
        if mode == "index":
            if date_str:
                params = [entity_type, date_str]
                date_sql = "and date = %s"
            else:
                params = [entity_type, days]
                date_sql = "and date >= (current_date - (%s || ' days')::interval) and date <= current_date"
            sql = f"""
                select date, 'Index' as entity_name, feature_type,
                       sum(total_count) as total_count,
                       sum(controlled_count) as controlled_count
                from serp_feature_control_daily_index_mv
                where entity_type = %s
                  {date_sql}
                group by date, feature_type
                order by date, feature_type
            """
        else:
            name_sql = "and s.entity_name = %s" if entity_name else ""
            params = [entity_type]
            scope_sql, params = scope_clause("c.id", params)
            if date_str:
                params.append(date_str)
                date_sql = "and s.date = %s"
            else:
                params.append(days)
                date_sql = "and s.date >= (current_date - (%s || ' days')::interval) and s.date <= current_date"
            if entity_name:
                params.append(entity_name)
            join_sql = "join ceos ceo on ceo.id = s.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
            sql = f"""
                select s.date, s.entity_name, s.feature_type,
                       s.total_count, s.controlled_count
                from serp_feature_control_daily_mv s
                {join_sql}
                where s.entity_type = %s
                  {date_sql}
                  {scope_sql}
                  {name_sql}
                order by date, feature_type
            """

    try:
        rows = query_dict(sql, tuple(params))
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    except Exception as exc:
        app.logger.exception("serp_feature_controls failed")
        if debug:
            return jsonify({'error': 'serp_feature_controls_failed', 'detail': str(exc)}), 500
        raise


@app.route('/api/v1/serp_feature_items')
def serp_feature_items_json():
    if PUBLIC_MODE:
        return jsonify({'error': 'not available'}), 403
    date_str = (request.args.get('date') or '').strip()
    if not date_str:
        return jsonify({'error': 'date is required (YYYY-MM-DD)'}), 400
    entity = request.args.get('entity', 'brand')
    entity_name = (request.args.get('entity_name') or '').strip()
    feature_type = (request.args.get('feature_type') or '').strip()
    try:
        limit = int(request.args.get('limit', '200') or 200)
    except ValueError:
        limit = 200
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    try:
        offset = int(request.args.get('offset', '0') or 0)
    except ValueError:
        offset = 0
    if offset < 0:
        offset = 0

    entity_type = 'company' if entity == 'brand' else 'ceo'
    if entity_type == 'company':
        entity_types = ['brand', 'company']
        params = [date_str, entity_types]
        scope_sql, params = scope_clause("c.id", params)
        name_sql = ""
        feat_sql = ""
        if entity_name:
            params.append(entity_name)
            name_sql = "and sfi.entity_name = %s"
        if feature_type:
            params.append(feature_type)
            feat_sql = "and sfi.feature_type = %s"
        params.append(limit)
        params.append(offset)
        sql = f"""
            select sfi.id, sfi.date, sfi.entity_name, sfi.feature_type,
                   sfi.title, sfi.snippet, sfi.url, sfi.domain, sfi.position, sfi.source,
                   coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) as sentiment,
                   coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) as control_class,
                   ov.override_sentiment_label as sentiment_override,
                   ov.override_control_class as control_override,
                   coalesce(sfi.llm_sentiment_label, sfi.llm_risk_label) as llm_label
            from serp_feature_items sfi
            join companies c on c.id = sfi.entity_id
            left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
            where sfi.date = %s
              and sfi.entity_type = any(%s)
              {scope_sql}
              {name_sql}
              {feat_sql}
            order by sfi.feature_type, sfi.position nulls last, sfi.sentiment_label
            limit %s offset %s
        """
    else:
        params = [date_str, entity_type]
        scope_sql, params = scope_clause("c.id", params)
        name_sql = ""
        feat_sql = ""
        if entity_name:
            params.append(entity_name)
            name_sql = "and sfi.entity_name = %s"
        if feature_type:
            params.append(feature_type)
            feat_sql = "and sfi.feature_type = %s"
        params.append(limit)
        params.append(offset)
        sql = f"""
            select sfi.id, sfi.date, sfi.entity_name, sfi.feature_type,
                   sfi.title, sfi.snippet, sfi.url, sfi.domain, sfi.position, sfi.source,
                   coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) as sentiment,
                   coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) as control_class,
                   ov.override_sentiment_label as sentiment_override,
                   ov.override_control_class as control_override,
                   coalesce(sfi.llm_sentiment_label, sfi.llm_risk_label) as llm_label
            from serp_feature_items sfi
            join ceos ceo on ceo.id = sfi.entity_id
            join companies c on c.id = ceo.company_id
            left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
            where sfi.date = %s
              and sfi.entity_type = %s
              {scope_sql}
              {name_sql}
              {feat_sql}
            order by sfi.feature_type, sfi.position nulls last, sfi.sentiment_label
            limit %s offset %s
        """

    rows = query_dict(sql, tuple(params))
    return jsonify(serialize_rows(rows))


@app.route('/api/v1/serp_feature_series')
def serp_feature_series_json():
    if PUBLIC_MODE:
        return jsonify({'error': 'not available'}), 403
    entity = request.args.get('entity', 'brand')
    entity_name = (request.args.get('entity_name') or '').strip()
    feature_type = (request.args.get('feature_type') or '').strip()
    if not entity_name or not feature_type:
        return jsonify({'error': 'entity_name and feature_type are required'}), 400
    try:
        days = int(request.args.get('days', '30') or 30)
    except ValueError:
        days = 30
    if days < 1:
        days = 1
    if days > 365:
        days = 365

    entity_type = 'company' if entity == 'brand' else 'ceo'
    if entity_type == 'company':
        entity_types = ['brand', 'company']
        params = [entity_types, entity_name, feature_type, days]
        scope_sql, params = scope_clause("c.id", params)
        sql = f"""
            select sfi.date,
                   count(*) as total_count,
                   sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'positive' then 1 else 0 end) as positive_count,
                   sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'neutral' then 1 else 0 end) as neutral_count,
                   sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative' then 1 else 0 end) as negative_count,
                   sum(case when coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'controlled' then 1 else 0 end) as controlled_count
            from serp_feature_items sfi
            join companies c on c.id = sfi.entity_id
            left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
            where sfi.entity_type = any(%s)
              and sfi.entity_name = %s
              and sfi.feature_type = %s
              and sfi.date >= (current_date - (%s || ' days')::interval)
              {scope_sql}
            group by sfi.date
            order by sfi.date
        """
    else:
        params = [entity_type, entity_name, feature_type, days]
        scope_sql, params = scope_clause("c.id", params)
        sql = f"""
            select sfi.date,
                   count(*) as total_count,
                   sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'positive' then 1 else 0 end) as positive_count,
                   sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'neutral' then 1 else 0 end) as neutral_count,
                   sum(case when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative' then 1 else 0 end) as negative_count,
                   sum(case when coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'controlled' then 1 else 0 end) as controlled_count
            from serp_feature_items sfi
            join ceos ceo on ceo.id = sfi.entity_id
            join companies c on c.id = ceo.company_id
            left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
            where sfi.entity_type = %s
              and sfi.entity_name = %s
              and sfi.feature_type = %s
              and sfi.date >= (current_date - (%s || ' days')::interval)
              {scope_sql}
            group by sfi.date
            order by sfi.date
        """

    rows = query_dict(sql, tuple(params))
    return jsonify(serialize_rows(rows))


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
    debug = (request.args.get('debug') or '').strip().lower() in {'1', 'true', 'yes'}
    mode = (request.args.get('mode') or '').strip().lower()
    company_filter = (request.args.get('company') or '').strip()
    days_raw = request.args.get('days', '').strip()
    days = None
    if days_raw:
        try:
            days = max(1, int(days_raw))
        except ValueError:
            days = None
    try:
        if mode == 'index':
            lookback_days = days or int(os.environ.get('NEGATIVE_SUMMARY_LOOKBACK_DAYS', '90'))
            start_date = datetime.utcnow().date() - timedelta(days=lookback_days)
            rows = query_dict(
                """
                select date,
                       'Index' as company,
                       '' as ceo,
                       article_type,
                       sum(negative_count) as negative_count,
                       '' as top_headlines,
                       sum(crisis_risk_count) as crisis_risk_count
                from negative_articles_summary_mv
                where date >= %s
                group by date, article_type
                order by date desc, article_type
                """,
                (start_date,),
            )
        else:
            rows = negative_summary_view(days=days, company=company_filter or None)
        if not rows:
            resp = negative_articles_summary(days=days)
            if resp.status_code != 200:
                return resp
            rows = list(csv.DictReader(io.StringIO(resp.get_data(as_text=True))))
        set_cached_json(cache_key, rows)
        return jsonify(rows)
    except Exception as exc:
        app.logger.exception("negative_summary failed")
        if debug:
            return jsonify({'error': 'negative_summary_failed', 'detail': str(exc)}), 500
        raise


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
        response.headers.setdefault('Vary', 'Accept-Encoding')
    if response.direct_passthrough:
        return response
    if response.status_code < 200 or response.status_code >= 300:
        return response
    if response.headers.get('Content-Encoding'):
        return response
    accept = request.headers.get('Accept-Encoding', '')
    if 'gzip' not in accept.lower():
        return response
    ctype = (response.headers.get('Content-Type') or '').lower()
    if not (ctype.startswith('application/json') or ctype.startswith('text/csv') or ctype.startswith('text/plain')):
        return response
    data = response.get_data()
    if not data or len(data) < 1024:
        return response
    gz = gzip.compress(data)
    response.set_data(gz)
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = str(len(gz))
    response.headers.setdefault('Vary', 'Accept-Encoding')
    return response


# --------------------------- Internal edit endpoint ---------------------------

@app.route('/api/internal/serp_feature_summary')
def serp_feature_summary():
    if PUBLIC_MODE:
        return jsonify({'error': 'not available'}), 403
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401

    date_str = (request.args.get('date') or '').strip()
    entity = request.args.get('entity', 'brand').strip()
    entity_name = (request.args.get('entity_name') or '').strip()
    feature_type = (request.args.get('feature_type') or '').strip()
    refresh = request.args.get('refresh', '').strip() in {'1', 'true', 'yes'}

    if not date_str or not entity_name or not feature_type:
        return jsonify({'error': 'date, entity_name, feature_type are required'}), 400

    entity_type = 'company' if entity == 'brand' else 'ceo'
    entity_types = ['brand', 'company'] if entity_type == 'company' else [entity_type]
    if request.args.get('debug', '').strip() in {'1', 'true', 'yes'}:
        return jsonify({
            'date': date_str,
            'entity': entity,
            'entity_name': entity_name,
            'feature_type': feature_type,
            'entity_type': entity_type,
            'entity_types': entity_types,
            'llm_provider': LLM_PROVIDER,
            'llm_model': LLM_MODEL,
            'llm_summary_items': LLM_SUMMARY_ITEMS,
            'llm_key_set': bool(LLM_API_KEY),
        })
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if entity_type == 'company':
                cur.execute("select id, name from companies where name = %s", (entity_name,))
            else:
                cur.execute("select id, name from ceos where name = %s", (entity_name,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'entity not found'}), 404
            entity_id, canonical_name = row[0], row[1]

            if not refresh:
                cur.execute(
                    """
                    select summary_text, provider, model, updated_at
                    from serp_feature_summaries
                    where date = %s and entity_type = %s and entity_id = %s and feature_type = %s
                    """,
                    (date_str, entity_type, entity_id, feature_type),
                )
                cached = cur.fetchone()
                if cached:
                    summary_text, provider, model, updated_at = cached
                    return jsonify({
                        'summary': summary_text,
                        'provider': provider,
                        'model': model,
                        'updated_at': updated_at.isoformat() if hasattr(updated_at, 'isoformat') else updated_at,
                        'cached': True,
                    })

            cur.execute(
                """
                select title, snippet, source, url
                from serp_feature_items
                where date = %s
                  and entity_type = any(%s)
                  and entity_id = %s
                  and feature_type = %s
                order by position nulls last, title
                limit %s
                """,
                (date_str, entity_types, entity_id, feature_type, LLM_SUMMARY_ITEMS),
            )
            items = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
            if not items:
                return jsonify({'summary': '', 'cached': False, 'status': 'no_items'})

            if not LLM_API_KEY:
                return jsonify({'summary': '', 'cached': False, 'status': 'llm_not_configured'}), 503

            prompt = build_serp_feature_summary_prompt(entity_type, canonical_name, feature_type, items)
            summary_text, llm_error, llm_detail = call_llm_text(prompt)
            if not summary_text:
                payload = {'summary': '', 'cached': False, 'status': 'llm_failed'}
                if llm_error:
                    payload['error'] = llm_error
                if llm_detail and request.args.get('debug', '').strip() in {'1', 'true', 'yes'}:
                    payload['detail'] = llm_detail
                return jsonify(payload), 502

            cur.execute(
                """
                insert into serp_feature_summaries
                  (date, entity_type, entity_id, entity_name, feature_type, summary_text, provider, model)
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (date, entity_type, entity_id, feature_type) do update set
                  summary_text = excluded.summary_text,
                  provider = excluded.provider,
                  model = excluded.model,
                  updated_at = now()
                """,
                (date_str, entity_type, entity_id, canonical_name, feature_type, summary_text, LLM_PROVIDER, LLM_MODEL),
            )
    finally:
        put_conn(conn)

    return jsonify({
        'summary': summary_text,
        'provider': LLM_PROVIDER,
        'model': LLM_MODEL,
        'cached': False,
    })


@app.route('/api/internal/refresh_negative_summary', methods=['POST'])
def refresh_negative_summary():
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401
    try:
        refresh_negative_summary_view()
        clear_api_cache_prefix("negative_summary:")
        return jsonify({'status': 'ok'})
    except Exception as exc:
        app.logger.exception("refresh_negative_summary failed")
        return jsonify({'error': 'refresh_failed', 'detail': str(exc)}), 500


@app.route('/api/internal/refresh_aggregates', methods=['POST'])
def refresh_aggregates():
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401
    global _refresh_in_progress, _refresh_last_status

    def run_refresh():
        global _refresh_in_progress, _refresh_last_status
        started = datetime.utcnow()
        with _refresh_lock:
            _refresh_last_status = {
                'status': 'in_progress',
                'started_at': started.isoformat() + 'Z',
                'finished_at': None,
                'duration_ms': None,
                'error': None,
            }
        app.logger.info("refresh_aggregates: started")
        try:
            refresh_negative_summary_view()
            refresh_serp_feature_daily_view()
            refresh_serp_feature_control_daily_view()
            refresh_serp_feature_daily_index_view()
            refresh_serp_feature_control_daily_index_view()
            refresh_article_daily_counts_view()
            refresh_serp_daily_counts_view()
            clear_api_cache_prefix("negative_summary:")
            clear_api_cache_prefix("daily_counts:")
            clear_api_cache_prefix("serp_features:")
            clear_api_cache_prefix("serp_feature_controls:")
            finished = datetime.utcnow()
            duration_ms = int((finished - started).total_seconds() * 1000)
            with _refresh_lock:
                _refresh_last_status = {
                    'status': 'ok',
                    'started_at': started.isoformat() + 'Z',
                    'finished_at': finished.isoformat() + 'Z',
                    'duration_ms': duration_ms,
                    'error': None,
                }
            app.logger.info("refresh_aggregates: ok duration_ms=%s", duration_ms)
        except Exception:
            app.logger.exception("refresh_aggregates failed")
            finished = datetime.utcnow()
            duration_ms = int((finished - started).total_seconds() * 1000)
            with _refresh_lock:
                _refresh_last_status = {
                    'status': 'failed',
                    'started_at': started.isoformat() + 'Z',
                    'finished_at': finished.isoformat() + 'Z',
                    'duration_ms': duration_ms,
                    'error': 'refresh_failed',
                }
        finally:
            with _refresh_lock:
                _refresh_in_progress = False

    with _refresh_lock:
        if _refresh_in_progress:
            return jsonify({'status': 'in_progress'}), 202
        _refresh_in_progress = True

    threading.Thread(target=run_refresh, daemon=True).start()
    return jsonify({'status': 'started'}), 202


@app.route('/api/internal/refresh_aggregates/status', methods=['GET'])
def refresh_aggregates_status():
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401
    with _refresh_lock:
        status = 'in_progress' if _refresh_in_progress else _refresh_last_status.get('status', 'idle')
        payload = {'status': status}
        payload.update(_refresh_last_status)
    return jsonify(payload)

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
    serp_feature_item_id = (payload.get('serp_feature_item_id') or '').strip()
    sentiment_override = payload.get('sentiment_override')
    control_override = payload.get('control_override')
    relevant_override = payload.get('relevant_override')
    note = payload.get('note') or 'dashboard edit'

    if mention_type in {'company_article', 'ceo_article'} and not mention_id:
        return jsonify({'error': 'mention_id is required'}), 400
    if mention_type == 'serp_result' and not serp_result_id:
        return jsonify({'error': 'serp_result_id is required'}), 400
    if mention_type == 'serp_feature_item' and not serp_feature_item_id:
        return jsonify({'error': 'serp_feature_item_id is required'}), 400

    if sentiment_override not in (None, 'positive', 'neutral', 'negative', 'risk', 'no_risk'):
        return jsonify({'error': 'invalid sentiment_override'}), 400
    if control_override not in (None, 'controlled', 'uncontrolled'):
        return jsonify({'error': 'invalid control_override'}), 400

    conn = get_conn()
    try:
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
            elif mention_type == 'serp_feature_item':
                cur.execute(
                    """
                    insert into serp_feature_item_overrides
                      (serp_feature_item_id, override_sentiment_label, override_control_class, note, edited_by)
                    values (%s, %s, %s, %s, %s)
                    on conflict (serp_feature_item_id) do update set
                      override_sentiment_label = excluded.override_sentiment_label,
                      override_control_class = excluded.override_control_class,
                      note = excluded.note,
                      edited_by = excluded.edited_by,
                      edited_at = now()
                    returning id
                    """,
                    (serp_feature_item_id, sentiment_override, control_override, note, user_email),
                )
            else:
                return jsonify({'error': 'invalid mention_type'}), 400
            row = cur.fetchone()
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        put_conn(conn)

    try:
        def _refresh_after_override():
            try:
                if mention_type in {'company_article', 'ceo_article'}:
                    refresh_negative_summary_view()
                    refresh_article_daily_counts_view()
                    clear_api_cache_prefix("negative_summary:")
                    clear_api_cache_prefix("daily_counts:")
                if mention_type == 'serp_feature_item':
                    refresh_serp_feature_daily_view()
                    refresh_serp_feature_control_daily_view()
                    refresh_serp_feature_daily_index_view()
                    refresh_serp_feature_control_daily_index_view()
                    clear_api_cache_prefix("serp_features:")
                    clear_api_cache_prefix("serp_feature_controls:")
                if mention_type == 'serp_result':
                    refresh_serp_daily_counts_view()
                    clear_api_cache_prefix("daily_counts:")
            except Exception:
                app.logger.exception("refresh after override failed")

        threading.Thread(target=_refresh_after_override, daemon=True).start()
    except Exception:
        app.logger.exception("override refresh setup failed")

    return jsonify({'status': 'ok', 'id': row[0] if row else None, 'refresh': 'started'})


@app.route('/api/internal/favorites', methods=['POST'])
def update_favorite():
    if PUBLIC_MODE or not ALLOW_EDITS:
        return jsonify({'error': 'editing disabled'}), 403
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401
    payload = request.get_json(force=True, silent=True) or {}
    entity_type = (payload.get('entity_type') or '').strip().lower()
    name = (payload.get('name') or '').strip()
    company = (payload.get('company') or '').strip()
    favorite = payload.get('favorite')
    if entity_type not in {'company', 'ceo'}:
        return jsonify({'error': 'invalid entity_type'}), 400
    if not name:
        return jsonify({'error': 'name is required'}), 400
    if favorite is None:
        return jsonify({'error': 'favorite is required'}), 400

    favorite = bool(favorite)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if entity_type == 'company':
                cur.execute("update companies set favorite = %s where name = %s", (favorite, name))
            else:
                if company:
                    cur.execute(
                        """
                        update ceos set favorite = %s
                        where name = %s and company_id = (select id from companies where name = %s)
                        """,
                        (favorite, name, company),
                    )
                else:
                    cur.execute("update ceos set favorite = %s where name = %s", (favorite, name))
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        put_conn(conn)

    _api_cache.pop('roster', None)
    return jsonify({'status': 'ok'})


# --------------------------- CSV builders ---------------------------


def brand_articles_daily_counts():
    params = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_rows(
        f"""
        select cad.date as date, c.name as company,
          sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='positive' then 1 else 0 end) as positive,
          sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='neutral' then 1 else 0 end) as neutral,
          sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='negative' then 1 else 0 end) as negative,
          count(*) as total,
          case when count(*) > 0
            then round((sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 6)
            else 0 end as neg_pct
        from company_article_mentions_daily cad
        join companies c on c.id = cad.company_id
        left join company_article_overrides ov on ov.company_id = cad.company_id and ov.article_id = cad.article_id
        where cad.date is not null {scope_sql}
        group by cad.date, c.name
        order by cad.date, c.name
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
        select cad.date as date, ceo.name as ceo, c.name as company,
          sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='positive' then 1 else 0 end) as positive,
          sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='neutral' then 1 else 0 end) as neutral,
          sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='negative' then 1 else 0 end) as negative,
          count(*) as total,
          case when count(*) > 0
            then round((sum(case when coalesce(ov.override_sentiment_label, cad.sentiment_label)='negative' then 1 else 0 end)::numeric / count(*))::numeric, 1)
            else 0 end as neg_pct,
          '' as theme,
          coalesce(ceo.alias, '') as alias
        from ceo_article_mentions_daily cad
        join ceos ceo on ceo.id = cad.ceo_id
        join companies c on c.id = ceo.company_id
        left join ceo_article_overrides ov on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
        where cad.date is not null {scope_sql}
        group by cad.date, ceo.name, c.name, ceo.alias
        order by cad.date, ceo.name
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
                       coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
                       coalesce(ov.override_control_class, cm.control_class) as control_class,
                       ov.override_sentiment_label as sentiment_override,
                       ov.override_control_class as control_override,
                       coalesce(cm.llm_sentiment_label, cm.llm_risk_label) as llm_label,
                       cm.id as mention_id
                from company_article_mentions_daily cad
                join companies c on c.id = cad.company_id
                join articles a on a.id = cad.article_id
                left join company_article_mentions cm on cm.company_id = cad.company_id and cm.article_id = cad.article_id
                left join company_article_overrides ov on ov.company_id = cad.company_id and ov.article_id = cad.article_id
                where cad.date = %s {scope_sql}
                order by c.name, a.title
                """,
                tuple(params)
            )
            headers = ['company','title','url','source','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
        else:
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select ceo.name as ceo, c.name as company, a.title, a.canonical_url as url, a.publisher as source,
                       coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
                       coalesce(ov.override_control_class, cm.control_class) as control_class,
                       ov.override_sentiment_label as sentiment_override,
                       ov.override_control_class as control_override,
                       coalesce(cm.llm_sentiment_label, cm.llm_risk_label) as llm_label,
                       cm.id as mention_id
                from ceo_article_mentions_daily cad
                join ceos ceo on ceo.id = cad.ceo_id
                join companies c on c.id = ceo.company_id
                join articles a on a.id = cad.article_id
                left join ceo_article_mentions cm on cm.ceo_id = cad.ceo_id and cm.article_id = cad.article_id
                left join ceo_article_overrides ov on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
                where cad.date = %s {scope_sql}
                order by ceo.name, a.title
                """,
                tuple(params)
            )
            headers = ['ceo','company','title','url','source','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
        csv_text = rows_to_csv(headers, rows)
        return Response(csv_text, content_type='text/csv')

    # table
    if entity == 'brand':
        params = [dstr]
        scope_sql, params = scope_clause("mv.company_id", params)
        rows = query_rows(
            f"""
            select mv.date as date, mv.company as company,
                   mv.positive, mv.neutral, mv.negative, mv.total, mv.neg_pct
            from article_daily_counts_mv mv
            where mv.entity_type = 'brand' and mv.date = %s {scope_sql}
            order by mv.company
            """,
            tuple(params)
        )
        headers = ['date','company','positive','neutral','negative','total','neg_pct']
    else:
        params = [dstr]
        scope_sql, params = scope_clause("mv.company_id", params)
        rows = query_rows(
            f"""
            select mv.date as date, mv.ceo as ceo, mv.company as company,
                   mv.positive, mv.neutral, mv.negative, mv.total, mv.neg_pct,
                   '' as theme, mv.alias
            from article_daily_counts_mv mv
            where mv.entity_type = 'ceo' and mv.date = %s {scope_sql}
            order by mv.ceo
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
        scope_sql, params = scope_clause("mv.company_id", params)
        rows = query_rows(
            f"""
            select mv.date as date, mv.company as company,
                   mv.total, mv.controlled, mv.negative_serp, mv.neutral_serp, mv.positive_serp
            from serp_daily_counts_mv mv
            where mv.entity_type = 'brand' and mv.date = %s {scope_sql}
            order by mv.company
            """,
            tuple(params)
        )
        headers = ['date','company','total','controlled','negative_serp','neutral_serp','positive_serp']
    else:
        params = [dstr]
        scope_sql, params = scope_clause("mv.company_id", params)
        rows = query_rows(
            f"""
            select mv.date as date, mv.ceo as ceo, mv.company as company,
                   mv.total, mv.controlled, mv.negative_serp, mv.neutral_serp, mv.positive_serp
            from serp_daily_counts_mv mv
            where mv.entity_type = 'ceo' and mv.date = %s {scope_sql}
            order by mv.ceo
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
               coalesce(c.websites, '') as websites, coalesce(c.ticker, '') as stock, coalesce(c.sector, '') as sector,
               coalesce(c.favorite, false) as company_favorite, coalesce(ceo.favorite, false) as ceo_favorite
        from companies c
        left join ceos ceo on ceo.company_id = c.id
        order by c.name, ceo.name
        """
    )
    csv_text = rows_to_csv(['CEO','Company','CEO Alias','Websites','Stock','Sector','Company Favorite','CEO Favorite'], rows)
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


def negative_articles_summary(days: int | None = None):
    lookback_days = days or int(os.environ.get('NEGATIVE_SUMMARY_LOOKBACK_DAYS', '90'))
    start_date = datetime.utcnow().date() - timedelta(days=lookback_days)

    params = [start_date]
    scope_sql, params = scope_clause("c.id", params)
    rows = query_dict(
        f"""
        select cad.date as date, c.name as company, ceo.name as ceo,
               coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
               a.title, 'brand' as article_type
        from company_article_mentions_daily cad
        join companies c on c.id = cad.company_id
        join articles a on a.id = cad.article_id
        left join company_article_overrides ov on ov.company_id = cad.company_id and ov.article_id = cad.article_id
        left join ceos ceo on ceo.company_id = c.id
        where cad.date >= %s {scope_sql}
        union all
        select cad.date as date, c.name as company, ceo.name as ceo,
               coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
               a.title, 'ceo' as article_type
        from ceo_article_mentions_daily cad
        join ceos ceo on ceo.id = cad.ceo_id
        join companies c on c.id = ceo.company_id
        join articles a on a.id = cad.article_id
        left join ceo_article_overrides ov on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
        where cad.date >= %s {scope_sql}
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


def negative_summary_view(days: int | None = None, company: str | None = None):
    lookback_days = days or int(os.environ.get('NEGATIVE_SUMMARY_LOOKBACK_DAYS', '90'))
    start_date = datetime.utcnow().date() - timedelta(days=lookback_days)
    params = [start_date]
    scope_sql, params = scope_clause("mv.company_id", params)
    company_sql = ""
    if company:
        params.append(company)
        company_sql = " and mv.company = %s"
    rows = query_dict(
        f"""
        select mv.date,
               mv.company,
               mv.ceo,
               mv.negative_count,
               mv.top_headlines,
               mv.article_type,
               mv.crisis_risk_count
        from negative_articles_summary_mv mv
        where mv.date >= %s {scope_sql}{company_sql}
        order by mv.date desc, mv.company
        """,
        tuple(params),
    )
    return rows


def clear_api_cache_prefix(prefix: str) -> None:
    if not prefix:
        return
    keys = [k for k in _api_cache.keys() if k.startswith(prefix)]
    for key in keys:
        _api_cache.pop(key, None)


def refresh_negative_summary_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently negative_articles_summary_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view negative_articles_summary_mv")
    finally:
        put_conn(conn)


def refresh_serp_feature_daily_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_daily_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_daily_mv")
    finally:
        put_conn(conn)


def refresh_serp_feature_control_daily_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_control_daily_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_control_daily_mv")
    finally:
        put_conn(conn)


def refresh_serp_feature_daily_index_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_daily_index_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_daily_index_mv")
    finally:
        put_conn(conn)


def refresh_serp_feature_control_daily_index_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_control_daily_index_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_control_daily_index_mv")
    finally:
        put_conn(conn)


def refresh_article_daily_counts_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently article_daily_counts_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view article_daily_counts_mv")
    finally:
        put_conn(conn)


def refresh_serp_daily_counts_view() -> None:
    conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_daily_counts_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_daily_counts_mv")
    finally:
        put_conn(conn)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
