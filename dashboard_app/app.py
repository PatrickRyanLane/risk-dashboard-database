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
from decimal import Decimal
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from statistics import median
from typing import Dict, Iterable, List, Tuple
from uuid import UUID

import psycopg2
from psycopg2 import extensions as pg_ext
from psycopg2.pool import PoolError, ThreadedConnectionPool
from psycopg2.extras import Json
import requests
from flask import Flask, Response, jsonify, request, send_from_directory, abort
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from crisis_event_rollups import recompute_entity_crisis_event_window
from narrative_runtime import NARRATIVE_MIN_NEG_TOP_STORIES, rollup_entity_day_narrative

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

REFRESH_LOCK_KEY = int(os.getenv("REFRESH_LOCK_KEY", "918273645"))


def _set_application_name(conn, name: str) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("set application_name = %s", (name,))
    except Exception:
        pass


def _reset_conn(conn) -> None:
    if conn is None:
        return
    try:
        if conn.closed:
            return
        conn.rollback()
    except Exception:
        pass


def _try_acquire_refresh_lock(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("select pg_try_advisory_lock(%s)", (REFRESH_LOCK_KEY,))
        return bool(cur.fetchone()[0])


def _release_refresh_lock(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("select pg_advisory_unlock(%s)", (REFRESH_LOCK_KEY,))
    except Exception:
        pass

DATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-(brand|ceo)-(articles|serps)-(modal|table)\.csv$')
STOCK_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-stock-data\.csv$')
TRENDS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-trends-data\.csv$')
NON_CRISIS_NARRATIVE_TAGS = {
    'Rebranding',
    'Mergers and acquisitions',
    'Planned Executive Turnover',
}
COMPANY_SUFFIX_TOKENS = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'companies',
    'group', 'holding', 'holdings', 'llc', 'ltd', 'limited', 'plc', 'sa',
    'ag', 'nv', 'lp', 'llp', 'the',
}
EXACT_LOOKUP_MATCH_TYPES = {
    'ticker_exact',
    'name_exact',
    'alias_exact',
    'name_normalized',
    'alias_normalized',
}
EXACT_SECTOR_MATCH_TYPES = {
    'sector_exact',
    'sector_normalized',
    'sector_singular',
}
SCREENABLE_METRICS = {
    'article_negative_count': {
        'column': 'article_negative_count',
        'label': 'negative articles',
    },
    'serp_negative_count': {
        'column': 'serp_negative_count',
        'label': 'negative SERP results',
    },
    'serp_uncontrolled_count': {
        'column': 'serp_uncontrolled_count',
        'label': 'uncontrolled SERP results',
    },
    'top_stories_negative_count': {
        'column': 'top_stories_negative_count',
        'label': 'negative Top Stories',
    },
    'top_stories_uncontrolled_count': {
        'column': 'top_stories_uncontrolled_count',
        'label': 'uncontrolled Top Stories',
    },
    'crisis_risk_count': {
        'column': 'crisis_risk_count',
        'label': 'crisis-risk article labels',
    },
}


BRAND_COMPAT_ENTITY_TYPES = ('brand', 'company')


def canonical_entity_type(entity: str) -> str:
    return 'company' if (entity or '').strip().lower() == 'brand' else 'ceo'


def compatible_entity_types(entity: str) -> List[str]:
    entity_type = canonical_entity_type(entity)
    if entity_type == 'company':
        return list(BRAND_COMPAT_ENTITY_TYPES)
    return [entity_type]


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


def narrative_display_tag(tag: str, group: str | None = None) -> str:
    txt = (tag or '').strip()
    if not txt:
        return ''
    g = (group or '').strip().lower()
    if g == 'non_crisis' or txt in NON_CRISIS_NARRATIVE_TAGS:
        return f"{txt} (non-crisis)"
    return txt


def narrative_group_for_tag(
    tag: str,
    fallback_group: str | None = None,
    fallback_is_crisis: bool | None = None,
) -> str | None:
    txt = (tag or '').strip()
    group = (fallback_group or '').strip().lower()
    if group in {'crisis', 'non_crisis'}:
        return group
    if txt in NON_CRISIS_NARRATIVE_TAGS:
        return 'non_crisis'
    if fallback_is_crisis is True:
        return 'crisis'
    if fallback_is_crisis is False:
        return 'non_crisis'
    if txt:
        return 'crisis'
    return None


def narrative_tag_count_map(raw_counts) -> Dict[str, int]:
    if not isinstance(raw_counts, dict):
        return {}
    counts: Dict[str, int] = {}
    for raw_tag, raw_value in raw_counts.items():
        tag = (str(raw_tag) or '').strip()
        if not tag:
            continue
        try:
            count = int(raw_value or 0)
        except (TypeError, ValueError):
            count = 0
        counts[tag] = max(count, 0)
    return counts


def expand_rollup_narrative_rows(rows: List[Dict]) -> List[Dict]:
    expanded: List[Dict] = []
    for row in rows:
        primary_tag = (row.get('narrative_primary_tag') or '').strip()
        primary_group = (row.get('narrative_primary_group') or '').strip().lower() or None
        primary_norm = primary_tag.casefold() if primary_tag else ''
        tag_counts = narrative_tag_count_map(row.get('tag_counts'))
        count_lookup = {tag.casefold(): count for tag, count in tag_counts.items()}
        ordered_tags: List[str] = []
        seen = set()

        for tag in tag_counts.keys():
            norm = tag.casefold()
            if norm not in seen:
                ordered_tags.append(tag)
                seen.add(norm)

        raw_tags = row.get('narrative_tags') or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        for raw_tag in raw_tags:
            tag = (str(raw_tag) or '').strip()
            if not tag:
                continue
            norm = tag.casefold()
            if norm not in seen:
                ordered_tags.append(tag)
                seen.add(norm)

        if primary_tag and primary_norm not in seen:
            ordered_tags.insert(0, primary_tag)
            seen.add(primary_norm)

        for tag in ordered_tags:
            norm = tag.casefold()
            group = narrative_group_for_tag(
                tag,
                primary_group if norm == primary_norm else None,
                row.get('narrative_is_crisis') if norm == primary_norm else None,
            )
            count = count_lookup.get(norm)
            if count is None:
                if norm == primary_norm:
                    try:
                        count = int(
                            row.get('negative_item_count')
                            or row.get('supporting_negative_items')
                            or 0
                        )
                    except (TypeError, ValueError):
                        count = 0
                else:
                    count = 1
            expanded.append({
                **row,
                'narrative_primary_tag': tag,
                'narrative_primary_group': group,
                'narrative_is_crisis': group == 'crisis' if group is not None else None,
                'negative_item_count': max(int(count or 0), 0),
            })
    return expanded


def ensure_entity_crisis_tag_daily_table(cur) -> None:
    cur.execute(
        """
        create table if not exists entity_crisis_tag_daily (
          date date not null,
          entity_type text not null,
          entity_id uuid,
          entity_name text not null,
          primary_tag text,
          primary_group text,
          tags text[],
          is_crisis boolean,
          negative_top_stories_count int not null default 0,
          tagged_item_count int not null default 0,
          unmatched_negative_items int not null default 0,
          supporting_negative_items int not null default 0,
          tag_counts jsonb not null default '{}'::jsonb,
          narrative_rule_version text,
          tagged_at timestamptz,
          created_at timestamptz not null default now(),
          updated_at timestamptz not null default now()
        )
        """
    )
    cur.execute(
        """
        create unique index if not exists entity_crisis_tag_daily_unique_idx
          on entity_crisis_tag_daily (date, entity_type, entity_name)
        """
    )


def clear_narrative_api_caches() -> None:
    clear_api_cache_prefix("narrative_tags:")
    clear_api_cache_prefix("narrative_timeline:")
    clear_api_cache_prefix("narrative_overlay:")
    clear_api_cache_prefix("insights_aggregate_crisis_patterns:")
    clear_api_cache_prefix("insights_aggregate_industry_durations:")
    clear_api_cache_prefix("insights_find_storylines:")
    clear_api_cache_prefix("insights_crisis_brand_impact:")


def load_serp_feature_item_narrative_context(cur, serp_feature_item_id: str) -> Dict | None:
    cur.execute(
        """
        select sfi.date,
               sfi.entity_type,
               sfi.entity_id,
               sfi.entity_name,
               sfi.feature_type
        from serp_feature_items sfi
        where sfi.id = %s
        """,
        (serp_feature_item_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        'date': row[0],
        'entity_type': row[1],
        'entity_id': row[2],
        'entity_name': row[3],
        'feature_type': row[4],
    }


def load_company_article_crisis_event_context(cur, mention_id: str) -> Dict | None:
    cur.execute(
        """
        select cam.company_id,
               c.name,
               min(cad.date) as first_date
        from company_article_mentions cam
        join companies c on c.id = cam.company_id
        left join company_article_mentions_daily cad
          on cad.company_id = cam.company_id
         and cad.article_id = cam.article_id
        where cam.id = %s
        group by cam.company_id, c.name
        """,
        (mention_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    start_date = row[2] or datetime.utcnow().date()
    return {
        'entity_types': ['brand'],
        'entity_id': row[0],
        'entity_name': row[1] or '',
        'start_date': start_date,
        'end_date': datetime.utcnow().date(),
    }


def load_ceo_article_crisis_event_context(cur, mention_id: str) -> Dict | None:
    cur.execute(
        """
        select cem.ceo_id,
               ceo.name,
               min(cad.date) as first_date
        from ceo_article_mentions cem
        join ceos ceo on ceo.id = cem.ceo_id
        left join ceo_article_mentions_daily cad
          on cad.ceo_id = cem.ceo_id
         and cad.article_id = cem.article_id
        where cem.id = %s
        group by cem.ceo_id, ceo.name
        """,
        (mention_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    start_date = row[2] or datetime.utcnow().date()
    return {
        'entity_types': ['ceo'],
        'entity_id': row[0],
        'entity_name': row[1] or '',
        'start_date': start_date,
        'end_date': datetime.utcnow().date(),
    }


def recompute_entity_day_narrative_rollup(
    cur,
    *,
    row_date,
    entity_type: str,
    entity_id,
    entity_name: str,
) -> Dict:
    ensure_entity_crisis_tag_daily_table(cur)
    cur.execute(
        """
        select sfi.id,
               sfi.title,
               sfi.snippet,
               sfi.url,
               sfi.source,
               coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) as sentiment_label,
               coalesce(sfi.finance_routine, false) as finance_routine
        from serp_feature_items sfi
        left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
        where sfi.date = %s
          and sfi.entity_type = %s
          and sfi.entity_name = %s
          and sfi.feature_type = 'top_stories_items'
        order by sfi.position nulls last, sfi.id
        """,
        (row_date, entity_type, entity_name),
    )
    rows = cur.fetchall()
    if not rows:
        cur.execute(
            """
            delete from entity_crisis_tag_daily
            where date = %s
              and entity_type = %s
              and entity_name = %s
            """,
            (row_date, entity_type, entity_name),
        )
        return {'primary_tag': None, 'updated_items': 0}

    items = []
    item_ids = []
    for item_id, title, snippet, url, source, sentiment_label, finance_routine in rows:
        item_ids.append(item_id)
        items.append({
            'title': title or '',
            'snippet': snippet or '',
            'url': url or '',
            'source': source or '',
            'sentiment_label': sentiment_label,
            'finance_routine': bool(finance_routine),
        })

    rollup = rollup_entity_day_narrative(
        items,
        min_negative_top_stories=NARRATIVE_MIN_NEG_TOP_STORIES,
    )
    now = datetime.utcnow()

    update_sql = """
        update serp_feature_items
           set narrative_primary_tag = %s,
               narrative_primary_group = %s,
               narrative_tags = %s,
               narrative_is_crisis = %s,
               narrative_rule_version = %s,
               narrative_tagged_at = %s,
               updated_at = now()
         where id = %s
    """
    update_rows = []
    for item_id, tag in zip(item_ids, rollup.get('item_results') or []):
        update_rows.append((
            tag.get('primary_tag') or None,
            tag.get('primary_group') or None,
            tag.get('tags') or None,
            tag.get('is_crisis'),
            tag.get('rule_version') or None,
            now if tag.get('primary_tag') else None,
            item_id,
        ))
    if update_rows:
        cur.executemany(update_sql, update_rows)

    cur.execute(
        """
        insert into entity_crisis_tag_daily (
          date,
          entity_type,
          entity_id,
          entity_name,
          primary_tag,
          primary_group,
          tags,
          is_crisis,
          negative_top_stories_count,
          tagged_item_count,
          unmatched_negative_items,
          supporting_negative_items,
          tag_counts,
          narrative_rule_version,
          tagged_at
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (date, entity_type, entity_name) do update set
          entity_id = excluded.entity_id,
          primary_tag = excluded.primary_tag,
          primary_group = excluded.primary_group,
          tags = excluded.tags,
          is_crisis = excluded.is_crisis,
          negative_top_stories_count = excluded.negative_top_stories_count,
          tagged_item_count = excluded.tagged_item_count,
          unmatched_negative_items = excluded.unmatched_negative_items,
          supporting_negative_items = excluded.supporting_negative_items,
          tag_counts = excluded.tag_counts,
          narrative_rule_version = excluded.narrative_rule_version,
          tagged_at = excluded.tagged_at,
          updated_at = now()
        """,
        (
            row_date,
            entity_type,
            entity_id,
            entity_name,
            rollup.get('primary_tag') or None,
            rollup.get('primary_group') or None,
            rollup.get('tags') or None,
            rollup.get('is_crisis'),
            int(rollup.get('negative_item_count') or 0),
            int(rollup.get('tagged_item_count') or 0),
            int(rollup.get('unmatched_negative_items') or 0),
            int(rollup.get('supporting_negative_items') or 0),
            Json(rollup.get('tag_counts') or {}),
            rollup.get('rule_version') or None,
            now if rollup.get('primary_tag') else None,
        ),
    )
    return {
        'primary_tag': rollup.get('primary_tag'),
        'updated_items': len(update_rows),
    }


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
        conn = psycopg2.connect(DB_DSN)
        _set_application_name(conn, "risk_dashboard_web")
        _reset_conn(conn)
        return conn
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_pool = ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, dsn=DB_DSN)
    try:
        conn = _db_pool.getconn()
        _set_application_name(conn, "risk_dashboard_web")
        _reset_conn(conn)
        return conn
    except PoolError:
        app.logger.warning("db_pool_exhausted_fallback")
        conn = psycopg2.connect(DB_DSN)
        _set_application_name(conn, "risk_dashboard_web")
        _reset_conn(conn)
        with _db_fallback_lock:
            _db_fallback_ids.add(id(conn))
        return conn


def get_autocommit_conn(app_name: str) -> psycopg2.extensions.connection:
    if not DB_DSN:
        raise RuntimeError('DATABASE_URL is required')
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    _set_application_name(conn, app_name)
    with _db_fallback_lock:
        _db_fallback_ids.add(id(conn))
    return conn


def put_conn(conn) -> None:
    if conn is None:
        return
    _reset_conn(conn)
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
    try:
        _reset_conn(conn)
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
        _reset_conn(conn)
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
        _reset_conn(conn)
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
        _reset_conn(conn)
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
            elif isinstance(v, UUID):
                clean[k] = str(v)
            elif isinstance(v, Decimal):
                clean[k] = int(v) if v == v.to_integral_value() else float(v)
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


def analytics_entity_type(entity: str) -> str:
    return 'brand' if canonical_entity_type(entity) == 'company' else 'ceo'


def normalize_lookup_text(text: str, strip_company_suffixes: bool = False) -> str:
    value = (text or '').strip().casefold()
    if not value:
        return ''
    value = value.replace('&', ' and ')
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    tokens = [token for token in value.split() if token]
    if strip_company_suffixes:
        while tokens and tokens[-1] in COMPANY_SUFFIX_TOKENS:
            tokens.pop()
    return ' '.join(tokens)


def singularize_lookup_token(token: str) -> str:
    if not token:
        return token
    if token.endswith('ies') and len(token) > 3:
        return token[:-3] + 'y'
    if token.endswith('ses') and len(token) > 3:
        return token[:-2]
    if token.endswith('s') and len(token) > 3 and not token.endswith(('ss', 'us', 'is')):
        return token[:-1]
    return token


def normalized_sector_keys(text: str) -> List[str]:
    normalized = normalize_lookup_text(text)
    if not normalized:
        return []
    tokens = normalized.split()
    singular = ' '.join(singularize_lookup_token(token) for token in tokens)
    keys = []
    for candidate in (normalized, singular):
        if candidate and candidate not in keys:
            keys.append(candidate)
    return keys


def _score_lookup_candidate(
    entity: str,
    query: str,
    row: Dict,
) -> Tuple[float, str]:
    query_raw = (query or '').strip()
    query_lower = query_raw.casefold()
    query_full = normalize_lookup_text(query_raw)
    query_base = normalize_lookup_text(query_raw, strip_company_suffixes=(canonical_entity_type(entity) == 'company'))
    if not query_raw or not query_full:
        return 0.0, ''

    entity_name = (row.get('entity_name') or '').strip()
    alias = (row.get('alias') or '').strip()
    ticker = (row.get('ticker') or '').strip()
    name_full = normalize_lookup_text(entity_name)
    name_base = normalize_lookup_text(entity_name, strip_company_suffixes=(canonical_entity_type(entity) == 'company'))
    alias_full = normalize_lookup_text(alias)
    alias_base = normalize_lookup_text(alias, strip_company_suffixes=(canonical_entity_type(entity) == 'company'))

    if canonical_entity_type(entity) == 'company' and ticker and query_lower == ticker.casefold():
        return 1.0, 'ticker_exact'
    if query_lower == entity_name.casefold():
        return 0.995, 'name_exact'
    if alias and query_lower == alias.casefold():
        return 0.992, 'alias_exact'
    if query_full == name_full:
        return 0.99, 'name_normalized'
    if alias and query_full == alias_full:
        return 0.988, 'alias_normalized'
    if query_base and query_base == name_base:
        return 0.985, 'name_base'
    if alias_base and query_base and query_base == alias_base:
        return 0.982, 'alias_base'

    if len(query_base) >= 4 and name_base.startswith(query_base):
        return 0.955, 'name_prefix'
    if alias_base and len(query_base) >= 4 and alias_base.startswith(query_base):
        return 0.95, 'alias_prefix'
    if len(query_base) >= 4 and query_base in name_base:
        return 0.935, 'name_contains'
    if alias_base and len(query_base) >= 4 and query_base in alias_base:
        return 0.93, 'alias_contains'

    scores = []
    if name_base:
        scores.append((SequenceMatcher(None, query_base, name_base).ratio(), 'name_fuzzy'))
        scores.append((SequenceMatcher(None, query_full, name_full).ratio(), 'name_fuzzy'))
    if alias_base:
        scores.append((SequenceMatcher(None, query_base, alias_base).ratio(), 'alias_fuzzy'))
        scores.append((SequenceMatcher(None, query_full, alias_full).ratio(), 'alias_fuzzy'))

    if canonical_entity_type(entity) == 'company' and ticker:
        ticker_score = SequenceMatcher(None, query_lower, ticker.casefold()).ratio()
        scores.append((ticker_score, 'ticker_fuzzy'))

    if not scores:
        return 0.0, ''
    return max(scores, key=lambda item: item[0])


def _score_sector_candidate(query: str, sector: str) -> Tuple[float, str]:
    query_raw = (query or '').strip()
    sector_raw = (sector or '').strip()
    if not query_raw or not sector_raw:
        return 0.0, ''

    query_lower = query_raw.casefold()
    sector_lower = sector_raw.casefold()
    if query_lower == sector_lower:
        return 1.0, 'sector_exact'

    query_keys = normalized_sector_keys(query_raw)
    sector_keys = normalized_sector_keys(sector_raw)
    if not query_keys or not sector_keys:
        return 0.0, ''

    if query_keys[0] == sector_keys[0]:
        return 0.99, 'sector_normalized'
    if len(query_keys) > 1 and query_keys[1] == sector_keys[-1]:
        return 0.985, 'sector_singular'

    best = (0.0, '')
    for query_key in query_keys:
        for sector_key in sector_keys:
            if len(query_key) >= 4 and sector_key.startswith(query_key):
                best = max(best, (0.955, 'sector_prefix'), key=lambda item: item[0])
            if len(query_key) >= 4 and query_key in sector_key:
                best = max(best, (0.94, 'sector_contains'), key=lambda item: item[0])
            ratio = SequenceMatcher(None, query_key, sector_key).ratio()
            if ratio > best[0]:
                best = (ratio, 'sector_fuzzy')
    return best


def entity_lookup_suggestions(entity: str, entity_name: str, limit: int = 5) -> List[Dict]:
    entity_name = (entity_name or '').strip()
    if not entity_name:
        return []

    if canonical_entity_type(entity) == 'company':
        params: List = []
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select c.id as entity_id,
                   c.id as company_id,
                   null::uuid as ceo_id,
                   c.name as entity_name,
                   c.name as company,
                   ''::text as ceo,
                   c.ticker as ticker,
                   null::text as alias,
                   'brand'::text as analytics_entity_type
            from companies c
            where 1=1 {scope_sql}
            order by c.name
            """,
            tuple(params),
        )
    else:
        params = []
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select ceo.id as entity_id,
                   c.id as company_id,
                   ceo.id as ceo_id,
                   ceo.name as entity_name,
                   c.name as company,
                   ceo.name as ceo,
                   null::text as ticker,
                   ceo.alias as alias,
                   'ceo'::text as analytics_entity_type
            from ceos ceo
            join companies c on c.id = ceo.company_id
            where 1=1 {scope_sql}
            order by ceo.name, c.name
            """,
            tuple(params),
        )

    scored: List[Dict] = []
    for row in rows:
        score, match_type = _score_lookup_candidate(entity, entity_name, row)
        if score < 0.72:
            continue
        candidate = dict(row)
        candidate['match_score'] = round(float(score), 4)
        candidate['match_type'] = match_type
        scored.append(candidate)

    scored.sort(
        key=lambda row: (
            -(row.get('match_score') or 0),
            str(row.get('entity_name') or '').casefold(),
            str(row.get('company') or '').casefold(),
        )
    )
    return scored[:limit]


def sector_lookup_suggestions(sector_name: str, limit: int = 5) -> List[Dict]:
    sector_name = (sector_name or '').strip()
    if not sector_name:
        return []

    params: List = []
    scope_sql, params = scope_clause("c.id", params)
    rows = query_dict(
        f"""
        select c.sector as sector,
               count(*) as company_count
        from companies c
        where coalesce(c.sector, '') <> ''
          {scope_sql}
        group by c.sector
        order by c.sector
        """,
        tuple(params),
    )

    scored: List[Dict] = []
    for row in rows:
        score, match_type = _score_sector_candidate(sector_name, row.get('sector') or '')
        if score < 0.72:
            continue
        candidate = dict(row)
        candidate['match_score'] = round(float(score), 4)
        candidate['match_type'] = match_type
        scored.append(candidate)

    scored.sort(
        key=lambda row: (
            -(row.get('match_score') or 0),
            -int(row.get('company_count') or 0),
            str(row.get('sector') or '').casefold(),
        )
    )
    return scored[:limit]


def resolve_sector_lookup(sector_name: str) -> Dict | None:
    suggestions = sector_lookup_suggestions(sector_name, limit=2)
    if not suggestions:
        return None
    top = suggestions[0]
    runner_up = suggestions[1] if len(suggestions) > 1 else None
    if (top.get('match_score') or 0) < 0.82:
        return None
    if runner_up and (top.get('match_score') or 0) < 0.99:
        if (runner_up.get('match_score') or 0) >= (top.get('match_score') or 0) - 0.02:
            return None
    resolved = dict(top)
    resolved['requested_sector_name'] = (sector_name or '').strip()
    return resolved


def sector_payload(resolved: Dict) -> Dict:
    return {
        'sector': resolved.get('sector') or '',
        'company_count': int(resolved.get('company_count') or 0),
        'requested_sector_name': resolved.get('requested_sector_name') or resolved.get('sector') or '',
        'match_type': resolved.get('match_type') or 'sector_exact',
        'match_score': float(resolved.get('match_score') or 1.0),
    }


def sector_lookup_resolution_status(resolved: Dict | None) -> str:
    if not resolved:
        return 'not_found'
    match_type = (resolved.get('match_type') or '').strip()
    score = float(resolved.get('match_score') or 0.0)
    if match_type in EXACT_SECTOR_MATCH_TYPES or score >= 0.99:
        return 'exact'
    return 'fuzzy'


def dedupe_sector_payloads(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for row in rows:
        payload = sector_payload(row)
        key = str(payload.get('sector') or '').casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def resolve_entity_lookup(entity: str, entity_name: str) -> Dict | None:
    entity_name = (entity_name or '').strip()
    if not entity_name:
        return None

    if canonical_entity_type(entity) == 'company':
        params = [entity_name]
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select c.id as entity_id,
                   c.id as company_id,
                   null::uuid as ceo_id,
                   c.name as entity_name,
                   c.name as company,
                   ''::text as ceo,
                   'brand'::text as analytics_entity_type
            from companies c
            where lower(c.name) = lower(%s) {scope_sql}
            order by c.name
            limit 1
            """,
            tuple(params),
        )
        if rows:
            row = dict(rows[0])
            row['match_type'] = 'name_exact'
            row['match_score'] = 1.0
            row['requested_entity_name'] = entity_name
            return row

        suggestions = entity_lookup_suggestions(entity, entity_name, limit=2)
        if not suggestions:
            return None
        top = suggestions[0]
        runner_up = suggestions[1] if len(suggestions) > 1 else None
        if (top.get('match_score') or 0) < 0.84:
            return None
        if runner_up and (top.get('match_score') or 0) < 0.98:
            if (runner_up.get('match_score') or 0) >= (top.get('match_score') or 0) - 0.015:
                return None
        top['requested_entity_name'] = entity_name
        return top

    params = [entity_name]
    scope_sql, params = scope_clause("c.id", params)
    rows = query_dict(
        f"""
        select ceo.id as entity_id,
               c.id as company_id,
               ceo.id as ceo_id,
               ceo.name as entity_name,
               c.name as company,
               ceo.name as ceo,
               'ceo'::text as analytics_entity_type
        from ceos ceo
        join companies c on c.id = ceo.company_id
        where lower(ceo.name) = lower(%s) {scope_sql}
        order by ceo.name, c.name
        limit 1
        """,
        tuple(params),
    )
    if rows:
        row = dict(rows[0])
        row['match_type'] = 'name_exact'
        row['match_score'] = 1.0
        row['requested_entity_name'] = entity_name
        return row

    suggestions = entity_lookup_suggestions(entity, entity_name, limit=2)
    if not suggestions:
        return None
    top = suggestions[0]
    runner_up = suggestions[1] if len(suggestions) > 1 else None
    if (top.get('match_score') or 0) < 0.88:
        return None
    if runner_up and (top.get('match_score') or 0) < 0.985:
        if (runner_up.get('match_score') or 0) >= (top.get('match_score') or 0) - 0.01:
            return None
    top['requested_entity_name'] = entity_name
    return top


def coerce_date(value):
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, 'year') and hasattr(value, 'month') and hasattr(value, 'day'):
        return value
    text = str(value or '').strip()
    if not text:
        return None
    return datetime.strptime(text[:10], '%Y-%m-%d').date()


def rows_within_dates(rows: List[Dict], start_date, end_date) -> List[Dict]:
    out = []
    for row in rows:
        row_date = coerce_date(row.get('date'))
        if row_date is None:
            continue
        if start_date <= row_date <= end_date:
            out.append(row)
    return out


def sum_metric(rows: List[Dict], key: str) -> int:
    total = 0
    for row in rows:
        total += int(row.get(key) or 0)
    return total


def avg_metric(rows: List[Dict], key: str) -> float:
    total = 0.0
    count = 0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        total += float(value)
        count += 1
    if count == 0:
        return 0.0
    return round(total / count, 6)


def top_stories_streak_days(rows: List[Dict], threshold: int = 4) -> int:
    streak = 0
    for row in reversed(rows):
        if int(row.get('top_stories_negative_count') or 0) >= threshold:
            streak += 1
        else:
            break
    return streak


def rolling_window_anchor_dates(rows: List[Dict], max_windows: int) -> List:
    dates = []
    for row in rows:
        row_date = coerce_date(row.get('date'))
        if row_date is not None:
            dates.append(row_date)
    if not dates:
        return []

    distinct_dates = sorted(set(dates))
    anchors = []
    used = set()
    target = distinct_dates[-1]
    while len(anchors) < max_windows:
        candidate = next((d for d in reversed(distinct_dates) if d <= target and d not in used), None)
        if candidate is None:
            break
        anchors.append(candidate)
        used.add(candidate)
        target = candidate - timedelta(days=7)
    anchors.sort()
    return anchors


def build_trailing_window_rollups(rows: List[Dict], max_windows: int) -> List[Dict]:
    rollups = []
    for anchor in rolling_window_anchor_dates(rows, max_windows):
        window_rows = rows_within_dates(rows, anchor - timedelta(days=6), anchor)
        if not window_rows:
            continue
        start_dates = [coerce_date(row.get('date')) for row in window_rows if coerce_date(row.get('date')) is not None]
        rollups.append({
            'window_start': min(start_dates).isoformat() if start_dates else anchor.isoformat(),
            'window_end': anchor.isoformat(),
            'article_negative_7d': sum_metric(window_rows, 'article_negative_count'),
            'article_total_7d': sum_metric(window_rows, 'article_total_count'),
            'article_negative_pct_avg_7d': avg_metric(window_rows, 'article_negative_pct'),
            'serp_negative_7d': sum_metric(window_rows, 'serp_negative_count'),
            'serp_total_7d': sum_metric(window_rows, 'serp_total_count'),
            'serp_controlled_7d': sum_metric(window_rows, 'serp_controlled_count'),
            'serp_uncontrolled_7d': sum_metric(window_rows, 'serp_uncontrolled_count'),
            'top_stories_total_7d': sum_metric(window_rows, 'top_stories_total_count'),
            'top_stories_negative_7d': sum_metric(window_rows, 'top_stories_negative_count'),
            'top_stories_controlled_7d': sum_metric(window_rows, 'top_stories_controlled_count'),
            'top_stories_uncontrolled_7d': sum_metric(window_rows, 'top_stories_uncontrolled_count'),
            'top_stories_crisis_days_7d': sum(
                1 for row in window_rows if int(row.get('top_stories_negative_count') or 0) >= 4
            ),
            'crisis_risk_7d': sum_metric(window_rows, 'crisis_risk_count'),
        })
    return rollups


def classify_search_impact(current_window: Dict[str, int]) -> str:
    news_signal = current_window.get('article_negative_count', 0) >= 7
    negative_search_signal = (
        current_window.get('serp_negative_count', 0) >= 3
        or current_window.get('top_stories_negative_count', 0) >= 4
    )
    uncontrolled_search_signal = (
        current_window.get('serp_uncontrolled_count', 0) >= 5
        or current_window.get('top_stories_uncontrolled_count', 0) >= 4
    )
    if negative_search_signal and news_signal:
        return 'news_and_search_negative'
    if negative_search_signal:
        return 'search_negative'
    if uncontrolled_search_signal and news_signal:
        return 'news_and_search_uncontrolled'
    if uncontrolled_search_signal:
        return 'search_uncontrolled'
    if news_signal:
        return 'news_only'
    return 'muted'


def build_search_nuance(current_window: Dict[str, int]) -> Dict:
    negative_search_signal = (
        current_window.get('serp_negative_count', 0) >= 3
        or current_window.get('top_stories_negative_count', 0) >= 4
    )
    control_gap_signal = (
        current_window.get('serp_uncontrolled_count', 0) >= 5
        or current_window.get('top_stories_uncontrolled_count', 0) >= 4
    )
    negative_search_items = (
        current_window.get('serp_negative_count', 0)
        + current_window.get('top_stories_negative_count', 0)
    )
    uncontrolled_search_items = (
        current_window.get('serp_uncontrolled_count', 0)
        + current_window.get('top_stories_uncontrolled_count', 0)
    )

    if negative_search_signal and control_gap_signal:
        label = 'negative_visibility_and_control_gap'
        interpretation = (
            'Negative search visibility is present and the search results are weakly controlled. '
            'Treat this as both reputational spillover and a SERP control opportunity.'
        )
    elif negative_search_signal:
        label = 'negative_visibility'
        interpretation = (
            'Negative search visibility is present, but weak control is not the dominant issue in the current window.'
        )
    elif control_gap_signal:
        label = 'control_gap_without_negative_visibility'
        interpretation = (
            'Search control is weak, but the current search signal is mostly neutral or non-negative. '
            'Treat this as a control opportunity rather than confirmed negative spillover.'
        )
    else:
        label = 'low_or_controlled_search_signal'
        interpretation = 'Search does not currently show a strong negative visibility or control-gap signal.'

    return {
        'label': label,
        'negative_search_signal': negative_search_signal,
        'control_gap_signal': control_gap_signal,
        'negative_search_items_7d': negative_search_items,
        'uncontrolled_search_items_7d': uncontrolled_search_items,
        'interpretation': interpretation,
    }


def summarize_evidence_rows(rows: List[Dict]) -> Dict:
    summary = {
        'total_rows': len(rows),
        'by_evidence_type': {},
        'by_included_reason': {},
        'search_rows_by_included_reason': {},
        'search_rows_by_sentiment': {},
        'neutral_uncontrolled_search_rows': 0,
        'negative_search_rows': 0,
        'search_interpretation': '',
    }

    for row in rows:
        evidence_type = (row.get('evidence_type') or '').strip() or 'unknown'
        included_reason = (row.get('included_reason') or '').strip() or 'unspecified'
        sentiment = ((row.get('sentiment_label') or '').strip() or 'unspecified').lower()

        summary['by_evidence_type'][evidence_type] = summary['by_evidence_type'].get(evidence_type, 0) + 1
        summary['by_included_reason'][included_reason] = summary['by_included_reason'].get(included_reason, 0) + 1

        if evidence_type not in {'serp', 'top_stories'}:
            continue

        summary['search_rows_by_included_reason'][included_reason] = (
            summary['search_rows_by_included_reason'].get(included_reason, 0) + 1
        )
        summary['search_rows_by_sentiment'][sentiment] = summary['search_rows_by_sentiment'].get(sentiment, 0) + 1

        if included_reason in {'negative', 'negative_and_uncontrolled'}:
            summary['negative_search_rows'] += 1
        if included_reason == 'uncontrolled' and sentiment in {'neutral', 'positive', 'unspecified'}:
            summary['neutral_uncontrolled_search_rows'] += 1

    if summary['negative_search_rows'] > 0 and summary['neutral_uncontrolled_search_rows'] > 0:
        summary['search_interpretation'] = (
            'Search evidence is mixed: some results are negatively coded, while others are neutral but weakly controlled.'
        )
    elif summary['negative_search_rows'] > 0:
        summary['search_interpretation'] = 'Search evidence is predominantly negative in the selected window.'
    elif summary['neutral_uncontrolled_search_rows'] > 0:
        summary['search_interpretation'] = (
            'Search evidence is mostly neutral but weakly controlled, which suggests a SERP control issue more than direct negative spillover.'
        )
    else:
        summary['search_interpretation'] = 'Search evidence is limited or not strongly directional in the selected window.'

    return summary


def entity_payload(resolved: Dict) -> Dict:
    return {
        'entity_type': resolved.get('analytics_entity_type'),
        'entity_id': str(resolved['entity_id']) if resolved.get('entity_id') else None,
        'company_id': str(resolved['company_id']) if resolved.get('company_id') else None,
        'ceo_id': str(resolved['ceo_id']) if resolved.get('ceo_id') else None,
        'entity_name': resolved.get('entity_name') or '',
        'company': resolved.get('company') or '',
        'ceo': resolved.get('ceo') or '',
        'requested_entity_name': resolved.get('requested_entity_name') or resolved.get('entity_name') or '',
        'match_type': resolved.get('match_type') or 'name_exact',
        'match_score': float(resolved.get('match_score') or 1.0),
    }


def lookup_resolution_status(resolved: Dict | None) -> str:
    if not resolved:
        return 'not_found'
    match_type = (resolved.get('match_type') or '').strip()
    score = float(resolved.get('match_score') or 0.0)
    if match_type in EXACT_LOOKUP_MATCH_TYPES or score >= 0.99:
        return 'exact'
    return 'fuzzy'


def dedupe_entity_payloads(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for row in rows:
        payload = entity_payload(row)
        key = (payload.get('entity_type'), payload.get('entity_id'))
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def consecutive_day_durations(days: List) -> List[int]:
    clean_days = sorted({day for day in days if day is not None})
    if not clean_days:
        return []
    durations: List[int] = []
    streak = 1
    previous = clean_days[0]
    for day in clean_days[1:]:
        if day == previous + timedelta(days=1):
            streak += 1
        else:
            durations.append(streak)
            streak = 1
        previous = day
    durations.append(streak)
    return durations


def consecutive_day_windows(days: List) -> List[Tuple]:
    clean_days = sorted({day for day in days if day is not None})
    if not clean_days:
        return []
    windows: List[Tuple] = []
    start = clean_days[0]
    previous = clean_days[0]
    for day in clean_days[1:]:
        if day == previous + timedelta(days=1):
            previous = day
            continue
        windows.append((start, previous, (previous - start).days + 1))
        start = day
        previous = day
    windows.append((start, previous, (previous - start).days + 1))
    return windows


def latest_negative_top_stories_narrative_date(entity: str, sector: str = ''):
    entity_type = canonical_entity_type(entity)
    try:
        if entity_type == 'company':
            latest_params: List = [compatible_entity_types(entity)]
            sector_sql = ""
            if sector:
                latest_params.append(f"%{sector}%")
                sector_sql = " and coalesce(c.sector, '') ilike %s"
            latest_scope_sql, latest_params = scope_clause("c.id", latest_params)
            latest_rows = query_rows(
                f"""
                select max(ecd.date)
                from entity_crisis_event_daily ecd
                join companies c on c.id = ecd.entity_id
                where ecd.entity_type = any(%s)
                  and ecd.primary_tag is not null
                  {sector_sql}
                  {latest_scope_sql}
                """,
                tuple(latest_params),
            )
            latest_date = latest_rows[0][0] if latest_rows and latest_rows[0] else None
            if latest_date is not None:
                return latest_date

        latest_params = [entity_type]
        sector_sql = ""
        if sector:
            latest_params.append(f"%{sector}%")
            sector_sql = " and coalesce(c.sector, '') ilike %s"
        latest_scope_sql, latest_params = scope_clause("c.id", latest_params)
        latest_rows = query_rows(
            f"""
            select max(ecd.date)
            from entity_crisis_event_daily ecd
            join ceos ceo on ceo.id = ecd.entity_id
            join companies c on c.id = ceo.company_id
            where ecd.entity_type = %s
              and ecd.primary_tag is not null
              {sector_sql}
              {latest_scope_sql}
            """,
            tuple(latest_params),
        )
        latest_date = latest_rows[0][0] if latest_rows and latest_rows[0] else None
        if latest_date is not None:
            return latest_date
    except Exception as exc:
        app.logger.warning("entity_crisis_event_daily latest lookup fallback: %s", exc)

    if entity_type == 'company':
        latest_params: List = [compatible_entity_types(entity)]
        sector_sql = ""
        if sector:
            latest_params.append(f"%{sector}%")
            sector_sql = " and coalesce(c.sector, '') ilike %s"
        latest_scope_sql, latest_params = scope_clause("c.id", latest_params)
        latest_rows = query_rows(
            f"""
            select max(sfi.date)
            from serp_feature_items sfi
            join companies c on c.id = sfi.entity_id
            left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
            where sfi.entity_type = any(%s)
              and sfi.feature_type = 'top_stories_items'
              and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
              and coalesce(sfi.finance_routine, false) = false
              and sfi.narrative_primary_tag is not null
              {sector_sql}
              {latest_scope_sql}
            """,
            tuple(latest_params),
        )
        return latest_rows[0][0] if latest_rows and latest_rows[0] else None

    latest_params = [entity_type]
    sector_sql = ""
    if sector:
        latest_params.append(f"%{sector}%")
        sector_sql = " and coalesce(c.sector, '') ilike %s"
    latest_scope_sql, latest_params = scope_clause("c.id", latest_params)
    latest_rows = query_rows(
        f"""
        select max(sfi.date)
        from serp_feature_items sfi
        join ceos ceo on ceo.id = sfi.entity_id
        join companies c on c.id = ceo.company_id
        left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
        where sfi.entity_type = %s
          and sfi.feature_type = 'top_stories_items'
          and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
          and coalesce(sfi.finance_routine, false) = false
          and sfi.narrative_primary_tag is not null
          {sector_sql}
          {latest_scope_sql}
        """,
        tuple(latest_params),
    )
    return latest_rows[0][0] if latest_rows and latest_rows[0] else None


def fetch_negative_top_stories_narrative_rows(entity: str, start_date, end_date, sector: str = '') -> Tuple[str, List[Dict]]:
    entity_type = canonical_entity_type(entity)
    try:
        if entity_type == 'company':
            params = [start_date, end_date, compatible_entity_types(entity)]
            sector_sql = ""
            if sector:
                params.append(f"%{sector}%")
                sector_sql = " and coalesce(c.sector, '') ilike %s"
            scope_sql, params = scope_clause("c.id", params)
            rows = query_dict(
                f"""
                select ecd.date,
                       c.id as company_id,
                       c.id as entity_id,
                       c.name as company,
                       ''::text as ceo,
                       c.name as entity_name,
                       coalesce(c.sector, '') as sector,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.is_crisis as narrative_is_crisis,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.supporting_negative_items as negative_item_count
                from entity_crisis_event_daily ecd
                join companies c on c.id = ecd.entity_id
                where ecd.date between %s and %s
                  and ecd.entity_type = any(%s)
                  and ecd.primary_tag is not null
                  {sector_sql}
                  {scope_sql}
                order by ecd.date, c.name, ecd.primary_tag
                """,
                tuple(params),
            )
            if rows:
                return 'brand', expand_rollup_narrative_rows(rows)

        params = [start_date, end_date, entity_type]
        sector_sql = ""
        if sector:
            params.append(f"%{sector}%")
            sector_sql = " and coalesce(c.sector, '') ilike %s"
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select ecd.date,
                   c.id as company_id,
                   ceo.id as entity_id,
                   c.name as company,
                   ceo.name as ceo,
                   ceo.name as entity_name,
                   coalesce(c.sector, '') as sector,
                   ecd.primary_tag as narrative_primary_tag,
                   ecd.primary_group as narrative_primary_group,
                   ecd.is_crisis as narrative_is_crisis,
                   ecd.tags as narrative_tags,
                   ecd.tag_counts,
                   ecd.supporting_negative_items as negative_item_count
            from entity_crisis_event_daily ecd
            join ceos ceo on ceo.id = ecd.entity_id
            join companies c on c.id = ceo.company_id
            where ecd.date between %s and %s
              and ecd.entity_type = %s
              and ecd.primary_tag is not null
              {sector_sql}
              {scope_sql}
            order by ecd.date, ceo.name, ecd.primary_tag
            """,
            tuple(params),
        )
        if rows:
            return 'ceo', expand_rollup_narrative_rows(rows)
    except Exception as exc:
        app.logger.warning("entity_crisis_event_daily narrative fetch fallback: %s", exc)

    if entity_type == 'company':
        params = [start_date, end_date, compatible_entity_types(entity)]
        sector_sql = ""
        if sector:
            params.append(f"%{sector}%")
            sector_sql = " and coalesce(c.sector, '') ilike %s"
        scope_sql, params = scope_clause("c.id", params)
        rows = query_dict(
            f"""
            select sfi.date,
                   c.id as company_id,
                   c.id as entity_id,
                   c.name as company,
                   ''::text as ceo,
                   c.name as entity_name,
                   coalesce(c.sector, '') as sector,
                   sfi.narrative_primary_tag,
                   sfi.narrative_primary_group,
                   sfi.narrative_is_crisis,
                   count(*) as negative_item_count
            from serp_feature_items sfi
            join companies c on c.id = sfi.entity_id
            left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
            where sfi.date between %s and %s
              and sfi.entity_type = any(%s)
              and sfi.feature_type = 'top_stories_items'
              and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
              and coalesce(sfi.finance_routine, false) = false
              and sfi.narrative_primary_tag is not null
              {sector_sql}
              {scope_sql}
            group by sfi.date, c.id, c.name, c.sector, sfi.narrative_primary_tag, sfi.narrative_primary_group, sfi.narrative_is_crisis
            order by sfi.date, c.name, sfi.narrative_primary_tag
            """,
            tuple(params),
        )
        return 'brand', rows

    params = [start_date, end_date, entity_type]
    sector_sql = ""
    if sector:
        params.append(f"%{sector}%")
        sector_sql = " and coalesce(c.sector, '') ilike %s"
    scope_sql, params = scope_clause("c.id", params)
    rows = query_dict(
        f"""
        select sfi.date,
               c.id as company_id,
               ceo.id as entity_id,
               c.name as company,
               ceo.name as ceo,
               ceo.name as entity_name,
               coalesce(c.sector, '') as sector,
               sfi.narrative_primary_tag,
               sfi.narrative_primary_group,
               sfi.narrative_is_crisis,
               count(*) as negative_item_count
        from serp_feature_items sfi
        join ceos ceo on ceo.id = sfi.entity_id
        join companies c on c.id = ceo.company_id
        left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
        where sfi.date between %s and %s
          and sfi.entity_type = %s
          and sfi.feature_type = 'top_stories_items'
          and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
          and coalesce(sfi.finance_routine, false) = false
          and sfi.narrative_primary_tag is not null
          {sector_sql}
          {scope_sql}
        group by sfi.date, c.id, ceo.id, c.name, ceo.name, c.sector, sfi.narrative_primary_tag, sfi.narrative_primary_group, sfi.narrative_is_crisis
        order by sfi.date, ceo.name, sfi.narrative_primary_tag
        """,
        tuple(params),
    )
    return 'ceo', rows


def normalized_narrative_group(tag: str, primary_group: str | None, is_crisis) -> str | None:
    group = (primary_group or '').strip().lower()
    if group in {'crisis', 'non_crisis'}:
        return group
    if tag in NON_CRISIS_NARRATIVE_TAGS:
        return 'non_crisis'
    if is_crisis is True:
        return 'crisis'
    if is_crisis is False:
        return 'non_crisis'
    return None


def resolve_insights_window(
    entity: str,
    sector: str = '',
    default_days: int = 90,
    min_days: int = 1,
    max_days: int = 365,
) -> Tuple:
    latest_available_date = latest_negative_top_stories_narrative_date(entity, sector)
    if latest_available_date is None:
        raise LookupError('no_data')

    requested_start_str = (request.args.get('start_date') or '').strip()
    requested_end_str = (request.args.get('end_date') or '').strip()

    if requested_start_str or requested_end_str:
        if not requested_start_str or not requested_end_str:
            raise ValueError('start_date and end_date are both required when using an explicit calendar window')
        try:
            requested_start_date = datetime.strptime(requested_start_str, '%Y-%m-%d').date()
            requested_end_date = datetime.strptime(requested_end_str, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError('invalid date format (YYYY-MM-DD)')
        if requested_start_date > requested_end_date:
            raise ValueError('start_date must be on or before end_date')
        if requested_start_date > latest_available_date:
            raise LookupError('no_data')
        actual_end_date = min(requested_end_date, latest_available_date)
        if requested_start_date > actual_end_date:
            raise LookupError('no_data')
        return (
            requested_start_date,
            actual_end_date,
            latest_available_date,
            (actual_end_date - requested_start_date).days + 1,
            'calendar',
            requested_start_date.isoformat(),
            requested_end_date.isoformat(),
        )

    try:
        days = int(request.args.get('days', str(default_days)) or default_days)
    except ValueError:
        days = default_days
    days = min(max(days, min_days), max_days)
    end_date = latest_available_date
    start_date = end_date - timedelta(days=days - 1)
    return (
        start_date,
        end_date,
        latest_available_date,
        days,
        'rolling',
        None,
        None,
    )


def build_storyline_candidates(analytics_type: str, rows: List[Dict]) -> List[Dict]:
    affected_key = 'brands_affected' if analytics_type == 'brand' else 'ceos_affected'
    entity_label_plural = 'brands' if analytics_type == 'brand' else 'CEOs'
    by_sector_tag: Dict[Tuple[str, str], Dict] = {}
    by_tag: Dict[str, Dict] = {}
    by_sector: Dict[str, Dict] = {}

    for row in rows:
        tag = (row.get('narrative_primary_tag') or '').strip()
        if not tag:
            continue
        day = coerce_date(row.get('date'))
        if day is None:
            continue
        sector = (row.get('sector') or '').strip() or 'Unspecified'
        entity_id = str(row.get('entity_id') or '')
        entity_name = (row.get('entity_name') or '').strip()
        negative_item_count = int(row.get('negative_item_count') or 0)
        primary_group = normalized_narrative_group(
            tag,
            row.get('narrative_primary_group'),
            row.get('narrative_is_crisis'),
        )
        tag_key = f"{tag.casefold()}::{primary_group or ''}"
        display_tag = narrative_display_tag(tag, primary_group)

        sector_tag_bucket = by_sector_tag.setdefault((sector, tag_key), {
            'sector': sector,
            'tag': tag,
            'group': primary_group,
            'display_tag': display_tag,
            'entity_days': {},
            'entity_names': {},
            'entity_totals': {},
            'entity_negative_item_totals': {},
            'days': set(),
            'total_negative_items': 0,
        })
        sector_tag_bucket['entity_days'].setdefault(entity_id, set()).add(day)
        sector_tag_bucket['entity_names'][entity_id] = entity_name
        sector_tag_bucket['entity_totals'][entity_id] = sector_tag_bucket['entity_totals'].get(entity_id, 0) + 1
        sector_tag_bucket['entity_negative_item_totals'][entity_id] = (
            sector_tag_bucket['entity_negative_item_totals'].get(entity_id, 0) + negative_item_count
        )
        sector_tag_bucket['days'].add(day)
        sector_tag_bucket['total_negative_items'] += negative_item_count

        tag_bucket = by_tag.setdefault(tag_key, {
            'tag': tag,
            'group': primary_group,
            'display_tag': display_tag,
            'sectors': set(),
            'entity_days': {},
            'entity_names': {},
            'entity_negative_item_totals': {},
            'sector_negative_item_totals': {},
            'days': set(),
            'total_negative_items': 0,
        })
        tag_bucket['sectors'].add(sector)
        tag_bucket['entity_days'].setdefault(entity_id, set()).add(day)
        tag_bucket['entity_names'][entity_id] = entity_name
        tag_bucket['entity_negative_item_totals'][entity_id] = (
            tag_bucket['entity_negative_item_totals'].get(entity_id, 0) + negative_item_count
        )
        tag_bucket['sector_negative_item_totals'][sector] = (
            tag_bucket['sector_negative_item_totals'].get(sector, 0) + negative_item_count
        )
        tag_bucket['days'].add(day)
        tag_bucket['total_negative_items'] += negative_item_count

        sector_bucket = by_sector.setdefault(sector, {
            'sector': sector,
            'entity_days': {},
            'entity_names': {},
            'entity_negative_item_totals': {},
            'tag_negative_item_totals': {},
            'days': set(),
            'total_negative_items': 0,
        })
        sector_bucket['entity_days'].setdefault(entity_id, set()).add(day)
        sector_bucket['entity_names'][entity_id] = entity_name
        sector_bucket['entity_negative_item_totals'][entity_id] = (
            sector_bucket['entity_negative_item_totals'].get(entity_id, 0) + negative_item_count
        )
        sector_bucket['tag_negative_item_totals'][display_tag] = (
            sector_bucket['tag_negative_item_totals'].get(display_tag, 0) + negative_item_count
        )
        sector_bucket['days'].add(day)
        sector_bucket['total_negative_items'] += negative_item_count

    candidates: List[Dict] = []

    for bucket in by_sector_tag.values():
        durations: List[int] = []
        for days_set in bucket['entity_days'].values():
            durations.extend(consecutive_day_durations(list(days_set)))
        if not durations:
            continue
        affected_count = len(bucket['entity_days'])
        avg_duration_days = round(sum(durations) / len(durations), 2)
        max_duration_days = max(durations)
        top_examples = sorted(
            bucket['entity_negative_item_totals'].items(),
            key=lambda item: (-item[1], bucket['entity_names'].get(item[0], '').casefold()),
        )[:3]
        score = round(
            affected_count * 6
            + bucket['total_negative_items'] * 0.35
            + avg_duration_days * 2
            + len(bucket['days']) * 0.4,
            2,
        )
        candidates.append({
            'storyline_key': f"sector_tag:{bucket['sector']}:{bucket['tag'].casefold()}",
            'storyline_type': 'sector_tag_pattern',
            'headline': f"{bucket['sector']} saw concentrated {bucket['display_tag'].lower()} pressure",
            'angle': (
                f"{affected_count} {entity_label_plural} in {bucket['sector']} showed "
                f"{bucket['display_tag']} in negative search/news coverage during the selected window."
            ),
            'why_interesting': (
                f"Average duration was {avg_duration_days} days, with a maximum streak of {max_duration_days} days "
                f"and {bucket['total_negative_items']} tagged negative evidence items."
            ),
            'score': score,
            'supporting_metrics': {
                affected_key: affected_count,
                'avg_duration_days': avg_duration_days,
                'max_duration_days': max_duration_days,
                'episode_count': len(durations),
                'total_negative_items': bucket['total_negative_items'],
                'active_days': len(bucket['days']),
                'sector': bucket['sector'],
                'display_tag': bucket['display_tag'],
            },
            'sample_entities': [bucket['entity_names'].get(entity_id, entity_id) for entity_id, _ in top_examples],
            'sample_sectors': [bucket['sector']],
        })

    for bucket in by_tag.values():
        if len(bucket['sectors']) < 2:
            continue
        durations = []
        for days_set in bucket['entity_days'].values():
            durations.extend(consecutive_day_durations(list(days_set)))
        if not durations:
            continue
        affected_count = len(bucket['entity_days'])
        avg_duration_days = round(sum(durations) / len(durations), 2)
        max_duration_days = max(durations)
        top_sectors = sorted(
            bucket['sector_negative_item_totals'].items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:3]
        top_examples = sorted(
            bucket['entity_negative_item_totals'].items(),
            key=lambda item: (-item[1], bucket['entity_names'].get(item[0], '').casefold()),
        )[:3]
        score = round(
            len(bucket['sectors']) * 7
            + affected_count * 4
            + avg_duration_days * 1.6
            + bucket['total_negative_items'] * 0.25,
            2,
        )
        candidates.append({
            'storyline_key': f"cross_sector:{bucket['tag'].casefold()}",
            'storyline_type': 'cross_sector_narrative',
            'headline': f"{bucket['display_tag']} crossed sector lines",
            'angle': (
                f"The {bucket['display_tag']} narrative appeared across {len(bucket['sectors'])} sectors and "
                f"{affected_count} {entity_label_plural} in the selected window."
            ),
            'why_interesting': (
                f"It persisted for {avg_duration_days} days on average, peaked at {max_duration_days} days, "
                f"and generated {bucket['total_negative_items']} tagged negative evidence items."
            ),
            'score': score,
            'supporting_metrics': {
                affected_key: affected_count,
                'sectors_affected': len(bucket['sectors']),
                'avg_duration_days': avg_duration_days,
                'max_duration_days': max_duration_days,
                'episode_count': len(durations),
                'total_negative_items': bucket['total_negative_items'],
                'display_tag': bucket['display_tag'],
            },
            'sample_entities': [bucket['entity_names'].get(entity_id, entity_id) for entity_id, _ in top_examples],
            'sample_sectors': [sector for sector, _ in top_sectors],
        })

    for bucket in by_sector.values():
        durations = []
        for days_set in bucket['entity_days'].values():
            durations.extend(consecutive_day_durations(list(days_set)))
        if not durations:
            continue
        affected_count = len(bucket['entity_days'])
        avg_duration_days = round(sum(durations) / len(durations), 2)
        median_duration_days = float(median(durations))
        max_duration_days = max(durations)
        top_tags = sorted(
            bucket['tag_negative_item_totals'].items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )[:3]
        top_examples = sorted(
            bucket['entity_negative_item_totals'].items(),
            key=lambda item: (-item[1], bucket['entity_names'].get(item[0], '').casefold()),
        )[:3]
        score = round(
            avg_duration_days * 3
            + affected_count * 3
            + bucket['total_negative_items'] * 0.18,
            2,
        )
        candidates.append({
            'storyline_key': f"sector_duration:{bucket['sector']}",
            'storyline_type': 'sector_duration_outlier',
            'headline': f"{bucket['sector']} crises lingered in search",
            'angle': (
                f"{bucket['sector']} showed one of the more persistent search-visible crisis patterns "
                f"for {affected_count} {entity_label_plural} in the selected window."
            ),
            'why_interesting': (
                f"Average duration was {avg_duration_days} days, the median episode lasted {median_duration_days} days, "
                f"and the strongest themes were {', '.join(tag for tag, _ in top_tags[:2]) or 'mixed'}."
            ),
            'score': score,
            'supporting_metrics': {
                affected_key: affected_count,
                'avg_duration_days': avg_duration_days,
                'median_duration_days': median_duration_days,
                'max_duration_days': max_duration_days,
                'episode_count': len(durations),
                'total_negative_items': bucket['total_negative_items'],
                'sector': bucket['sector'],
                'dominant_tags': [tag for tag, _ in top_tags],
            },
            'sample_entities': [bucket['entity_names'].get(entity_id, entity_id) for entity_id, _ in top_examples],
            'sample_sectors': [bucket['sector']],
        })

    return candidates


def build_crisis_brand_impact_summary(rows: List[Dict], start_date, end_date) -> Tuple[List[Dict], Dict[str, Dict], int, int, int, List[str], List[Dict]]:
    by_crisis: Dict[str, Dict] = {}
    affected_brand_ids = set()
    active_brand_ids = set()
    trend_dates = [
        (start_date + timedelta(days=offset)).isoformat()
        for offset in range((end_date - start_date).days + 1)
    ]

    for row in rows:
        tag = (row.get('narrative_primary_tag') or '').strip()
        if not tag:
            continue
        group = normalized_narrative_group(
            tag,
            row.get('narrative_primary_group'),
            row.get('narrative_is_crisis'),
        )
        if group != 'crisis':
            continue

        row_date = coerce_date(row.get('date'))
        brand_name = (row.get('company') or row.get('entity_name') or '').strip()
        if row_date is None or not brand_name:
            continue

        brand_id = str(row.get('company_id') or row.get('entity_id') or brand_name.casefold())
        sector_name = (row.get('sector') or '').strip()
        negative_item_count = max(int(row.get('negative_item_count') or 0), 0)
        crisis_key = tag.casefold()
        day_iso = row_date.isoformat()

        affected_brand_ids.add(brand_id)
        if row_date == end_date:
            active_brand_ids.add(brand_id)

        crisis_bucket = by_crisis.setdefault(crisis_key, {
            'tag': tag,
            'display_tag': narrative_display_tag(tag, group),
            'group': group,
            'days': set(),
            'brands': {},
            'brands_by_day': {},
            'negative_items_by_day': {},
            'total_negative_items': 0,
        })
        crisis_bucket['days'].add(row_date)
        crisis_bucket['total_negative_items'] += negative_item_count
        crisis_bucket['brands_by_day'].setdefault(day_iso, set()).add(brand_name)
        crisis_bucket['negative_items_by_day'][day_iso] = (
            crisis_bucket['negative_items_by_day'].get(day_iso, 0) + negative_item_count
        )

        brand_bucket = crisis_bucket['brands'].setdefault(brand_id, {
            'brand_id': brand_id,
            'brand': brand_name,
            'sector': sector_name,
            'days': set(),
            'total_negative_items': 0,
        })
        brand_bucket['days'].add(row_date)
        brand_bucket['total_negative_items'] += negative_item_count
        if sector_name and not brand_bucket.get('sector'):
            brand_bucket['sector'] = sector_name

    crisis_rows: List[Dict] = []
    detail_lookup: Dict[str, Dict] = {}
    total_brand_days = 0
    trend_rows: List[Dict] = []

    for crisis_key, crisis_bucket in by_crisis.items():
        brand_rows: List[Dict] = []
        active_brands_latest = 0
        brand_day_count = 0

        for brand_bucket in crisis_bucket['brands'].values():
            days_sorted = sorted(brand_bucket['days'])
            if not days_sorted:
                continue

            windows = [
                {
                    'start_date': window_start.isoformat(),
                    'end_date': window_end.isoformat(),
                    'duration_days': duration_days,
                }
                for window_start, window_end, duration_days in consecutive_day_windows(days_sorted)
            ]
            days_affected = len(days_sorted)
            active_on_window_end = end_date in brand_bucket['days']
            if active_on_window_end:
                active_brands_latest += 1
            brand_day_count += days_affected

            brand_rows.append({
                'brand_id': brand_bucket['brand_id'],
                'brand': brand_bucket['brand'],
                'sector': brand_bucket.get('sector') or '',
                'days_affected': days_affected,
                'first_seen_date': days_sorted[0].isoformat(),
                'last_seen_date': days_sorted[-1].isoformat(),
                'active_on_window_end': active_on_window_end,
                'episodes': len(windows),
                'total_negative_items': int(brand_bucket.get('total_negative_items') or 0),
                'windows': windows,
            })

        total_brand_days += brand_day_count
        brand_rows.sort(
            key=lambda item: (
                0 if item.get('active_on_window_end') else 1,
                -(item.get('days_affected') or 0),
                -(item.get('total_negative_items') or 0),
                str(item.get('brand') or '').casefold(),
            )
        )

        affected_brands = sorted(
            [item.get('brand') or '' for item in brand_rows if item.get('brand')],
            key=lambda value: value.casefold(),
        )
        crisis_days = sorted(crisis_bucket['days'])
        if not crisis_days:
            continue
        avg_active_days = round(brand_day_count / len(brand_rows), 1) if brand_rows else 0
        longest_active_days = max((int(item.get('days_affected') or 0) for item in brand_rows), default=0)

        daily_impact = []
        active_brands_series = []
        for day_iso in trend_dates:
            active_count = len(crisis_bucket['brands_by_day'].get(day_iso, set()))
            active_brands_series.append(active_count)
        for day_iso in sorted(crisis_bucket['brands_by_day'].keys(), reverse=True):
            daily_impact.append({
                'date': day_iso,
                'brands_affected': len(crisis_bucket['brands_by_day'][day_iso]),
                'brands': sorted(
                    crisis_bucket['brands_by_day'][day_iso],
                    key=lambda value: value.casefold(),
                ),
                'total_negative_items': int(crisis_bucket['negative_items_by_day'].get(day_iso, 0) or 0),
            })

        summary = {
            'tag': crisis_bucket['tag'],
            'display_tag': crisis_bucket['display_tag'],
            'group': crisis_bucket['group'],
            'is_crisis': True,
            'brands_affected': len(brand_rows),
            'active_brands_latest': active_brands_latest,
            'avg_active_days': avg_active_days,
            'longest_active_days': longest_active_days,
            'crisis_days': len(crisis_days),
            'brand_days': brand_day_count,
            'first_seen_date': crisis_days[0].isoformat(),
            'last_seen_date': crisis_days[-1].isoformat(),
            'total_negative_items': int(crisis_bucket['total_negative_items'] or 0),
            'affected_brands': affected_brands,
        }
        crisis_rows.append(summary)
        detail_lookup[crisis_key] = {
            **summary,
            'brands': brand_rows,
            'daily_impact': daily_impact,
        }
        trend_rows.append({
            'tag': crisis_bucket['tag'],
            'display_tag': crisis_bucket['display_tag'],
            'brands_affected': len(brand_rows),
            'active_brands_latest': active_brands_latest,
            'avg_active_days': avg_active_days,
            'longest_active_days': longest_active_days,
            'active_brands_series': active_brands_series,
        })

    crisis_rows.sort(
        key=lambda item: (
            -(item.get('brands_affected') or 0),
            -(item.get('active_brands_latest') or 0),
            -(item.get('brand_days') or 0),
            -(item.get('total_negative_items') or 0),
            str(item.get('tag') or '').casefold(),
        )
    )
    trend_rows.sort(
        key=lambda item: (
            -(item.get('brands_affected') or 0),
            -(item.get('active_brands_latest') or 0),
            -(item.get('avg_active_days') or 0),
            str(item.get('tag') or '').casefold(),
        )
    )
    return (
        crisis_rows,
        detail_lookup,
        len(affected_brand_ids),
        len(active_brand_ids),
        total_brand_days,
        trend_dates,
        trend_rows,
    )


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
@app.route('/crises.html')
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
                           to_char(a.published_at at time zone 'UTC', 'YYYY-MM-DD') as published_date,
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
                headers = ['company','title','url','source','published_date','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
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
                           to_char(a.published_at at time zone 'UTC', 'YYYY-MM-DD') as published_date,
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
                headers = ['ceo','company','title','url','source','published_date','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
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
                       to_jsonb(r) ->> 'published_date' as published_date,
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
            headers = ['company','title','url','position','snippet','published_date','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
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
                       to_jsonb(r) ->> 'published_date' as published_date,
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
            headers = ['ceo','company','title','url','position','snippet','published_date','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
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
    entity_type = canonical_entity_type(entity)
    entity_types = compatible_entity_types(entity)

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
    entity_type = canonical_entity_type(entity)
    entity_types = compatible_entity_types(entity)

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

    entity_type = canonical_entity_type(entity)
    if entity_type == 'company':
        entity_types = compatible_entity_types(entity)
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
                   to_jsonb(sfi) ->> 'published_date' as published_date,
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
                   to_jsonb(sfi) ->> 'published_date' as published_date,
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


@app.route('/api/v1/narrative_tags')
def narrative_tags_json():
    if PUBLIC_MODE:
        return jsonify([])
    date_str = (request.args.get('date') or '').strip()
    if not date_str:
        return jsonify({'error': 'date is required (YYYY-MM-DD)'}), 400
    entity = request.args.get('entity', 'brand')
    debug = (request.args.get('debug') or '').strip().lower() in {'1', 'true', 'yes'}
    cache_key = f"narrative_tags:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    entity_type = canonical_entity_type(entity)
    try:
        if entity_type == 'company':
            entity_types = compatible_entity_types(entity)
            params = [date_str, entity_types]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join companies c on c.id = ecd.entity_id" if scope_sql else ""
            sql = f"""
                select ecd.entity_name,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.is_crisis as narrative_is_crisis
                from entity_crisis_event_daily ecd
                {join_sql}
                where ecd.date = %s
                  and ecd.entity_type = any(%s)
                  and ecd.primary_tag is not null
                  {scope_sql}
            """
        else:
            params = [date_str, entity_type]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join ceos ceo on ceo.id = ecd.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
            sql = f"""
                select ecd.entity_name,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.is_crisis as narrative_is_crisis
                from entity_crisis_event_daily ecd
                {join_sql}
                where ecd.date = %s
                  and ecd.entity_type = %s
                  and ecd.primary_tag is not null
                  {scope_sql}
            """
        rows = query_dict(sql, tuple(params))
        if not rows:
            raise LookupError("no_crisis_event_rows")
    except Exception as exc:
        app.logger.warning("narrative_tags rollup fallback: %s", exc)
        try:
            if entity_type == 'company':
                entity_types = compatible_entity_types(entity)
                params = [date_str, entity_types]
                scope_sql, params = scope_clause("c.id", params)
                join_sql = "join companies c on c.id = sfi.entity_id" if scope_sql else ""
                sql = f"""
                    select sfi.entity_name,
                           sfi.narrative_primary_tag,
                           sfi.narrative_primary_group,
                           sfi.narrative_tags,
                           sfi.narrative_is_crisis
                    from serp_feature_items sfi
                    {join_sql}
                    left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                    where sfi.date = %s
                      and sfi.entity_type = any(%s)
                      and sfi.feature_type = 'top_stories_items'
                      and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                      and coalesce(sfi.finance_routine, false) = false
                      and sfi.narrative_primary_tag is not null
                      {scope_sql}
                """
            else:
                params = [date_str, entity_type]
                scope_sql, params = scope_clause("c.id", params)
                join_sql = "join ceos ceo on ceo.id = sfi.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
                sql = f"""
                    select sfi.entity_name,
                           sfi.narrative_primary_tag,
                           sfi.narrative_primary_group,
                           sfi.narrative_tags,
                           sfi.narrative_is_crisis
                    from serp_feature_items sfi
                    {join_sql}
                    left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                    where sfi.date = %s
                      and sfi.entity_type = %s
                      and sfi.feature_type = 'top_stories_items'
                      and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                      and coalesce(sfi.finance_routine, false) = false
                      and sfi.narrative_primary_tag is not null
                      {scope_sql}
                """
            rows = query_dict(sql, tuple(params))
        except Exception:
            app.logger.exception("narrative_tags failed")
            if debug:
                return jsonify({'error': 'narrative_tags_failed', 'detail': str(exc)}), 500
            return jsonify([])

    by_entity: Dict[str, Dict] = {}
    for r in rows:
        entity_name = (r.get('entity_name') or '').strip()
        if not entity_name:
            continue
        bucket = by_entity.setdefault(entity_name, {
            'entity_name': entity_name,
            'primary_counts': {},
            'tag_counts': {},
            'has_crisis': False,
            'has_non_crisis': False,
        })
        primary_tag = (r.get('narrative_primary_tag') or '').strip()
        primary_group = (r.get('narrative_primary_group') or '').strip().lower() or None
        if primary_tag:
            key = (primary_tag, primary_group)
            bucket['primary_counts'][key] = bucket['primary_counts'].get(key, 0) + 1
            if primary_group == 'non_crisis':
                bucket['has_non_crisis'] = True
            elif primary_group == 'crisis':
                bucket['has_crisis'] = True
        is_crisis = r.get('narrative_is_crisis')
        if is_crisis is True:
            bucket['has_crisis'] = True
        elif is_crisis is False:
            bucket['has_non_crisis'] = True
        raw_tags = r.get('narrative_tags') or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        raw_tag_counts = narrative_tag_count_map(r.get('tag_counts'))
        for tag in raw_tags:
            tag_txt = (str(tag) or '').strip()
            if not tag_txt:
                continue
            bucket['tag_counts'][tag_txt] = bucket['tag_counts'].get(tag_txt, 0) + max(
                raw_tag_counts.get(tag_txt, 1),
                1,
            )
            if tag_txt in NON_CRISIS_NARRATIVE_TAGS:
                bucket['has_non_crisis'] = True

    out = []
    for entity_name, bucket in by_entity.items():
        primary_tag = None
        primary_group = None
        if bucket['primary_counts']:
            (primary_tag, primary_group), _ = max(
                bucket['primary_counts'].items(),
                key=lambda kv: (kv[1], kv[0][0]),
            )
        sorted_tags = sorted(
            bucket['tag_counts'].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        tags = [t for t, _ in sorted_tags]
        display_tags = [narrative_display_tag(t) for t in tags]
        out.append({
            'entity_name': entity_name,
            'primary_tag': primary_tag,
            'primary_group': primary_group,
            'primary_display_tag': narrative_display_tag(primary_tag or '', primary_group),
            'tags': tags,
            'display_tags': display_tags,
            'has_crisis': bool(bucket['has_crisis']),
            'has_non_crisis': bool(bucket['has_non_crisis']),
        })

    out.sort(key=lambda r: r.get('entity_name') or '')
    set_cached_json(cache_key, out)
    return jsonify(out)


@app.route('/api/v1/narrative_timeline')
def narrative_timeline_json():
    if PUBLIC_MODE:
        return jsonify([])
    date_str = (request.args.get('date') or '').strip()
    entity = (request.args.get('entity') or 'brand').strip().lower()
    entity_name = (request.args.get('entity_name') or '').strip()
    if not date_str:
        return jsonify({'error': 'date is required (YYYY-MM-DD)'}), 400
    if not entity_name:
        return jsonify({'error': 'entity_name is required'}), 400
    if entity not in {'brand', 'ceo'}:
        return jsonify({'error': 'entity must be brand or ceo'}), 400

    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'invalid date format (YYYY-MM-DD)'}), 400

    try:
        days = int(request.args.get('days', '90') or 90)
    except ValueError:
        days = 90
    if days < 1:
        days = 1
    if days > 365:
        days = 365

    start_date = target_date - timedelta(days=days - 1)
    cache_key = f"narrative_timeline:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    entity_type = canonical_entity_type(entity)
    try:
        if entity_type == 'company':
            entity_types = compatible_entity_types(entity)
            params = [start_date, target_date, entity_types, entity_name]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join companies c on c.id = ecd.entity_id" if scope_sql else ""
            sql = f"""
                select ecd.date,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.is_crisis as narrative_is_crisis
                from entity_crisis_event_daily ecd
                {join_sql}
                where ecd.date between %s and %s
                  and ecd.entity_type = any(%s)
                  and ecd.entity_name = %s
                  and ecd.primary_tag is not null
                  {scope_sql}
            """
        else:
            params = [start_date, target_date, entity_type, entity_name]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join ceos ceo on ceo.id = ecd.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
            sql = f"""
                select ecd.date,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.is_crisis as narrative_is_crisis
                from entity_crisis_event_daily ecd
                {join_sql}
                where ecd.date between %s and %s
                  and ecd.entity_type = %s
                  and ecd.entity_name = %s
                  and ecd.primary_tag is not null
                  {scope_sql}
            """
        rows = query_dict(sql, tuple(params))
        if not rows:
            raise LookupError("no_crisis_event_rows")
    except Exception as exc:
        app.logger.warning("narrative_timeline rollup fallback: %s", exc)
        if entity_type == 'company':
            entity_types = compatible_entity_types(entity)
            params = [start_date, target_date, entity_types, entity_name]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join companies c on c.id = sfi.entity_id" if scope_sql else ""
            sql = f"""
                select sfi.date,
                       sfi.narrative_primary_tag,
                       sfi.narrative_primary_group,
                       sfi.narrative_tags,
                       sfi.narrative_is_crisis
                from serp_feature_items sfi
                {join_sql}
                left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                where sfi.date between %s and %s
                  and sfi.entity_type = any(%s)
                  and sfi.entity_name = %s
                  and sfi.feature_type = 'top_stories_items'
                  and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                  and coalesce(sfi.finance_routine, false) = false
                  and sfi.narrative_primary_tag is not null
                  {scope_sql}
            """
        else:
            params = [start_date, target_date, entity_type, entity_name]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join ceos ceo on ceo.id = sfi.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
            sql = f"""
                select sfi.date,
                       sfi.narrative_primary_tag,
                       sfi.narrative_primary_group,
                       sfi.narrative_tags,
                       sfi.narrative_is_crisis
                from serp_feature_items sfi
                {join_sql}
                left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                where sfi.date between %s and %s
                  and sfi.entity_type = %s
                  and sfi.entity_name = %s
                  and sfi.feature_type = 'top_stories_items'
                  and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                  and coalesce(sfi.finance_routine, false) = false
                  and sfi.narrative_primary_tag is not null
                  {scope_sql}
            """
        rows = query_dict(sql, tuple(params))

    target_iso = target_date.isoformat()
    by_tag: Dict[str, Dict] = {}
    for r in rows:
        row_date = r.get('date')
        if isinstance(row_date, datetime):
            day = row_date.date()
        elif hasattr(row_date, 'isoformat'):
            iso = str(row_date.isoformat()).strip()[:10]
            if not iso:
                continue
            try:
                day = datetime.strptime(iso, '%Y-%m-%d').date()
            except ValueError:
                continue
        else:
            raw = str(row_date or '').strip()
            if not raw:
                continue
            try:
                day = datetime.strptime(raw[:10], '%Y-%m-%d').date()
            except ValueError:
                continue
        day_iso = day.isoformat()

        primary_tag = (r.get('narrative_primary_tag') or '').strip()
        primary_group = (r.get('narrative_primary_group') or '').strip().lower()
        is_crisis = r.get('narrative_is_crisis')
        raw_tags = r.get('narrative_tags') or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        raw_tag_counts = narrative_tag_count_map(r.get('tag_counts'))
        row_tags = []
        seen = set()
        for tag in raw_tags:
            txt = (str(tag) or '').strip()
            if not txt:
                continue
            norm = txt.casefold()
            if norm in seen:
                continue
            row_tags.append(txt)
            seen.add(norm)
        if primary_tag:
            norm = primary_tag.casefold()
            if norm not in seen:
                row_tags.insert(0, primary_tag)
        if not row_tags:
            continue

        primary_norm = primary_tag.casefold() if primary_tag else ''
        for tag in row_tags:
            norm = tag.casefold()
            bucket = by_tag.setdefault(norm, {
                'tag': tag,
                'days': set(),
                'mentions_total': 0,
                'mentions_by_date': {},
                'group_counts': {'crisis': 0, 'non_crisis': 0},
            })
            weight = max(raw_tag_counts.get(tag, 1), 1)
            bucket['mentions_total'] += weight
            bucket['days'].add(day)
            bucket['mentions_by_date'][day_iso] = bucket['mentions_by_date'].get(day_iso, 0) + weight

            vote = None
            if primary_norm and norm == primary_norm and primary_group in {'crisis', 'non_crisis'}:
                vote = primary_group
            elif tag in NON_CRISIS_NARRATIVE_TAGS:
                vote = 'non_crisis'
            elif isinstance(is_crisis, bool):
                vote = 'crisis' if is_crisis else 'non_crisis'
            if vote:
                bucket['group_counts'][vote] = bucket['group_counts'].get(vote, 0) + 1

    tags_out = []
    for bucket in by_tag.values():
        days_set = bucket['days']
        if not days_set:
            continue
        days_sorted = sorted(days_set)
        first_seen = days_sorted[0].isoformat()
        last_seen = days_sorted[-1].isoformat()
        active_on_date = target_date in days_set

        duration_days = 0
        start_iso = None
        end_iso = None
        if active_on_date:
            cursor = target_date
            while cursor in days_set:
                duration_days += 1
                cursor = cursor - timedelta(days=1)
            start_iso = (target_date - timedelta(days=duration_days - 1)).isoformat()
            end_iso = target_iso

        group_counts = bucket.get('group_counts') or {}
        crisis_votes = int(group_counts.get('crisis') or 0)
        non_crisis_votes = int(group_counts.get('non_crisis') or 0)
        group = None
        if crisis_votes > non_crisis_votes:
            group = 'crisis'
        elif non_crisis_votes > 0:
            group = 'non_crisis'

        tag = bucket.get('tag') or ''
        tags_out.append({
            'tag': tag,
            'display_tag': narrative_display_tag(tag, group),
            'group': group,
            'is_crisis': group == 'crisis',
            'is_non_crisis': group == 'non_crisis',
            'active_on_date': active_on_date,
            'mentions_on_date': int((bucket.get('mentions_by_date') or {}).get(target_iso, 0) or 0),
            'mentions_total': int(bucket.get('mentions_total') or 0),
            'days_present': len(days_set),
            'first_seen_date': first_seen,
            'last_seen_date': last_seen,
            'current_duration_days': duration_days,
            'current_start_date': start_iso,
            'current_end_date': end_iso,
        })

    tags_out.sort(
        key=lambda r: (
            0 if r.get('active_on_date') else 1,
            -(r.get('current_duration_days') or 0),
            -(r.get('mentions_on_date') or 0),
            -(r.get('mentions_total') or 0),
            str(r.get('tag') or '').casefold(),
        )
    )
    payload = {
        'entity': entity,
        'entity_name': entity_name,
        'date': target_iso,
        'lookback_days': days,
        'tags': tags_out,
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/narrative_overlay')
def narrative_overlay_json():
    if PUBLIC_MODE:
        return jsonify([])
    entity = (request.args.get('entity') or 'brand').strip().lower()
    entity_name = (request.args.get('entity_name') or '').strip()
    start_str = (request.args.get('start_date') or '').strip()
    end_str = (request.args.get('end_date') or '').strip()
    include_non_crisis = (request.args.get('include_non_crisis') or '').strip().lower() in {'1', 'true', 'yes'}
    try:
        limit = int(request.args.get('limit', '0') or 0)
    except ValueError:
        limit = 0
    limit = min(max(limit, 0), 200)

    if entity not in {'brand', 'ceo'}:
        return jsonify({'error': 'entity must be brand or ceo'}), 400
    if not entity_name:
        return jsonify({'error': 'entity_name is required'}), 400
    if not start_str or not end_str:
        return jsonify({'error': 'start_date and end_date are required (YYYY-MM-DD)'}), 400

    try:
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'invalid date format (YYYY-MM-DD)'}), 400

    if start_date > end_date:
        return jsonify({'error': 'start_date must be on or before end_date'}), 400

    cache_key = f"narrative_overlay:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    entity_type = canonical_entity_type(entity)
    try:
        if entity_type == 'company':
            entity_types = compatible_entity_types(entity)
            params = [start_date, end_date, entity_types, entity_name]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join companies c on c.id = ecd.entity_id" if scope_sql else ""
            sql = f"""
                select ecd.date,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.is_crisis as narrative_is_crisis,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.supporting_negative_items as negative_item_count
                from entity_crisis_event_daily ecd
                {join_sql}
                where ecd.date between %s and %s
                  and ecd.entity_type = any(%s)
                  and ecd.entity_name = %s
                  and ecd.primary_tag is not null
                  {scope_sql}
                order by ecd.date, ecd.primary_tag
            """
        else:
            params = [start_date, end_date, entity_type, entity_name]
            scope_sql, params = scope_clause("c.id", params)
            join_sql = "join ceos ceo on ceo.id = ecd.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
            sql = f"""
                select ecd.date,
                       ecd.primary_tag as narrative_primary_tag,
                       ecd.primary_group as narrative_primary_group,
                       ecd.is_crisis as narrative_is_crisis,
                       ecd.tags as narrative_tags,
                       ecd.tag_counts,
                       ecd.supporting_negative_items as negative_item_count
                from entity_crisis_event_daily ecd
                {join_sql}
                where ecd.date between %s and %s
                  and ecd.entity_type = %s
                  and ecd.entity_name = %s
                  and ecd.primary_tag is not null
                  {scope_sql}
                order by ecd.date, ecd.primary_tag
            """
        rows = query_dict(sql, tuple(params))
        if not rows:
            raise LookupError("no_crisis_event_rows")
        rows = expand_rollup_narrative_rows(rows)
    except Exception as exc:
        app.logger.warning("narrative_overlay rollup fallback: %s", exc)
        try:
            if entity_type == 'company':
                entity_types = compatible_entity_types(entity)
                params = [start_date, end_date, entity_types, entity_name]
                scope_sql, params = scope_clause("c.id", params)
                join_sql = "join companies c on c.id = sfi.entity_id" if scope_sql else ""
                sql = f"""
                    select sfi.date,
                           sfi.narrative_primary_tag,
                           sfi.narrative_primary_group,
                           sfi.narrative_is_crisis,
                           count(*) as negative_item_count
                    from serp_feature_items sfi
                    {join_sql}
                    left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                    where sfi.date between %s and %s
                      and sfi.entity_type = any(%s)
                      and sfi.entity_name = %s
                      and sfi.feature_type = 'top_stories_items'
                      and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                      and coalesce(sfi.finance_routine, false) = false
                      and sfi.narrative_primary_tag is not null
                      {scope_sql}
                    group by sfi.date, sfi.narrative_primary_tag, sfi.narrative_primary_group, sfi.narrative_is_crisis
                    order by sfi.date, sfi.narrative_primary_tag
                """
            else:
                params = [start_date, end_date, entity_type, entity_name]
                scope_sql, params = scope_clause("c.id", params)
                join_sql = "join ceos ceo on ceo.id = sfi.entity_id join companies c on c.id = ceo.company_id" if scope_sql else ""
                sql = f"""
                    select sfi.date,
                           sfi.narrative_primary_tag,
                           sfi.narrative_primary_group,
                           sfi.narrative_is_crisis,
                           count(*) as negative_item_count
                    from serp_feature_items sfi
                    {join_sql}
                    left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                    where sfi.date between %s and %s
                      and sfi.entity_type = %s
                      and sfi.entity_name = %s
                      and sfi.feature_type = 'top_stories_items'
                      and coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                      and coalesce(sfi.finance_routine, false) = false
                      and sfi.narrative_primary_tag is not null
                      {scope_sql}
                    group by sfi.date, sfi.narrative_primary_tag, sfi.narrative_primary_group, sfi.narrative_is_crisis
                    order by sfi.date, sfi.narrative_primary_tag
                """
            rows = query_dict(sql, tuple(params))
        except Exception:
            app.logger.exception("narrative_overlay failed")
            return jsonify({
                'entity': entity,
                'entity_name': entity_name,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'include_non_crisis': include_non_crisis,
                'windows': [],
            })

    by_tag: Dict[str, Dict] = {}
    for row in rows:
        day = coerce_date(row.get('date'))
        if day is None:
            continue
        tag = (row.get('narrative_primary_tag') or '').strip()
        if not tag:
            continue
        primary_group = (row.get('narrative_primary_group') or '').strip().lower()
        if primary_group not in {'crisis', 'non_crisis'}:
            if tag in NON_CRISIS_NARRATIVE_TAGS:
                primary_group = 'non_crisis'
            elif row.get('narrative_is_crisis') is True:
                primary_group = 'crisis'
            elif row.get('narrative_is_crisis') is False:
                primary_group = 'non_crisis'
        if not include_non_crisis and primary_group == 'non_crisis':
            continue
        tag_key = f"{tag.casefold()}::{primary_group or ''}"
        bucket = by_tag.setdefault(tag_key, {
            'tag': tag,
            'group': primary_group or None,
            'display_tag': narrative_display_tag(tag, primary_group or None),
            'days': {},
        })
        day_iso = day.isoformat()
        bucket['days'][day_iso] = bucket['days'].get(day_iso, 0) + int(row.get('negative_item_count') or 0)

    windows_out: List[Dict] = []
    for bucket in by_tag.values():
        day_counts = bucket.get('days') or {}
        day_values = sorted(
            ((coerce_date(day_iso), int(count or 0)) for day_iso, count in day_counts.items()),
            key=lambda item: item[0],
        )
        if not day_values:
            continue
        count_by_day = {day: count for day, count in day_values if day is not None}
        for window_start, window_end, duration_days in consecutive_day_windows(list(count_by_day.keys())):
            negative_item_count = sum(
                count_by_day.get(day, 0)
                for day in count_by_day.keys()
                if window_start <= day <= window_end
            )
            windows_out.append({
                'tag': bucket['tag'],
                'display_tag': bucket['display_tag'],
                'group': bucket['group'],
                'is_crisis': bucket['group'] == 'crisis',
                'is_non_crisis': bucket['group'] == 'non_crisis',
                'start_date': window_start.isoformat(),
                'end_date': window_end.isoformat(),
                'duration_days': duration_days,
                'negative_item_count': negative_item_count,
                'active_on_end_date': window_start <= end_date <= window_end,
            })

    windows_out.sort(key=lambda row: str(row.get('display_tag') or '').casefold())
    windows_out.sort(key=lambda row: row.get('negative_item_count') or 0, reverse=True)
    windows_out.sort(key=lambda row: str(row.get('end_date') or ''), reverse=True)
    windows_out.sort(key=lambda row: row.get('duration_days') or 0, reverse=True)
    windows_out.sort(key=lambda row: 0 if row.get('active_on_end_date') else 1)
    selected_windows = windows_out if limit == 0 else windows_out[:limit]
    selected_windows.sort(
        key=lambda row: (
            str(row.get('start_date') or ''),
            str(row.get('end_date') or ''),
            str(row.get('display_tag') or '').casefold(),
        )
    )
    payload = {
        'entity': entity,
        'entity_name': entity_name,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'include_non_crisis': include_non_crisis,
        'windows': selected_windows,
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


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

    entity_type = canonical_entity_type(entity)
    if entity_type == 'company':
        entity_types = compatible_entity_types(entity)
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
            rows = negative_summary_live(days=days, company=None, mode='index')
        else:
            rows = negative_summary_live(days=days, company=company_filter or None, mode='detail')
        data = serialize_rows(rows)
        set_cached_json(cache_key, data)
        return jsonify(data)
    except Exception as exc:
        app.logger.exception("negative_summary failed")
        if debug:
            return jsonify({'error': 'negative_summary_failed', 'detail': str(exc)}), 500
        raise


@app.route('/api/v1/insights/resolve_entity')
def insights_resolve_entity_json():
    entity = request.args.get('entity', 'brand')
    entity_name = (request.args.get('entity_name') or '').strip()
    if not entity_name:
        return jsonify({'error': 'entity_name is required'}), 400
    try:
        limit = int(request.args.get('limit', '5') or 5)
    except ValueError:
        limit = 5
    limit = min(max(limit, 1), 10)

    cache_key = f"insights_resolve_entity:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    resolved = resolve_entity_lookup(entity, entity_name)
    suggestion_rows: List[Dict] = []
    if resolved:
        suggestion_rows.append(resolved)
    suggestion_rows.extend(entity_lookup_suggestions(entity, entity_name, limit=limit))
    suggestions = dedupe_entity_payloads(suggestion_rows)[:limit]

    payload = {
        'query': {
            'entity': analytics_entity_type(entity),
            'entity_name': entity_name,
        },
        'resolution_status': (
            'suggestions_only'
            if resolved is None and suggestions
            else lookup_resolution_status(resolved)
        ),
        'resolved': entity_payload(resolved) if resolved else None,
        'suggestions': suggestions,
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/resolve_sector')
def insights_resolve_sector_json():
    sector_name = (request.args.get('sector_name') or '').strip()
    if not sector_name:
        return jsonify({'error': 'sector_name is required'}), 400
    try:
        limit = int(request.args.get('limit', '5') or 5)
    except ValueError:
        limit = 5
    limit = min(max(limit, 1), 10)

    cache_key = f"insights_resolve_sector:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    resolved = resolve_sector_lookup(sector_name)
    suggestion_rows: List[Dict] = []
    if resolved:
        suggestion_rows.append(resolved)
    suggestion_rows.extend(sector_lookup_suggestions(sector_name, limit=limit))
    suggestions = dedupe_sector_payloads(suggestion_rows)[:limit]

    payload = {
        'query': {
            'sector_name': sector_name,
        },
        'resolution_status': (
            'suggestions_only'
            if resolved is None and suggestions
            else sector_lookup_resolution_status(resolved)
        ),
        'resolved': sector_payload(resolved) if resolved else None,
        'suggestions': suggestions,
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/screen_entities')
def insights_screen_entities_json():
    entity = request.args.get('entity', 'brand')
    metric = (request.args.get('metric') or 'top_stories_negative_count').strip()
    sector = (request.args.get('sector') or '').strip()
    if metric not in SCREENABLE_METRICS:
        return jsonify({
            'error': 'unsupported metric',
            'supported_metrics': sorted(SCREENABLE_METRICS.keys()),
        }), 400

    try:
        days = int(request.args.get('days', '1') or 1)
    except ValueError:
        days = 1
    try:
        limit = int(request.args.get('limit', '20') or 20)
    except ValueError:
        limit = 20
    try:
        min_value = float(request.args.get('min_value', '1') or 1)
    except ValueError:
        min_value = 1.0

    days = min(max(days, 1), 90)
    limit = min(max(limit, 1), 100)
    min_value = max(min_value, 0.0)

    cache_key = f"insights_screen_entities:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    analytics_type = analytics_entity_type(entity)
    latest_params: List = [analytics_type]
    sector_sql = ""
    if sector:
        latest_params.append(f"%{sector}%")
        sector_sql = " and coalesce(c.sector, '') ilike %s"
    latest_scope_sql, latest_params = scope_clause("edm.company_id", latest_params)
    latest_rows = query_rows(
        f"""
        select max(edm.date)
        from entity_daily_metrics_v edm
        join companies c on c.id = edm.company_id
        where edm.entity_type = %s
          {sector_sql}
          {latest_scope_sql}
        """,
        tuple(latest_params),
    )
    end_date = latest_rows[0][0] if latest_rows and latest_rows[0] else None
    if end_date is None:
        return jsonify({'error': 'no_data'}), 404

    start_date = end_date - timedelta(days=days - 1)
    metric_sql = SCREENABLE_METRICS[metric]['column']
    params = [end_date, analytics_type, start_date, end_date]
    sector_sql = ""
    if sector:
        params.append(f"%{sector}%")
        sector_sql = " and coalesce(c.sector, '') ilike %s"
    scope_sql, params = scope_clause("edm.company_id", params)
    params.extend([min_value, limit])
    rows = query_dict(
        f"""
        select edm.entity_type,
               edm.entity_id,
               edm.company_id,
               edm.ceo_id,
               max(edm.entity_name) as entity_name,
               max(edm.company) as company,
               max(edm.ceo) as ceo,
               max(coalesce(c.sector, '')) as sector,
               sum(edm.{metric_sql})::numeric as window_value,
               max(case when edm.date = %s then edm.{metric_sql} end)::numeric as latest_value,
               max(edm.{metric_sql})::numeric as peak_value,
               count(*) filter (where edm.{metric_sql} > 0) as signal_days
        from entity_daily_metrics_v edm
        join companies c on c.id = edm.company_id
        where edm.entity_type = %s
          and edm.date between %s and %s
          {sector_sql}
          {scope_sql}
        group by edm.entity_type, edm.entity_id, edm.company_id, edm.ceo_id
        having sum(edm.{metric_sql}) >= %s
        order by window_value desc, latest_value desc, entity_name
        limit %s
        """,
        tuple(params),
    )

    payload_rows = []
    for row in rows:
        payload_rows.append(
            {
                **entity_payload({
                    'analytics_entity_type': row.get('entity_type'),
                    'entity_id': row.get('entity_id'),
                    'company_id': row.get('company_id'),
                    'ceo_id': row.get('ceo_id'),
                    'entity_name': row.get('entity_name'),
                    'company': row.get('company'),
                    'ceo': row.get('ceo'),
                }),
                'window_value': row.get('window_value'),
                'latest_value': row.get('latest_value'),
                'peak_value': row.get('peak_value'),
                'signal_days': row.get('signal_days'),
                'sector': row.get('sector') or '',
            }
        )

    payload = {
        'entity': analytics_type,
        'metric': metric,
        'metric_label': SCREENABLE_METRICS[metric]['label'],
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'latest_available_date': end_date.isoformat(),
        'days': days,
        'limit': limit,
        'min_value': min_value,
        'sector': sector,
        'rows': serialize_rows(payload_rows),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/sector_baseline')
def insights_sector_baseline_json():
    entity = request.args.get('entity', 'brand')
    sector_name = (request.args.get('sector') or '').strip()
    metric = (request.args.get('metric') or 'top_stories_negative_count').strip()
    entity_name = (request.args.get('entity_name') or '').strip()

    if not sector_name:
        return jsonify({'error': 'sector is required'}), 400
    if metric not in SCREENABLE_METRICS:
        return jsonify({
            'error': 'unsupported metric',
            'supported_metrics': sorted(SCREENABLE_METRICS.keys()),
        }), 400

    try:
        days = int(request.args.get('days', '30') or 30)
    except ValueError:
        days = 30
    try:
        limit = int(request.args.get('limit', '10') or 10)
    except ValueError:
        limit = 10

    days = min(max(days, 1), 180)
    limit = min(max(limit, 1), 100)

    cache_key = f"insights_sector_baseline:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    resolved_sector = resolve_sector_lookup(sector_name)
    sector_suggestion_rows: List[Dict] = []
    if resolved_sector:
        sector_suggestion_rows.append(resolved_sector)
    sector_suggestion_rows.extend(sector_lookup_suggestions(sector_name, limit=5))
    sector_suggestions = dedupe_sector_payloads(sector_suggestion_rows)[:5]
    if resolved_sector is None:
        payload = {
            'query': {
                'entity': analytics_entity_type(entity),
                'sector': sector_name,
                'metric': metric,
                'entity_name': entity_name,
            },
            'resolution_status': (
                'suggestions_only' if sector_suggestions else 'not_found'
            ),
            'resolved_sector': None,
            'sector_suggestions': sector_suggestions,
            'rows': [],
        }
        return jsonify(payload), 404

    analytics_type = analytics_entity_type(entity)
    latest_params: List = [analytics_type, resolved_sector['sector']]
    latest_scope_sql, latest_params = scope_clause("edm.company_id", latest_params)
    latest_rows = query_rows(
        f"""
        select max(edm.date)
        from entity_daily_metrics_v edm
        join companies c on c.id = edm.company_id
        where edm.entity_type = %s
          and c.sector = %s
          {latest_scope_sql}
        """,
        tuple(latest_params),
    )
    end_date = latest_rows[0][0] if latest_rows and latest_rows[0] else None
    if end_date is None:
        return jsonify({'error': 'no_data'}), 404

    start_date = end_date - timedelta(days=days - 1)
    metric_sql = SCREENABLE_METRICS[metric]['column']
    params = [end_date, analytics_type, resolved_sector['sector'], start_date, end_date]
    scope_sql, params = scope_clause("edm.company_id", params)
    rows = query_dict(
        f"""
        select edm.entity_type,
               edm.entity_id,
               edm.company_id,
               edm.ceo_id,
               max(edm.entity_name) as entity_name,
               max(edm.company) as company,
               max(edm.ceo) as ceo,
               max(coalesce(c.sector, '')) as sector,
               sum(edm.{metric_sql})::numeric as window_value,
               avg(edm.{metric_sql})::numeric as avg_daily_value,
               max(case when edm.date = %s then edm.{metric_sql} end)::numeric as latest_value,
               max(edm.{metric_sql})::numeric as peak_value,
               count(*) filter (where edm.{metric_sql} > 0) as signal_days
        from entity_daily_metrics_v edm
        join companies c on c.id = edm.company_id
        where edm.entity_type = %s
          and c.sector = %s
          and edm.date between %s and %s
          {scope_sql}
        group by edm.entity_type, edm.entity_id, edm.company_id, edm.ceo_id
        order by window_value desc, latest_value desc, entity_name
        """,
        tuple(params),
    )

    payload_rows = []
    for row in rows:
        payload_rows.append(
            {
                **entity_payload({
                    'analytics_entity_type': row.get('entity_type'),
                    'entity_id': row.get('entity_id'),
                    'company_id': row.get('company_id'),
                    'ceo_id': row.get('ceo_id'),
                    'entity_name': row.get('entity_name'),
                    'company': row.get('company'),
                    'ceo': row.get('ceo'),
                }),
                'window_value': row.get('window_value'),
                'avg_daily_value': row.get('avg_daily_value'),
                'latest_value': row.get('latest_value'),
                'peak_value': row.get('peak_value'),
                'signal_days': row.get('signal_days'),
                'sector': row.get('sector') or '',
            }
        )

    window_values = [float(row.get('window_value') or 0) for row in payload_rows]
    avg_window_value = round(sum(window_values) / len(window_values), 4) if window_values else 0.0
    median_window_value = round(float(median(window_values)), 4) if window_values else 0.0
    active_entity_count = sum(1 for row in payload_rows if float(row.get('window_value') or 0) > 0)

    peer_payload = None
    peer_resolution = None
    peer_suggestions: List[Dict] = []
    if entity_name:
        resolved_entity = resolve_entity_lookup(entity, entity_name)
        peer_resolution = entity_payload(resolved_entity) if resolved_entity else None
        if resolved_entity is None:
            peer_suggestions = dedupe_entity_payloads(
                entity_lookup_suggestions(entity, entity_name, limit=5)
            )[:5]
        else:
            row_index = next(
                (
                    idx for idx, row in enumerate(payload_rows)
                    if row.get('entity_type') == peer_resolution.get('entity_type')
                    and row.get('entity_id') == peer_resolution.get('entity_id')
                ),
                None,
            )
            if row_index is not None:
                peer_row = dict(payload_rows[row_index])
                rank = row_index + 1
                peer_count = len(payload_rows)
                percentile = 100.0
                if peer_count > 1:
                    percentile = round(100.0 * ((peer_count - rank) / (peer_count - 1)), 1)
                peer_payload = {
                    **peer_row,
                    'rank': rank,
                    'peer_count': peer_count,
                    'percentile': percentile,
                    'vs_sector_avg': round(float(peer_row.get('window_value') or 0) - avg_window_value, 4),
                    'vs_sector_median': round(float(peer_row.get('window_value') or 0) - median_window_value, 4),
                    'in_sector': True,
                }
            else:
                peer_payload = {
                    **peer_resolution,
                    'in_sector': False,
                    'reason': 'entity_not_in_sector_or_no_metric_data_for_window',
                }

    payload = {
        'entity': analytics_type,
        'metric': metric,
        'metric_label': SCREENABLE_METRICS[metric]['label'],
        'days': days,
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'latest_available_date': end_date.isoformat(),
        'resolution_status': sector_lookup_resolution_status(resolved_sector),
        'resolved_sector': sector_payload(resolved_sector),
        'sector_suggestions': sector_suggestions,
        'sector_summary': {
            'entity_count': len(payload_rows),
            'active_entity_count': active_entity_count,
            'avg_window_value': avg_window_value,
            'median_window_value': median_window_value,
            'max_window_value': max(window_values) if window_values else 0,
        },
        'peer_entity_resolution': peer_resolution,
        'peer_entity': serialize_rows([peer_payload])[0] if peer_payload else None,
        'peer_entity_suggestions': peer_suggestions,
        'rows': serialize_rows(payload_rows[:limit]),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/aggregate_crisis_patterns')
def insights_aggregate_crisis_patterns_json():
    entity = request.args.get('entity', 'brand')
    sector = (request.args.get('sector') or '').strip()
    try:
        limit = int(request.args.get('limit', '10') or 10)
    except ValueError:
        limit = 10
    include_non_crisis = (request.args.get('include_non_crisis') or '').strip().lower() in {'1', 'true', 'yes'}
    limit = min(max(limit, 1), 50)

    cache_key = f"insights_aggregate_crisis_patterns:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        start_date, end_date, latest_available_date, days, window_mode, requested_start_date, requested_end_date = resolve_insights_window(
            entity,
            sector=sector,
            default_days=90,
            min_days=7,
            max_days=365,
        )
    except LookupError:
        return jsonify({'error': 'no_data'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    analytics_type, rows = fetch_negative_top_stories_narrative_rows(entity, start_date, end_date, sector)

    by_pattern: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        tag = (row.get('narrative_primary_tag') or '').strip()
        if not tag:
            continue
        primary_group = (row.get('narrative_primary_group') or '').strip().lower()
        if primary_group not in {'crisis', 'non_crisis'}:
            if tag in NON_CRISIS_NARRATIVE_TAGS:
                primary_group = 'non_crisis'
            elif row.get('narrative_is_crisis') is True:
                primary_group = 'crisis'
            elif row.get('narrative_is_crisis') is False:
                primary_group = 'non_crisis'
        if not include_non_crisis and primary_group == 'non_crisis':
            continue

        day = coerce_date(row.get('date'))
        entity_id = str(row.get('entity_id') or '')
        company = (row.get('company') or '').strip()
        entity_name = (row.get('entity_name') or '').strip()
        key = (tag.casefold(), primary_group or '')
        bucket = by_pattern.setdefault(key, {
            'tag': tag,
            'group': primary_group or None,
            'display_tag': narrative_display_tag(tag, primary_group or None),
            'entity_days': {},
            'entity_names': {},
            'entity_totals': {},
            'total_negative_items': 0,
            'active_entity_ids': set(),
            'sector_values': set(),
        })
        bucket['entity_days'].setdefault(entity_id, set()).add(day)
        bucket['entity_names'][entity_id] = entity_name
        bucket['entity_totals'][entity_id] = bucket['entity_totals'].get(entity_id, 0) + int(row.get('negative_item_count') or 0)
        bucket['total_negative_items'] += int(row.get('negative_item_count') or 0)
        bucket['sector_values'].add((row.get('sector') or '').strip())
        if day == end_date:
            bucket['active_entity_ids'].add(entity_id)

    pattern_rows = []
    for bucket in by_pattern.values():
        durations: List[int] = []
        for entity_id, days_set in bucket['entity_days'].items():
            durations.extend(consecutive_day_durations(list(days_set)))
        if not durations:
            continue

        top_examples = sorted(
            bucket['entity_totals'].items(),
            key=lambda item: (-item[1], bucket['entity_names'].get(item[0], '').casefold()),
        )[:3]
        pattern_rows.append({
            'tag': bucket['tag'],
            'display_tag': bucket['display_tag'],
            'group': bucket['group'],
            'is_crisis': bucket['group'] == 'crisis',
            'entity_type': analytics_type,
            'brands_affected' if analytics_type == 'brand' else 'ceos_affected': len(bucket['entity_days']),
            'episode_count': len(durations),
            'avg_duration_days': round(sum(durations) / len(durations), 2),
            'median_duration_days': float(median(durations)),
            'max_duration_days': max(durations),
            'active_entities_latest': len(bucket['active_entity_ids']),
            'total_negative_items': bucket['total_negative_items'],
            'sample_entities': [bucket['entity_names'].get(entity_id, entity_id) for entity_id, _ in top_examples],
        })

    affected_key = 'brands_affected' if analytics_type == 'brand' else 'ceos_affected'
    pattern_rows.sort(
        key=lambda row: (
            -(row.get(affected_key) or 0),
            -(row.get('episode_count') or 0),
            -(row.get('total_negative_items') or 0),
            str(row.get('tag') or '').casefold(),
        )
    )

    payload = {
        'entity': analytics_type,
        'sector': sector,
        'days': days,
        'limit': limit,
        'window_mode': window_mode,
        'requested_start_date': requested_start_date,
        'requested_end_date': requested_end_date,
        'include_non_crisis': include_non_crisis,
        'latest_available_date': latest_available_date.isoformat(),
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'duration_definition': (
            'Duration is measured as consecutive days with at least one tagged negative crisis event for the same entity, '
            'using Top Stories and recent negative newsfeed coverage.'
        ),
        'rows': serialize_rows(pattern_rows[:limit]),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/crisis_brand_impact')
def insights_crisis_brand_impact_json():
    sector = (request.args.get('sector') or '').strip()
    requested_crisis_tag = (request.args.get('crisis_tag') or '').strip()

    cache_key = f"insights_crisis_brand_impact:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        start_date, end_date, latest_available_date, days, window_mode, requested_start_date, requested_end_date = resolve_insights_window(
            'brand',
            sector=sector,
            default_days=30,
            min_days=7,
            max_days=365,
        )
    except LookupError:
        return jsonify({'error': 'no_data'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    analytics_type, rows = fetch_negative_top_stories_narrative_rows('brand', start_date, end_date, sector)
    crisis_rows, detail_lookup, affected_brand_count, active_brand_count, brand_day_count, trend_dates, trend_rows = build_crisis_brand_impact_summary(
        rows,
        start_date,
        end_date,
    )

    selected_crisis = None
    selected_lookup_key = requested_crisis_tag.casefold()
    if selected_lookup_key:
        selected_crisis = detail_lookup.get(selected_lookup_key)
    if selected_crisis is None and crisis_rows:
        selected_crisis = detail_lookup.get((crisis_rows[0].get('tag') or '').casefold())

    payload = {
        'entity': analytics_type,
        'sector': sector,
        'days': days,
        'window_mode': window_mode,
        'requested_start_date': requested_start_date,
        'requested_end_date': requested_end_date,
        'requested_crisis_tag': requested_crisis_tag or None,
        'latest_available_date': latest_available_date.isoformat(),
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'crisis_count': len(crisis_rows),
        'affected_brand_count': affected_brand_count,
        'active_brand_count': active_brand_count,
        'brand_day_count': brand_day_count,
        'crises': crisis_rows,
        'trend_dates': trend_dates,
        'trend_rows': trend_rows,
        'selected_crisis': selected_crisis,
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/aggregate_industry_durations')
def insights_aggregate_industry_durations_json():
    entity = request.args.get('entity', 'brand')
    try:
        limit = int(request.args.get('limit', '25') or 25)
    except ValueError:
        limit = 25
    include_non_crisis = (request.args.get('include_non_crisis') or '').strip().lower() in {'1', 'true', 'yes'}
    limit = min(max(limit, 1), 100)

    cache_key = f"insights_aggregate_industry_durations:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        start_date, end_date, latest_available_date, days, window_mode, requested_start_date, requested_end_date = resolve_insights_window(
            entity,
            default_days=90,
            min_days=7,
            max_days=365,
        )
    except LookupError:
        return jsonify({'error': 'no_data'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    analytics_type, rows = fetch_negative_top_stories_narrative_rows(entity, start_date, end_date)

    by_sector: Dict[str, Dict] = {}
    for row in rows:
        tag = (row.get('narrative_primary_tag') or '').strip()
        if not tag:
            continue
        primary_group = (row.get('narrative_primary_group') or '').strip().lower()
        if primary_group not in {'crisis', 'non_crisis'}:
            if tag in NON_CRISIS_NARRATIVE_TAGS:
                primary_group = 'non_crisis'
            elif row.get('narrative_is_crisis') is True:
                primary_group = 'crisis'
            elif row.get('narrative_is_crisis') is False:
                primary_group = 'non_crisis'
        if not include_non_crisis and primary_group == 'non_crisis':
            continue

        day = coerce_date(row.get('date'))
        entity_id = str(row.get('entity_id') or '')
        sector = (row.get('sector') or '').strip() or 'Unspecified'
        tag_key = f"{tag.casefold()}::{primary_group or ''}"
        bucket = by_sector.setdefault(sector, {
            'sector': sector,
            'entity_ids': set(),
            'active_entity_ids': set(),
            'tag_entity_days': {},
            'tag_display': {},
            'tag_totals': {},
            'total_negative_items': 0,
        })
        bucket['entity_ids'].add(entity_id)
        if day == end_date:
            bucket['active_entity_ids'].add(entity_id)
        bucket['tag_entity_days'].setdefault((entity_id, tag_key), set()).add(day)
        bucket['tag_display'][tag_key] = narrative_display_tag(tag, primary_group or None)
        bucket['tag_totals'][tag_key] = bucket['tag_totals'].get(tag_key, 0) + int(row.get('negative_item_count') or 0)
        bucket['total_negative_items'] += int(row.get('negative_item_count') or 0)

    sector_rows = []
    affected_key = 'brands_affected' if analytics_type == 'brand' else 'ceos_affected'
    for bucket in by_sector.values():
        durations: List[int] = []
        for days_set in bucket['tag_entity_days'].values():
            durations.extend(consecutive_day_durations(list(days_set)))
        if not durations:
            continue

        top_tags = sorted(
            bucket['tag_totals'].items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        sector_rows.append({
            'sector': bucket['sector'],
            'entity_type': analytics_type,
            affected_key: len(bucket['entity_ids']),
            'episode_count': len(durations),
            'avg_duration_days': round(sum(durations) / len(durations), 2),
            'median_duration_days': float(median(durations)),
            'max_duration_days': max(durations),
            'active_entities_latest': len(bucket['active_entity_ids']),
            'total_negative_items': bucket['total_negative_items'],
            'most_common_tags': [bucket['tag_display'].get(tag_key, tag_key.split('::', 1)[0]) for tag_key, _ in top_tags],
        })

    sector_rows.sort(
        key=lambda row: (
            -(row.get('avg_duration_days') or 0),
            -(row.get('episode_count') or 0),
            -(row.get(affected_key) or 0),
            str(row.get('sector') or '').casefold(),
        )
    )

    payload = {
        'entity': analytics_type,
        'days': days,
        'limit': limit,
        'window_mode': window_mode,
        'requested_start_date': requested_start_date,
        'requested_end_date': requested_end_date,
        'include_non_crisis': include_non_crisis,
        'latest_available_date': latest_available_date.isoformat(),
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'duration_definition': (
            'Duration is measured per crisis episode as consecutive days with at least one tagged negative crisis event '
            'for the same entity, then averaged within each sector.'
        ),
        'rows': serialize_rows(sector_rows[:limit]),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/find_storylines')
def insights_find_storylines_json():
    entity = request.args.get('entity', 'brand')
    sector = (request.args.get('sector') or '').strip()
    period_label = (request.args.get('period_label') or '').strip()
    include_non_crisis = (request.args.get('include_non_crisis') or '').strip().lower() in {'1', 'true', 'yes'}
    try:
        limit = int(request.args.get('limit', '3') or 3)
    except ValueError:
        limit = 3
    limit = min(max(limit, 1), 10)

    cache_key = f"insights_find_storylines:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    try:
        start_date, end_date, latest_available_date, days, window_mode, requested_start_date, requested_end_date = resolve_insights_window(
            entity,
            sector=sector,
            default_days=90,
            min_days=7,
            max_days=365,
        )
    except LookupError:
        return jsonify({'error': 'no_data'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    analytics_type, rows = fetch_negative_top_stories_narrative_rows(entity, start_date, end_date, sector)
    filtered_rows = []
    for row in rows:
        tag = (row.get('narrative_primary_tag') or '').strip()
        group = normalized_narrative_group(
            tag,
            row.get('narrative_primary_group'),
            row.get('narrative_is_crisis'),
        )
        if not include_non_crisis and group == 'non_crisis':
            continue
        filtered_rows.append(row)

    candidates = build_storyline_candidates(analytics_type, filtered_rows)
    candidates.sort(
        key=lambda row: (
            0 if row.get('storyline_type') == 'cross_sector_narrative' else 1,
            -(row.get('score') or 0),
            str(row.get('headline') or '').casefold(),
        )
    )

    selected: List[Dict] = []
    selected_keys = set()
    preferred_types = [
        'cross_sector_narrative',
        'sector_duration_outlier',
        'sector_tag_pattern',
    ]
    for storyline_type in preferred_types:
        match = next(
            (
                row for row in candidates
                if row.get('storyline_type') == storyline_type
                and row.get('storyline_key') not in selected_keys
            ),
            None,
        )
        if match:
            selected.append(match)
            selected_keys.add(match.get('storyline_key'))
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for row in sorted(candidates, key=lambda item: (-(item.get('score') or 0), str(item.get('headline') or '').casefold())):
            if row.get('storyline_key') in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(row.get('storyline_key'))
            if len(selected) >= limit:
                break

    payload = {
        'entity': analytics_type,
        'sector': sector,
        'period_label': period_label,
        'days': days,
        'limit': limit,
        'window_mode': window_mode,
        'requested_start_date': requested_start_date,
        'requested_end_date': requested_end_date,
        'include_non_crisis': include_non_crisis,
        'latest_available_date': latest_available_date.isoformat(),
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'duration_definition': (
            'Duration is measured from consecutive days with at least one tagged negative crisis event for the same entity, '
            'using Top Stories and recent negative newsfeed coverage.'
        ),
        'storylines': serialize_rows(selected),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


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


@app.route('/api/v1/insights/trend_summary')
def insights_trend_summary_json():
    entity = request.args.get('entity', 'brand')
    entity_name = (request.args.get('entity_name') or '').strip()
    if not entity_name:
        return jsonify({'error': 'entity_name is required'}), 400
    try:
        days = int(request.args.get('days', '30') or 30)
    except ValueError:
        days = 30
    try:
        weeks = int(request.args.get('weeks', '8') or 8)
    except ValueError:
        weeks = 8
    days = min(max(days, 7), 180)
    weeks = min(max(weeks, 1), 26)

    cache_key = f"insights_trend_summary:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    resolved = resolve_entity_lookup(entity, entity_name)
    if not resolved:
        suggestions = [entity_payload(row) for row in entity_lookup_suggestions(entity, entity_name)]
        return jsonify({'error': 'entity not found', 'suggestions': suggestions}), 404

    analytics_type = resolved['analytics_entity_type']
    entity_id = resolved['entity_id']
    lookback_days = max(days, (weeks * 7) + 6)
    daily_rows = query_dict(
        """
        select date,
               entity_type,
               entity_name,
               company,
               ceo,
               article_negative_count,
               article_total_count,
               article_negative_pct,
               serp_negative_count,
               serp_total_count,
               serp_controlled_count,
               serp_uncontrolled_count,
               top_stories_negative_count,
               top_stories_total_count,
               top_stories_controlled_count,
               top_stories_uncontrolled_count,
               crisis_risk_count
        from entity_daily_metrics_v
        where entity_type = %s
          and entity_id = %s
          and date >= (current_date - (%s || ' days')::interval)
        order by date
        """,
        (analytics_type, entity_id, lookback_days),
    )
    if not daily_rows:
        return jsonify({'error': 'no_data'}), 404

    anomaly_rows = query_dict(
        """
        select date,
               anomaly_type,
               severity_score,
               observed_value,
               baseline_value,
               article_negative_count,
               serp_uncontrolled_count,
               top_stories_negative_count,
               summary
        from entity_anomalies_v
        where entity_type = %s
          and entity_id = %s
          and date >= (current_date - (%s || ' days')::interval)
        order by date desc, severity_score desc
        limit 12
        """,
        (analytics_type, entity_id, days),
    )

    latest_row = daily_rows[-1]
    latest_date = coerce_date(latest_row.get('date'))
    visible_daily_rows = rows_within_dates(daily_rows, latest_date - timedelta(days=days - 1), latest_date)
    current_rows = rows_within_dates(daily_rows, latest_date - timedelta(days=6), latest_date)
    prior_rows = rows_within_dates(daily_rows, latest_date - timedelta(days=13), latest_date - timedelta(days=7))
    weekly_rollups = build_trailing_window_rollups(daily_rows, weeks)

    current_window = {
        'article_negative_count': sum_metric(current_rows, 'article_negative_count'),
        'serp_negative_count': sum_metric(current_rows, 'serp_negative_count'),
        'serp_uncontrolled_count': sum_metric(current_rows, 'serp_uncontrolled_count'),
        'top_stories_negative_count': sum_metric(current_rows, 'top_stories_negative_count'),
        'top_stories_uncontrolled_count': sum_metric(current_rows, 'top_stories_uncontrolled_count'),
        'crisis_risk_count': sum_metric(current_rows, 'crisis_risk_count'),
    }
    prior_window = {
        'article_negative_count': sum_metric(prior_rows, 'article_negative_count'),
        'serp_negative_count': sum_metric(prior_rows, 'serp_negative_count'),
        'serp_uncontrolled_count': sum_metric(prior_rows, 'serp_uncontrolled_count'),
        'top_stories_negative_count': sum_metric(prior_rows, 'top_stories_negative_count'),
        'top_stories_uncontrolled_count': sum_metric(prior_rows, 'top_stories_uncontrolled_count'),
        'crisis_risk_count': sum_metric(prior_rows, 'crisis_risk_count'),
    }
    delta_window = {
        key: current_window[key] - prior_window.get(key, 0)
        for key in current_window.keys()
    }
    search_nuance = build_search_nuance(current_window)

    payload = {
        'entity': entity_payload(resolved),
        'latest_date': latest_date.isoformat() if latest_date else '',
        'latest_metrics': serialize_rows([latest_row])[0],
        'current_7d': current_window,
        'prior_7d': prior_window,
        'delta_7d': delta_window,
        'search_impact': {
            'label': classify_search_impact(current_window),
            'negative_top_stories_days': sum(1 for row in visible_daily_rows if int(row.get('top_stories_negative_count') or 0) > 0),
            'crisis_top_stories_streak_days': top_stories_streak_days(visible_daily_rows),
            'latest_top_stories_negative_count': int(latest_row.get('top_stories_negative_count') or 0),
            'latest_top_stories_uncontrolled_count': int(latest_row.get('top_stories_uncontrolled_count') or 0),
            'latest_serp_negative_count': int(latest_row.get('serp_negative_count') or 0),
            'latest_serp_uncontrolled_count': int(latest_row.get('serp_uncontrolled_count') or 0),
        },
        'search_nuance': search_nuance,
        'daily_series': serialize_rows(visible_daily_rows),
        'weekly_rollups': weekly_rollups,
        'recent_anomalies': serialize_rows(anomaly_rows),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/anomalies')
def insights_anomalies_json():
    entity = (request.args.get('entity') or '').strip()
    entity_name = (request.args.get('entity_name') or '').strip()
    try:
        days = int(request.args.get('days', '30') or 30)
    except ValueError:
        days = 30
    try:
        limit = int(request.args.get('limit', '50') or 50)
    except ValueError:
        limit = 50
    days = min(max(days, 1), 180)
    limit = min(max(limit, 1), 200)

    cache_key = f"insights_anomalies:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    params = [days]
    scope_sql, params = scope_clause("company_id", params)
    entity_sql = ""
    payload_entity = None

    if entity_name:
        resolved = resolve_entity_lookup(entity or 'brand', entity_name)
        if not resolved:
            suggestions = [entity_payload(row) for row in entity_lookup_suggestions(entity or 'brand', entity_name)]
            return jsonify({'error': 'entity not found', 'suggestions': suggestions}), 404
        payload_entity = entity_payload(resolved)
        params.extend([resolved['analytics_entity_type'], resolved['entity_id']])
        entity_sql = " and entity_type = %s and entity_id = %s"
    elif entity:
        params.append(analytics_entity_type(entity))
        entity_sql = " and entity_type = %s"

    params.append(limit)
    rows = query_dict(
        f"""
        select date,
               entity_type,
               entity_id,
               company_id,
               ceo_id,
               entity_name,
               company,
               ceo,
               anomaly_type,
               severity_score,
               observed_value,
               baseline_value,
               article_negative_count,
               serp_uncontrolled_count,
               top_stories_negative_count,
               summary
        from entity_anomalies_v
        where date >= (current_date - (%s || ' days')::interval)
          {scope_sql}
          {entity_sql}
        order by date desc, severity_score desc
        limit %s
        """,
        tuple(params),
    )
    payload = {
        'entity': payload_entity,
        'days': days,
        'rows': serialize_rows(rows),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


@app.route('/api/v1/insights/evidence')
def insights_evidence_json():
    entity = request.args.get('entity', 'brand')
    entity_name = (request.args.get('entity_name') or '').strip()
    if not entity_name:
        return jsonify({'error': 'entity_name is required'}), 400
    try:
        days = int(request.args.get('days', '14') or 14)
    except ValueError:
        days = 14
    try:
        limit = int(request.args.get('limit', '50') or 50)
    except ValueError:
        limit = 50
    days = min(max(days, 1), 90)
    limit = min(max(limit, 1), 200)

    cache_key = f"insights_evidence:{request.query_string.decode('utf-8')}"
    cached = get_cached_json(cache_key)
    if cached is not None:
        return jsonify(cached)

    resolved = resolve_entity_lookup(entity, entity_name)
    if not resolved:
        suggestions = [entity_payload(row) for row in entity_lookup_suggestions(entity, entity_name)]
        return jsonify({'error': 'entity not found', 'suggestions': suggestions}), 404

    analytics_type = resolved['analytics_entity_type']
    entity_id = resolved['entity_id']
    latest_rows = query_rows(
        """
        select max(date)
        from entity_daily_metrics_v
        where entity_type = %s and entity_id = %s
        """,
        (analytics_type, entity_id),
    )
    latest_date = latest_rows[0][0] if latest_rows and latest_rows[0] else None
    if latest_date is None:
        return jsonify({'error': 'no_data'}), 404

    start_raw = (request.args.get('start_date') or '').strip()
    end_raw = (request.args.get('end_date') or '').strip()
    try:
        end_date = datetime.strptime(end_raw, '%Y-%m-%d').date() if end_raw else latest_date
        start_date = datetime.strptime(start_raw, '%Y-%m-%d').date() if start_raw else (end_date - timedelta(days=days - 1))
    except ValueError:
        return jsonify({'error': 'invalid date format (YYYY-MM-DD)'}), 400

    if canonical_entity_type(entity) == 'company':
        rows = query_dict(
            """
            with evidence as (
                select cad.date,
                       'article'::text as evidence_type,
                       a.title,
                       a.snippet,
                       a.canonical_url as url,
                       a.publisher as source,
                       coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment_label,
                       coalesce(ov.override_control_class, cam.llm_control_class, cam.control_class, '') as control_class,
                       cam.llm_risk_label,
                       null::text as narrative_primary_tag,
                       'negative'::text as included_reason,
                       2 as sort_weight
                from company_article_mentions_daily cad
                join articles a on a.id = cad.article_id
                left join company_article_mentions cam
                  on cam.company_id = cad.company_id and cam.article_id = cad.article_id
                left join company_article_overrides ov
                  on ov.company_id = cad.company_id and ov.article_id = cad.article_id
                where cad.company_id = %s
                  and cad.date between %s and %s
                  and coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'negative'

                union all

                select sr.run_at::date as date,
                       'serp'::text as evidence_type,
                       r.title,
                       r.snippet,
                       r.url,
                       coalesce(r.domain, sr.provider) as source,
                       coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) as sentiment_label,
                       coalesce(ov.override_control_class, r.llm_control_class, r.control_class, '') as control_class,
                       r.llm_risk_label,
                       null::text as narrative_primary_tag,
                       case
                           when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative'
                                and coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'uncontrolled'
                               then 'negative_and_uncontrolled'
                           when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative'
                               then 'negative'
                           else 'uncontrolled'
                       end as included_reason,
                       case
                           when coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'uncontrolled' then 3
                           else 1
                       end as sort_weight
                from serp_runs sr
                join serp_results r on r.serp_run_id = sr.id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where sr.entity_type = 'company'
                  and sr.company_id = %s
                  and sr.run_at::date between %s and %s
                  and (
                      coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative'
                      or coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'uncontrolled'
                  )

                union all

                select sfi.date,
                       'top_stories'::text as evidence_type,
                       sfi.title,
                       sfi.snippet,
                       sfi.url,
                       coalesce(sfi.source, sfi.domain) as source,
                       coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) as sentiment_label,
                       coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class, '') as control_class,
                       sfi.llm_risk_label,
                       sfi.narrative_primary_tag,
                       case
                           when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                                and coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'uncontrolled'
                               then 'negative_and_uncontrolled'
                           when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                               then 'negative'
                           else 'uncontrolled'
                       end as included_reason,
                       4 as sort_weight
                from serp_feature_items sfi
                left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                where sfi.entity_type = any(%s)
                  and sfi.entity_id = %s
                  and sfi.feature_type = 'top_stories_items'
                  and sfi.date between %s and %s
                  and (
                      coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                      or coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'uncontrolled'
                  )
            ),
            deduped as (
                select distinct on (date, evidence_type, coalesce(url, ''), coalesce(title, ''))
                       date,
                       evidence_type,
                       title,
                       snippet,
                       url,
                       source,
                       sentiment_label,
                       control_class,
                       llm_risk_label,
                       narrative_primary_tag,
                       included_reason,
                       sort_weight
                from evidence
                order by date desc,
                         evidence_type,
                         coalesce(url, ''),
                         coalesce(title, ''),
                         sort_weight desc
            )
            select date,
                   evidence_type,
                   title,
                   snippet,
                   url,
                   source,
                   sentiment_label,
                   control_class,
                   llm_risk_label,
                   narrative_primary_tag,
                   included_reason
            from deduped
            order by date desc, sort_weight desc, title
            limit %s
            """,
            (
                entity_id,
                start_date,
                end_date,
                entity_id,
                start_date,
                end_date,
                compatible_entity_types(entity),
                entity_id,
                start_date,
                end_date,
                limit,
            ),
        )
    else:
        rows = query_dict(
            """
            with evidence as (
                select cad.date,
                       'article'::text as evidence_type,
                       a.title,
                       a.snippet,
                       a.canonical_url as url,
                       a.publisher as source,
                       coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment_label,
                       coalesce(ov.override_control_class, cem.llm_control_class, cem.control_class, '') as control_class,
                       cem.llm_risk_label,
                       null::text as narrative_primary_tag,
                       'negative'::text as included_reason,
                       2 as sort_weight
                from ceo_article_mentions_daily cad
                join articles a on a.id = cad.article_id
                left join ceo_article_mentions cem
                  on cem.ceo_id = cad.ceo_id and cem.article_id = cad.article_id
                left join ceo_article_overrides ov
                  on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
                where cad.ceo_id = %s
                  and cad.date between %s and %s
                  and coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'negative'

                union all

                select sr.run_at::date as date,
                       'serp'::text as evidence_type,
                       r.title,
                       r.snippet,
                       r.url,
                       coalesce(r.domain, sr.provider) as source,
                       coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) as sentiment_label,
                       coalesce(ov.override_control_class, r.llm_control_class, r.control_class, '') as control_class,
                       r.llm_risk_label,
                       null::text as narrative_primary_tag,
                       case
                           when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative'
                                and coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'uncontrolled'
                               then 'negative_and_uncontrolled'
                           when coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative'
                               then 'negative'
                           else 'uncontrolled'
                       end as included_reason,
                       case
                           when coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'uncontrolled' then 3
                           else 1
                       end as sort_weight
                from serp_runs sr
                join serp_results r on r.serp_run_id = sr.id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where sr.entity_type = 'ceo'
                  and sr.ceo_id = %s
                  and sr.run_at::date between %s and %s
                  and (
                      coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) = 'negative'
                      or coalesce(ov.override_control_class, r.llm_control_class, r.control_class) = 'uncontrolled'
                  )

                union all

                select sfi.date,
                       'top_stories'::text as evidence_type,
                       sfi.title,
                       sfi.snippet,
                       sfi.url,
                       coalesce(sfi.source, sfi.domain) as source,
                       coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) as sentiment_label,
                       coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class, '') as control_class,
                       sfi.llm_risk_label,
                       sfi.narrative_primary_tag,
                       case
                           when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                                and coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'uncontrolled'
                               then 'negative_and_uncontrolled'
                           when coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                               then 'negative'
                           else 'uncontrolled'
                       end as included_reason,
                       4 as sort_weight
                from serp_feature_items sfi
                left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                where sfi.entity_type = 'ceo'
                  and sfi.entity_id = %s
                  and sfi.feature_type = 'top_stories_items'
                  and sfi.date between %s and %s
                  and (
                      coalesce(ov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) = 'negative'
                      or coalesce(ov.override_control_class, sfi.llm_control_class, sfi.control_class) = 'uncontrolled'
                  )
            ),
            deduped as (
                select distinct on (date, evidence_type, coalesce(url, ''), coalesce(title, ''))
                       date,
                       evidence_type,
                       title,
                       snippet,
                       url,
                       source,
                       sentiment_label,
                       control_class,
                       llm_risk_label,
                       narrative_primary_tag,
                       included_reason,
                       sort_weight
                from evidence
                order by date desc,
                         evidence_type,
                         coalesce(url, ''),
                         coalesce(title, ''),
                         sort_weight desc
            )
            select date,
                   evidence_type,
                   title,
                   snippet,
                   url,
                   source,
                   sentiment_label,
                   control_class,
                   llm_risk_label,
                   narrative_primary_tag,
                   included_reason
            from deduped
            order by date desc, sort_weight desc, title
            limit %s
            """,
            (
                entity_id,
                start_date,
                end_date,
                entity_id,
                start_date,
                end_date,
                entity_id,
                start_date,
                end_date,
                limit,
            ),
        )

    payload = {
        'entity': entity_payload(resolved),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'summary': summarize_evidence_rows(rows),
        'rows': serialize_rows(rows),
    }
    set_cached_json(cache_key, payload)
    return jsonify(payload)


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

    entity_type = canonical_entity_type(entity)
    entity_types = compatible_entity_types(entity)
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


@app.route('/api/internal/refresh_aggregates', methods=['POST'])
def refresh_aggregates():
    user_email = require_internal_user()
    if not user_email:
        return jsonify({'error': 'unauthorized'}), 401
    global _refresh_in_progress, _refresh_last_status

    def run_refresh():
        global _refresh_in_progress, _refresh_last_status
        started = datetime.utcnow()
        try:
            lock_conn = get_autocommit_conn("refresh_aggregates")
        except Exception:
            with _refresh_lock:
                _refresh_last_status = {
                    'status': 'failed',
                    'started_at': started.isoformat() + 'Z',
                    'finished_at': started.isoformat() + 'Z',
                    'duration_ms': 0,
                    'error': 'db_unavailable',
                }
                _refresh_in_progress = False
            return
        if not _try_acquire_refresh_lock(lock_conn):
            with _refresh_lock:
                _refresh_last_status = {
                    'status': 'skipped',
                    'started_at': started.isoformat() + 'Z',
                    'finished_at': datetime.utcnow().isoformat() + 'Z',
                    'duration_ms': 0,
                    'error': 'refresh_locked',
                }
                _refresh_in_progress = False
            put_conn(lock_conn)
            return
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
            refresh_serp_feature_daily_view(conn=lock_conn)
            refresh_serp_feature_control_daily_view(conn=lock_conn)
            refresh_serp_feature_daily_index_view(conn=lock_conn)
            refresh_serp_feature_control_daily_index_view(conn=lock_conn)
            refresh_article_daily_counts_view(conn=lock_conn)
            refresh_serp_daily_counts_view(conn=lock_conn)
            clear_api_cache_prefix("negative_summary:")
            clear_api_cache_prefix("daily_counts:")
            clear_api_cache_prefix("serp_features:")
            clear_api_cache_prefix("serp_feature_controls:")
            clear_narrative_api_caches()
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
            _release_refresh_lock(lock_conn)
            put_conn(lock_conn)
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
    serp_feature_item_context = None
    crisis_event_context = None
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
            if mention_type == 'company_article':
                crisis_event_context = load_company_article_crisis_event_context(cur, mention_id)
            elif mention_type == 'ceo_article':
                crisis_event_context = load_ceo_article_crisis_event_context(cur, mention_id)
            if mention_type == 'serp_feature_item':
                serp_feature_item_context = load_serp_feature_item_narrative_context(cur, serp_feature_item_id)
                if (
                    serp_feature_item_context
                    and (serp_feature_item_context.get('feature_type') or '').strip() == 'top_stories_items'
                ):
                    raw_entity_type = str(serp_feature_item_context.get('entity_type') or '').strip().lower()
                    crisis_event_context = {
                        'entity_types': ['brand'] if raw_entity_type in {'brand', 'company'} else ['ceo'],
                        'entity_id': serp_feature_item_context.get('entity_id'),
                        'entity_name': serp_feature_item_context.get('entity_name') or '',
                        'start_date': serp_feature_item_context.get('date'),
                        'end_date': datetime.utcnow().date(),
                    }
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
                lock_conn = get_autocommit_conn("refresh_after_override")
            except Exception:
                app.logger.warning("refresh after override skipped: db unavailable")
                return
            if not _try_acquire_refresh_lock(lock_conn):
                app.logger.info("refresh after override skipped: refresh already running")
                put_conn(lock_conn)
                return
            try:
                if crisis_event_context:
                    try:
                        with lock_conn.cursor() as event_cur:
                            rows = recompute_entity_crisis_event_window(
                                event_cur,
                                crisis_event_context.get('start_date'),
                                crisis_event_context.get('end_date'),
                                crisis_event_context.get('entity_types') or [],
                                entity_id=crisis_event_context.get('entity_id'),
                            )
                        app.logger.info(
                            "override_crisis_event_recomputed entity_types=%s entity_name=%s start_date=%s end_date=%s rows=%s",
                            crisis_event_context.get('entity_types'),
                            crisis_event_context.get('entity_name'),
                            crisis_event_context.get('start_date'),
                            crisis_event_context.get('end_date'),
                            len(rows),
                        )
                    except Exception:
                        app.logger.exception("override crisis event recompute failed")
                if mention_type in {'company_article', 'ceo_article'}:
                    refresh_article_daily_counts_view(conn=lock_conn)
                    clear_api_cache_prefix("negative_summary:")
                    clear_api_cache_prefix("daily_counts:")
                    clear_narrative_api_caches()
                if mention_type == 'serp_feature_item':
                    if (
                        serp_feature_item_context
                        and (serp_feature_item_context.get('feature_type') or '').strip() == 'top_stories_items'
                    ):
                        try:
                            with lock_conn.cursor() as narrative_cur:
                                result = recompute_entity_day_narrative_rollup(
                                    narrative_cur,
                                    row_date=serp_feature_item_context.get('date'),
                                    entity_type=str(serp_feature_item_context.get('entity_type') or ''),
                                    entity_id=serp_feature_item_context.get('entity_id'),
                                    entity_name=str(serp_feature_item_context.get('entity_name') or ''),
                                )
                            app.logger.info(
                                "override_narrative_rollup_recomputed date=%s entity_type=%s entity_name=%s primary_tag=%s updated_items=%s",
                                serp_feature_item_context.get('date'),
                                serp_feature_item_context.get('entity_type'),
                                serp_feature_item_context.get('entity_name'),
                                result.get('primary_tag'),
                                result.get('updated_items'),
                            )
                        except Exception:
                            app.logger.exception("override narrative rollup recompute failed")
                    refresh_serp_feature_daily_view(conn=lock_conn)
                    refresh_serp_feature_control_daily_view(conn=lock_conn)
                    refresh_serp_feature_daily_index_view(conn=lock_conn)
                    refresh_serp_feature_control_daily_index_view(conn=lock_conn)
                    clear_api_cache_prefix("serp_features:")
                    clear_api_cache_prefix("serp_feature_controls:")
                    clear_narrative_api_caches()
                if mention_type == 'serp_result':
                    refresh_serp_daily_counts_view(conn=lock_conn)
                    clear_api_cache_prefix("daily_counts:")
            except Exception:
                app.logger.exception("refresh after override failed")
            finally:
                _release_refresh_lock(lock_conn)
                put_conn(lock_conn)

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
                       to_char(a.published_at at time zone 'UTC', 'YYYY-MM-DD') as published_date,
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
            headers = ['company','title','url','source','published_date','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
        else:
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select ceo.name as ceo, c.name as company, a.title, a.canonical_url as url, a.publisher as source,
                       to_char(a.published_at at time zone 'UTC', 'YYYY-MM-DD') as published_date,
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
            headers = ['ceo','company','title','url','source','published_date','sentiment','control_class','sentiment_override','control_override','llm_label','mention_id']
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
                       to_jsonb(r) ->> 'published_date' as published_date,
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
            headers = ['company','title','url','position','snippet','published_date','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
        else:
            params = [dstr]
            scope_sql, params = scope_clause("c.id", params)
            rows = query_rows(
                f"""
                select ceo.name as ceo, c.name as company, r.title, r.url, r.rank as position, r.snippet,
                       to_jsonb(r) ->> 'published_date' as published_date,
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
            headers = ['ceo','company','title','url','position','snippet','published_date','sentiment','controlled','sentiment_override','control_override','llm_label','serp_result_id']
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
    rows = negative_summary_live(days=days, company=None, mode='detail')
    out_rows = []
    for row in rows:
        negative_count = int(row.get('negative_count') or 0)
        if negative_count <= 0:
            continue
        date_val = row.get('date')
        if hasattr(date_val, 'isoformat'):
            date_str = date_val.isoformat()
        else:
            date_str = str(date_val or '')
        out_rows.append((
            date_str,
            row.get('company') or '',
            row.get('ceo') or '',
            negative_count,
            row.get('top_headlines') or '',
            row.get('article_type') or '',
        ))
    headers = ['date','company','ceo','negative_count','top_headlines','article_type']
    csv_text = rows_to_csv(headers, out_rows)
    return Response(csv_text, content_type='text/csv')


def negative_summary_live(days: int | None = None, company: str | None = None, mode: str = 'detail'):
    lookback_days = days or int(os.environ.get('NEGATIVE_SUMMARY_LOOKBACK_DAYS', '90'))
    start_date = datetime.utcnow().date() - timedelta(days=lookback_days)
    params = [start_date]
    scope_sql, params = scope_clause("c.id", params)
    company_sql = ""
    if company and mode != 'index':
        params.append(company)
        company_sql = " and c.name = %s"

    if mode == 'index':
        sql = f"""
            select cad.date as date,
                   'Index' as company,
                   '' as ceo,
                   'brand'::text as article_type,
                   count(*) filter (where coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'negative') as negative_count,
                   '' as top_headlines,
                   count(*) filter (where cam.llm_risk_label = 'crisis_risk') as crisis_risk_count
            from company_article_mentions_daily cad
            join companies c on c.id = cad.company_id
            left join company_article_mentions cam
              on cam.company_id = cad.company_id and cam.article_id = cad.article_id
            left join company_article_overrides ov
              on ov.company_id = cad.company_id and ov.article_id = cad.article_id
            where cad.date >= %s {scope_sql}
            group by cad.date
            union all
            select cad.date as date,
                   'Index' as company,
                   '' as ceo,
                   'ceo'::text as article_type,
                   count(*) filter (where coalesce(ov.override_sentiment_label, cad.sentiment_label) = 'negative') as negative_count,
                   '' as top_headlines,
                   count(*) filter (where cem.llm_risk_label = 'crisis_risk') as crisis_risk_count
            from ceo_article_mentions_daily cad
            join ceos ceo on ceo.id = cad.ceo_id
            join companies c on c.id = ceo.company_id
            left join ceo_article_mentions cem
              on cem.ceo_id = cad.ceo_id and cem.article_id = cad.article_id
            left join ceo_article_overrides ov
              on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
            where cad.date >= %s {scope_sql}
            group by cad.date
            order by date desc, article_type
        """
        rows = query_dict(sql, tuple(params + params))
        return rows

    base = f"""
        with base as (
            select cad.date as date,
                   c.id as company_id,
                   c.name as company,
                   ''::text as ceo,
                   coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
                   a.title as title,
                   cam.llm_risk_label as llm_risk_label,
                   'brand'::text as article_type
            from company_article_mentions_daily cad
            join companies c on c.id = cad.company_id
            join articles a on a.id = cad.article_id
            left join company_article_mentions cam
              on cam.company_id = cad.company_id and cam.article_id = cad.article_id
            left join company_article_overrides ov
              on ov.company_id = cad.company_id and ov.article_id = cad.article_id
            where cad.date >= %s {scope_sql}{company_sql}
            union all
            select cad.date as date,
                   c.id as company_id,
                   c.name as company,
                   coalesce(ceo.name, '') as ceo,
                   coalesce(ov.override_sentiment_label, cad.sentiment_label) as sentiment,
                   a.title as title,
                   cem.llm_risk_label as llm_risk_label,
                   'ceo'::text as article_type
            from ceo_article_mentions_daily cad
            join ceos ceo on ceo.id = cad.ceo_id
            join companies c on c.id = ceo.company_id
            join articles a on a.id = cad.article_id
            left join ceo_article_mentions cem
              on cem.ceo_id = cad.ceo_id and cem.article_id = cad.article_id
            left join ceo_article_overrides ov
              on ov.ceo_id = cad.ceo_id and ov.article_id = cad.article_id
            where cad.date >= %s {scope_sql}{company_sql}
        )
    """
    sql = f"""
        {base}
        select date,
               company,
               ceo,
               article_type,
               count(*) filter (where sentiment = 'negative') as negative_count,
               array_to_string(
                   (array_agg(title order by title) filter (where sentiment = 'negative'))[1:3],
                   ' | '
               ) as top_headlines,
               count(*) filter (where llm_risk_label = 'crisis_risk') as crisis_risk_count
        from base
        group by date, company, ceo, article_type
        order by date desc, company
    """
    rows = query_dict(sql, tuple(params + params))
    return rows


def clear_api_cache_prefix(prefix: str) -> None:
    if not prefix:
        return
    keys = [k for k in _api_cache.keys() if k.startswith(prefix)]
    for key in keys:
        _api_cache.pop(key, None)


def refresh_serp_feature_daily_view(conn=None) -> None:
    owned = conn is None
    if conn is None:
        conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    _set_application_name(conn, "refresh_serp_feature_daily_view")
    try:
        _reset_conn(conn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_daily_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_daily_mv")
    finally:
        if owned:
            put_conn(conn)


def refresh_serp_feature_control_daily_view(conn=None) -> None:
    owned = conn is None
    if conn is None:
        conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    _set_application_name(conn, "refresh_serp_feature_control_daily_view")
    try:
        _reset_conn(conn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_control_daily_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_control_daily_mv")
    finally:
        if owned:
            put_conn(conn)


def refresh_serp_feature_daily_index_view(conn=None) -> None:
    owned = conn is None
    if conn is None:
        conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    _set_application_name(conn, "refresh_serp_feature_daily_index_view")
    try:
        _reset_conn(conn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_daily_index_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_daily_index_mv")
    finally:
        if owned:
            put_conn(conn)


def refresh_serp_feature_control_daily_index_view(conn=None) -> None:
    owned = conn is None
    if conn is None:
        conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    _set_application_name(conn, "refresh_serp_feature_control_daily_index_view")
    try:
        _reset_conn(conn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_feature_control_daily_index_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_feature_control_daily_index_mv")
    finally:
        if owned:
            put_conn(conn)


def refresh_article_daily_counts_view(conn=None) -> None:
    owned = conn is None
    if conn is None:
        conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    _set_application_name(conn, "refresh_article_daily_counts_view")
    try:
        _reset_conn(conn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently article_daily_counts_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view article_daily_counts_mv")
    finally:
        if owned:
            put_conn(conn)


def refresh_serp_daily_counts_view(conn=None) -> None:
    owned = conn is None
    if conn is None:
        conn = get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not configured")
    _set_application_name(conn, "refresh_serp_daily_counts_view")
    try:
        _reset_conn(conn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view concurrently serp_daily_counts_mv")
    except Exception:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("refresh materialized view serp_daily_counts_mv")
    finally:
        if owned:
            put_conn(conn)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
