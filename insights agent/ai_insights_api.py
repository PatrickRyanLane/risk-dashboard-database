#!/usr/bin/env python3
"""Thin wrappers over the dashboard AI insights endpoints."""

import base64
import json
import os
import secrets
import sys
import subprocess
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlencode, urlparse

import requests


BASE_URL = os.environ.get("AI_INSIGHTS_BASE_URL", "").rstrip("/")
AUTH_BEARER = os.environ.get("AI_INSIGHTS_AUTH_BEARER", "")
AUTH_AUDIENCE = os.environ.get("AI_INSIGHTS_AUTH_AUDIENCE", BASE_URL).rstrip("/")
AUTO_AUTH = os.environ.get("AI_INSIGHTS_AUTO_AUTH", "true").lower() in {"1", "true", "yes"}
AUTH_MODE = os.environ.get("AI_INSIGHTS_AUTH_MODE", "auto").strip().lower()
REQUEST_TIMEOUT = int(os.environ.get("AI_INSIGHTS_TIMEOUT", "60"))
IAP_CLIENT_ID = os.environ.get("AI_INSIGHTS_IAP_CLIENT_ID", "").strip()
IAP_CLIENT_SECRET = os.environ.get("AI_INSIGHTS_IAP_CLIENT_SECRET", "").strip()
IAP_TOKEN_FILE = Path(
    os.environ.get(
        "AI_INSIGHTS_IAP_TOKEN_FILE",
        "~/.config/ai-insights-agent/iap_desktop_auth.json",
    )
).expanduser()
OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
IAP_SCOPES = "openid email"

BASE_DIR = Path(__file__).resolve().parent
TOOL_CONTRACT_PATH = BASE_DIR / "ai-insights-tool-contract.json"
SYSTEM_PROMPT_PATH = BASE_DIR / "ai-insights-system-prompt.txt"

_TOKEN_CACHE: Dict[str, Any] = {"token": "", "exp": 0}


def load_tool_contract() -> Dict[str, Any]:
    return json.loads(TOOL_CONTRACT_PATH.read_text())


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text().strip()


def build_openai_tools() -> list[dict[str, Any]]:
    contract = load_tool_contract()
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in contract["tools"]
    ]


def build_gemini_function_declarations() -> list[dict[str, Any]]:
    contract = load_tool_contract()
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in contract["tools"]
    ]


def _jwt_exp_unverified(token: str) -> int:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return 0
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload + padding)
        body = json.loads(decoded.decode("utf-8"))
        return int(body.get("exp") or 0)
    except Exception:
        return 0


def _run_gcloud_identity_token(*extra_args: str) -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token", *extra_args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_iap_token_state() -> Dict[str, Any]:
    if not IAP_TOKEN_FILE.exists():
        return {}
    try:
        return json.loads(IAP_TOKEN_FILE.read_text())
    except Exception:
        return {}


def _save_iap_token_state(state: Dict[str, Any]) -> None:
    IAP_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    IAP_TOKEN_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.chmod(IAP_TOKEN_FILE, 0o600)


def _resolve_iap_client_id(state: Dict[str, Any]) -> str:
    return IAP_CLIENT_ID or str(state.get("client_id") or "").strip()


def _resolve_iap_client_secret(state: Dict[str, Any]) -> str:
    return IAP_CLIENT_SECRET or str(state.get("client_secret") or "").strip()


def _desktop_iap_mode_enabled() -> bool:
    if AUTH_MODE in {"iap-desktop", "iap_desktop", "iap"}:
        return True
    if AUTH_MODE != "auto":
        return False
    return bool(IAP_CLIENT_ID or IAP_TOKEN_FILE.exists())


def _post_oauth_token(data: Dict[str, str]) -> Dict[str, Any]:
    response = requests.post(OAUTH_TOKEN_URL, data=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        self.server.oauth_response = {k: v[0] for k, v in params.items()}  # type: ignore[attr-defined]

        if "error" in params:
            message = "Authentication failed. You can close this window."
        else:
            message = "Authentication complete. You can close this window."

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            (
                "<html><body style='font-family: sans-serif; padding: 24px;'>"
                f"<p>{message}</p></body></html>"
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def init_iap_desktop_auth(open_browser: bool = True, timeout_seconds: int = 300) -> str:
    """Run a one-time desktop OAuth flow and save a refresh token locally."""
    client_id = IAP_CLIENT_ID.strip()
    client_secret = IAP_CLIENT_SECRET.strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "AI_INSIGHTS_IAP_CLIENT_ID and AI_INSIGHTS_IAP_CLIENT_SECRET are required "
            "to initialize IAP desktop OAuth."
        )

    server = HTTPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
    redirect_uri = f"http://127.0.0.1:{server.server_port}/"
    state = secrets.token_urlsafe(24)

    auth_params = {
        "client_id": client_id,
        "response_type": "code",
        "scope": IAP_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "redirect_uri": redirect_uri,
        "cred_ref": "true",
        "state": state,
    }
    auth_url = f"{OAUTH_AUTH_URL}?{urlencode(auth_params)}"

    print("\nOpen this URL to sign in for IAP programmatic access:\n")
    print(auth_url)
    print("")

    if open_browser:
        webbrowser.open(auth_url, new=1, autoraise=True)

    server.timeout = timeout_seconds
    server.handle_request()
    response = getattr(server, "oauth_response", {})
    if not response:
        raise RuntimeError("Timed out waiting for the OAuth redirect back from the browser.")
    if response.get("state") != state:
        raise RuntimeError("OAuth state mismatch. Please try again.")
    if response.get("error"):
        raise RuntimeError(f"OAuth sign-in failed: {response['error']}")

    code = response.get("code", "")
    if not code:
        raise RuntimeError("OAuth redirect did not include an authorization code.")

    token_data = _post_oauth_token(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    )

    refresh_token = str(token_data.get("refresh_token") or "").strip()
    id_token = str(token_data.get("id_token") or "").strip()
    if not refresh_token:
        raise RuntimeError(
            "OAuth exchange succeeded but no refresh_token was returned. "
            "Try again after revoking the app or confirm the desktop client is allowlisted for IAP programmatic access."
        )
    if not id_token:
        raise RuntimeError("OAuth exchange succeeded but no id_token was returned.")

    state_doc = {
        "auth_type": "iap_desktop_oauth",
        "base_url": BASE_URL,
        "client_id": client_id,
        "client_secret": client_secret,
        "id_token": id_token,
        "id_token_exp": _jwt_exp_unverified(id_token),
        "refresh_token": refresh_token,
        "scopes": IAP_SCOPES.split(),
        "token_uri": OAUTH_TOKEN_URL,
        "updated_at": int(time.time()),
    }
    _save_iap_token_state(state_doc)
    return str(IAP_TOKEN_FILE)


def _desktop_iap_bearer_token() -> str:
    state = _load_iap_token_state()
    client_id = _resolve_iap_client_id(state)
    client_secret = _resolve_iap_client_secret(state)
    refresh_token = str(state.get("refresh_token") or "").strip()

    if not client_id or not client_secret:
        raise RuntimeError(
            "IAP desktop OAuth is enabled, but client credentials are missing. "
            "Set AI_INSIGHTS_IAP_CLIENT_ID and AI_INSIGHTS_IAP_CLIENT_SECRET."
        )
    if not refresh_token:
        raise RuntimeError(
            "IAP desktop OAuth is enabled, but no refresh token was found. "
            "Run `python ai_insights_api.py init-iap-desktop-auth` first."
        )

    now = int(time.time())
    cached = str(state.get("id_token") or "").strip()
    exp = int(state.get("id_token_exp") or 0)
    if cached and exp and now < exp - 120:
        return cached

    token_data = _post_oauth_token(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    )

    id_token = str(token_data.get("id_token") or "").strip()
    if not id_token:
        raise RuntimeError("Refresh token exchange succeeded but no id_token was returned.")

    state.update(
        {
            "auth_type": "iap_desktop_oauth",
            "client_id": client_id,
            "client_secret": client_secret,
            "id_token": id_token,
            "id_token_exp": _jwt_exp_unverified(id_token) or (now + 3300),
            "updated_at": now,
        }
    )
    _save_iap_token_state(state)
    return id_token


def _auto_cloud_run_bearer_token() -> str:
    now = int(time.time())
    cached = _TOKEN_CACHE.get("token") or ""
    exp = int(_TOKEN_CACHE.get("exp") or 0)
    if cached and exp and now < exp - 120:
        return cached

    if not AUTO_AUTH or not BASE_URL.startswith("https://"):
        return ""

    token = ""
    if AUTH_AUDIENCE:
        try:
            token = _run_gcloud_identity_token(f"--audiences={AUTH_AUDIENCE}")
        except Exception:
            token = ""
    if not token:
        token = _run_gcloud_identity_token()

    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["exp"] = _jwt_exp_unverified(token) or (now + 3300)
    return token


def _resolve_bearer_token() -> str:
    if AUTH_BEARER:
        return AUTH_BEARER
    if _desktop_iap_mode_enabled():
        return _desktop_iap_bearer_token()
    return _auto_cloud_run_bearer_token()


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    bearer = _resolve_bearer_token()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not BASE_URL:
        raise RuntimeError("AI_INSIGHTS_BASE_URL is required")
    response = requests.get(
        f"{BASE_URL}{path}",
        params={k: v for k, v in params.items() if v is not None and v != ""},
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        detail = (response.text or "").strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        raise RuntimeError(
            f"Insights API request failed: {response.status_code} {response.reason} "
            f"for {response.url}. Response body: {detail}"
        )
    return response.json()


def resolve_entity(entity_name: str, entity: str = "brand", limit: int = 5) -> Dict[str, Any]:
    """Resolve a brand or CEO name to the canonical entity in the dashboard."""
    return _get(
        "/api/v1/insights/resolve_entity",
        {"entity": entity, "entity_name": entity_name, "limit": limit},
    )


def resolve_sector(sector_name: str, limit: int = 5) -> Dict[str, Any]:
    """Resolve an industry or sector name to the canonical dashboard sector label."""
    return _get(
        "/api/v1/insights/resolve_sector",
        {"sector_name": sector_name, "limit": limit},
    )


def screen_entities(
    metric: str = "top_stories_negative_count",
    entity: str = "brand",
    days: int = 1,
    limit: int = 20,
    min_value: float = 1,
    sector: str | None = None,
) -> Dict[str, Any]:
    """Rank visible entities by a selected metric over the latest available window."""
    return _get(
        "/api/v1/insights/screen_entities",
        {
            "metric": metric,
            "entity": entity,
            "days": days,
            "limit": limit,
            "min_value": min_value,
            "sector": sector,
        },
    )


def get_sector_baseline(
    sector: str,
    metric: str = "top_stories_negative_count",
    entity: str = "brand",
    days: int = 30,
    limit: int = 10,
    entity_name: str | None = None,
) -> Dict[str, Any]:
    """Compare a brand or CEO, or the whole peer set, against the selected sector baseline."""
    return _get(
        "/api/v1/insights/sector_baseline",
        {
            "sector": sector,
            "metric": metric,
            "entity": entity,
            "days": days,
            "limit": limit,
            "entity_name": entity_name,
        },
    )


def aggregate_crisis_patterns(
    sector: str | None = None,
    entity: str = "brand",
    days: int = 90,
    start_date: str | None = None,
    end_date: str | None = None,
    period_label: str | None = None,
    limit: int = 10,
    include_non_crisis: bool = False,
) -> Dict[str, Any]:
    """Aggregate crisis types and search-duration patterns across a sector or the visible scope."""
    return _get(
        "/api/v1/insights/aggregate_crisis_patterns",
        {
            "sector": sector,
            "entity": entity,
            "days": days,
            "start_date": start_date,
            "end_date": end_date,
            "period_label": period_label,
            "limit": limit,
            "include_non_crisis": "true" if include_non_crisis else None,
        },
    )


def aggregate_industry_durations(
    entity: str = "brand",
    days: int = 90,
    start_date: str | None = None,
    end_date: str | None = None,
    period_label: str | None = None,
    limit: int = 25,
    include_non_crisis: bool = False,
) -> Dict[str, Any]:
    """Return average crisis duration broken down by industry/sector."""
    return _get(
        "/api/v1/insights/aggregate_industry_durations",
        {
            "entity": entity,
            "days": days,
            "start_date": start_date,
            "end_date": end_date,
            "period_label": period_label,
            "limit": limit,
            "include_non_crisis": "true" if include_non_crisis else None,
        },
    )


def find_storylines(
    entity: str = "brand",
    days: int = 90,
    start_date: str | None = None,
    end_date: str | None = None,
    period_label: str | None = None,
    sector: str | None = None,
    limit: int = 3,
    include_non_crisis: bool = False,
) -> Dict[str, Any]:
    """Find editorially useful storyline candidates for thought leadership and research prompts."""
    return _get(
        "/api/v1/insights/find_storylines",
        {
            "entity": entity,
            "days": days,
            "start_date": start_date,
            "end_date": end_date,
            "period_label": period_label,
            "sector": sector,
            "limit": limit,
            "include_non_crisis": "true" if include_non_crisis else None,
        },
    )


def _latest_scope_date(entity: str = "brand", sector: str | None = None) -> str:
    result = screen_entities(
        metric="article_negative_count",
        entity=entity,
        days=1,
        limit=1,
        min_value=0,
        sector=sector,
    )
    latest_date = str(result.get("latest_available_date") or "").strip()
    if not latest_date:
        raise RuntimeError("Could not determine the latest available date for the requested scope.")
    return latest_date


def _latest_entity_date(entity_name: str, entity: str = "brand") -> str:
    result = get_trend_summary(entity_name=entity_name, entity=entity, days=30, weeks=1)
    latest_date = str(result.get("latest_date") or "").strip()
    if not latest_date:
        raise RuntimeError(f"Could not determine the latest available date for {entity_name}.")
    return latest_date


def get_trend_summary(
    entity_name: str,
    entity: str = "brand",
    days: int = 30,
    weeks: int = 8,
) -> Dict[str, Any]:
    """Return trend summary metrics for one brand or CEO."""
    return _get(
        "/api/v1/insights/trend_summary",
        {"entity": entity, "entity_name": entity_name, "days": days, "weeks": weeks},
    )


def get_anomalies(
    entity: str | None = None,
    entity_name: str | None = None,
    days: int = 30,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return recent anomaly rows for one entity or the visible scope."""
    return _get(
        "/api/v1/insights/anomalies",
        {"entity": entity, "entity_name": entity_name, "days": days, "limit": limit},
    )


def get_evidence(
    entity_name: str,
    entity: str = "brand",
    days: int = 14,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return supporting evidence rows for one brand or CEO."""
    return _get(
        "/api/v1/insights/evidence",
        {
            "entity": entity,
            "entity_name": entity_name,
            "days": days,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
    )


def get_narrative_timeline(
    entity_name: str,
    entity: str = "brand",
    days: int = 90,
    date: str | None = None,
) -> Dict[str, Any]:
    """Return narrative-tag persistence and current crisis windows for one brand or CEO."""
    target_date = date or _latest_entity_date(entity_name=entity_name, entity=entity)
    return _get(
        "/api/v1/narrative_timeline",
        {
            "entity": entity,
            "entity_name": entity_name,
            "days": days,
            "date": target_date,
        },
    )


def get_narrative_tags(
    entity: str = "brand",
    date: str | None = None,
) -> Dict[str, Any]:
    """Return the dominant narrative tags visible on the latest or requested date."""
    target_date = date or _latest_scope_date(entity=entity)
    rows = _get(
        "/api/v1/narrative_tags",
        {
            "entity": entity,
            "date": target_date,
        },
    )
    return {
        "entity": entity,
        "date": target_date,
        "rows": rows,
    }


def get_search_feature_series(
    entity_name: str,
    feature_type: str,
    entity: str = "brand",
    days: int = 30,
) -> Dict[str, Any]:
    """Return daily counts for one search feature, split by sentiment and control."""
    rows = _get(
        "/api/v1/serp_feature_series",
        {
            "entity": entity,
            "entity_name": entity_name,
            "feature_type": feature_type,
            "days": days,
        },
    )
    return {
        "entity": entity,
        "entity_name": entity_name,
        "feature_type": feature_type,
        "days": days,
        "rows": rows,
    }


def get_search_feature_items(
    entity: str = "brand",
    entity_name: str | None = None,
    feature_type: str | None = None,
    date: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Return the underlying search-feature items for a specific date and optional entity."""
    target_date = date or (
        _latest_entity_date(entity_name=entity_name, entity=entity)
        if entity_name
        else _latest_scope_date(entity=entity)
    )
    rows = _get(
        "/api/v1/serp_feature_items",
        {
            "entity": entity,
            "entity_name": entity_name,
            "feature_type": feature_type,
            "date": target_date,
            "limit": limit,
            "offset": offset,
        },
    )
    return {
        "entity": entity,
        "entity_name": entity_name,
        "feature_type": feature_type,
        "date": target_date,
        "limit": limit,
        "offset": offset,
        "rows": rows,
    }


def compare_entities(
    entity_a_name: str,
    entity_b_name: str,
    entity: str = "brand",
    days: int = 30,
    weeks: int = 8,
) -> Dict[str, Any]:
    """Compare two brands or CEOs across recent coverage, search, and control metrics."""
    summary_a = get_trend_summary(entity_name=entity_a_name, entity=entity, days=days, weeks=weeks)
    summary_b = get_trend_summary(entity_name=entity_b_name, entity=entity, days=days, weeks=weeks)

    entity_a = summary_a.get("entity") or {"entity_name": entity_a_name, "entity_type": entity}
    entity_b = summary_b.get("entity") or {"entity_name": entity_b_name, "entity_type": entity}
    metrics = [
        "article_negative_count",
        "serp_negative_count",
        "serp_uncontrolled_count",
        "top_stories_negative_count",
        "top_stories_uncontrolled_count",
        "crisis_risk_count",
    ]

    metric_rows = []
    for metric in metrics:
        current_a = float((summary_a.get("current_7d") or {}).get(metric) or 0)
        current_b = float((summary_b.get("current_7d") or {}).get(metric) or 0)
        delta_a = float((summary_a.get("delta_7d") or {}).get(metric) or 0)
        delta_b = float((summary_b.get("delta_7d") or {}).get(metric) or 0)
        if current_a > current_b:
            current_leader = entity_a.get("entity_name")
        elif current_b > current_a:
            current_leader = entity_b.get("entity_name")
        else:
            current_leader = "tie"
        if delta_a > delta_b:
            delta_leader = entity_a.get("entity_name")
        elif delta_b > delta_a:
            delta_leader = entity_b.get("entity_name")
        else:
            delta_leader = "tie"
        metric_rows.append(
            {
                "metric": metric,
                "entity_a_current_7d": current_a,
                "entity_b_current_7d": current_b,
                "entity_a_delta_7d": delta_a,
                "entity_b_delta_7d": delta_b,
                "current_7d_leader": current_leader,
                "delta_7d_leader": delta_leader,
            }
        )

    return {
        "entity": entity,
        "days": days,
        "weeks": weeks,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "latest_dates": {
            "entity_a": summary_a.get("latest_date"),
            "entity_b": summary_b.get("latest_date"),
        },
        "comparison": metric_rows,
        "trend_summaries": {
            "entity_a": {
                "entity": entity_a,
                "latest_date": summary_a.get("latest_date"),
                "current_7d": summary_a.get("current_7d"),
                "delta_7d": summary_a.get("delta_7d"),
                "search_impact": summary_a.get("search_impact"),
                "search_nuance": summary_a.get("search_nuance"),
                "recent_anomalies": summary_a.get("recent_anomalies"),
            },
            "entity_b": {
                "entity": entity_b,
                "latest_date": summary_b.get("latest_date"),
                "current_7d": summary_b.get("current_7d"),
                "delta_7d": summary_b.get("delta_7d"),
                "search_impact": summary_b.get("search_impact"),
                "search_nuance": summary_b.get("search_nuance"),
                "recent_anomalies": summary_b.get("recent_anomalies"),
            },
        },
    }


TOOL_DISPATCH = {
    "resolve_entity": resolve_entity,
    "resolve_sector": resolve_sector,
    "screen_entities": screen_entities,
    "get_sector_baseline": get_sector_baseline,
    "aggregate_crisis_patterns": aggregate_crisis_patterns,
    "aggregate_industry_durations": aggregate_industry_durations,
    "find_storylines": find_storylines,
    "get_trend_summary": get_trend_summary,
    "get_narrative_timeline": get_narrative_timeline,
    "get_narrative_tags": get_narrative_tags,
    "get_search_feature_series": get_search_feature_series,
    "get_search_feature_items": get_search_feature_items,
    "compare_entities": compare_entities,
    "get_anomalies": get_anomalies,
    "get_evidence": get_evidence,
}


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOL_DISPATCH:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_DISPATCH[name](**arguments)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python ai_insights_api.py init-iap-desktop-auth\n"
            "  python ai_insights_api.py print-bearer-token"
        )
        return 1

    command = sys.argv[1].strip().lower()
    if command == "init-iap-desktop-auth":
        token_path = init_iap_desktop_auth()
        print(f"Saved IAP desktop OAuth credentials to {token_path}")
        return 0
    if command == "print-bearer-token":
        print(_resolve_bearer_token())
        return 0

    print(f"Unknown command: {sys.argv[1]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
