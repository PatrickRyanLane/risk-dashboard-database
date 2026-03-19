#!/usr/bin/env python3
"""Heuristic planner for vague or broad AI insights questions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ai_insights_api import (
    aggregate_crisis_patterns,
    aggregate_industry_durations,
    compare_entities,
    find_storylines,
    get_sector_baseline,
    resolve_sector,
    screen_entities,
)
from period_parser import resolve_time_window


SECTOR_PATTERNS = [
    re.compile(r"\b(?:in|within|across)\s+the\s+([a-z0-9&/\- ]+?)\s+(?:sector|industry)\b", re.IGNORECASE),
    re.compile(r"\b([a-z0-9&/\- ]+?)\s+(?:sector|industry)\b", re.IGNORECASE),
]
COMPARE_PATTERN = re.compile(
    r"\bcompare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\s+(?:over|for|in)\b|$)",
    re.IGNORECASE,
)


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def extract_sector(user_message: str) -> str | None:
    for pattern in SECTOR_PATTERNS:
        match = pattern.search(user_message or "")
        if not match:
            continue
        sector = _normalize_spaces(match.group(1))
        if sector:
            return sector
    return None


def infer_days(user_message: str, default_days: int = 30) -> int:
    text = (user_message or "").casefold()
    if "today" in text:
        return 1
    if "this week" in text or "last 7 days" in text or "past week" in text:
        return 7
    if "this month" in text or "last 30 days" in text or "past month" in text:
        return 30
    if "this quarter" in text or "last quarter" in text or "last 90 days" in text:
        return 90
    if "this year" in text or "last year" in text or "last 365 days" in text:
        return 365
    return default_days


def build_window_arguments(window: Dict[str, Any]) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "period_label": window.get("period_label"),
    }
    if window.get("mode") == "calendar":
        arguments["start_date"] = window.get("start_date")
        arguments["end_date"] = window.get("end_date")
    else:
        arguments["days"] = window.get("days")
    return arguments


def _summarize_screen_entities(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result.get("rows") or []
    top_rows = []
    for row in rows[:3]:
        top_rows.append(
            {
                "entity_name": row.get("entity_name"),
                "company": row.get("company"),
                "sector": row.get("sector"),
                "window_value": row.get("window_value"),
                "latest_value": row.get("latest_value"),
                "peak_value": row.get("peak_value"),
                "signal_days": row.get("signal_days"),
            }
        )
    return {
        "metric": result.get("metric"),
        "metric_label": result.get("metric_label"),
        "days": result.get("days"),
        "window_start": result.get("window_start"),
        "window_end": result.get("window_end"),
        "latest_available_date": result.get("latest_available_date"),
        "sector": result.get("sector"),
        "top_rows": top_rows,
    }


def _summarize_crisis_patterns(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result.get("rows") or []
    top_rows = []
    for row in rows[:5]:
        top_rows.append(
            {
                "display_tag": row.get("display_tag"),
                "group": row.get("group"),
                "avg_duration_days": row.get("avg_duration_days"),
                "median_duration_days": row.get("median_duration_days"),
                "max_duration_days": row.get("max_duration_days"),
                "brands_affected": row.get("brands_affected"),
                "ceos_affected": row.get("ceos_affected"),
                "episode_count": row.get("episode_count"),
                "sample_entities": row.get("sample_entities"),
            }
        )
    return {
        "sector": result.get("sector"),
        "days": result.get("days"),
        "window_start": result.get("window_start"),
        "window_end": result.get("window_end"),
        "latest_available_date": result.get("latest_available_date"),
        "duration_definition": result.get("duration_definition"),
        "top_rows": top_rows,
    }


def _summarize_industry_durations(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result.get("rows") or []
    top_rows = []
    for row in rows[:10]:
        top_rows.append(
            {
                "sector": row.get("sector"),
                "avg_duration_days": row.get("avg_duration_days"),
                "median_duration_days": row.get("median_duration_days"),
                "max_duration_days": row.get("max_duration_days"),
                "episode_count": row.get("episode_count"),
                "brands_affected": row.get("brands_affected"),
                "ceos_affected": row.get("ceos_affected"),
                "most_common_tags": row.get("most_common_tags"),
            }
        )
    return {
        "days": result.get("days"),
        "window_start": result.get("window_start"),
        "window_end": result.get("window_end"),
        "latest_available_date": result.get("latest_available_date"),
        "duration_definition": result.get("duration_definition"),
        "top_rows": top_rows,
    }


def _summarize_storylines(result: Dict[str, Any]) -> Dict[str, Any]:
    top_rows = []
    for row in (result.get("storylines") or [])[:5]:
        top_rows.append(
            {
                "storyline_type": row.get("storyline_type"),
                "headline": row.get("headline"),
                "angle": row.get("angle"),
                "why_interesting": row.get("why_interesting"),
                "sample_sectors": row.get("sample_sectors"),
                "sample_entities": row.get("sample_entities"),
                "supporting_metrics": row.get("supporting_metrics"),
            }
        )
    return {
        "period_label": result.get("period_label"),
        "window_mode": result.get("window_mode"),
        "window_start": result.get("window_start"),
        "window_end": result.get("window_end"),
        "latest_available_date": result.get("latest_available_date"),
        "top_storylines": top_rows,
    }


def plan_query(user_message: str) -> Dict[str, Any] | None:
    text = (user_message or "").strip()
    if not text:
        return None

    lowered = text.casefold()
    sector = extract_sector(text)
    window = resolve_time_window(text, default_days=90)

    compare_match = COMPARE_PATTERN.search(text)
    if compare_match:
        entity_a_name = _normalize_spaces(compare_match.group(1))
        entity_b_name = _normalize_spaces(compare_match.group(2))
        if entity_a_name and entity_b_name:
            return {
                "plan_type": "entity_comparison",
                "assumption": (
                    f"Interpret the question as a direct comparison between {entity_a_name} and "
                    f"{entity_b_name} over the latest {infer_days(text, default_days=30)}-day window."
                ),
                "tool_calls": [
                    {
                        "name": "compare_entities",
                        "arguments": {
                            "entity": "brand",
                            "entity_a_name": entity_a_name,
                            "entity_b_name": entity_b_name,
                            "days": infer_days(text, default_days=30),
                            "weeks": 8,
                        },
                    }
                ],
            }

    if "average crisis duration" in lowered and ("by industry" in lowered or "by sector" in lowered or "broken down by" in lowered):
        return {
            "plan_type": "industry_duration_breakdown",
            "assumption": (
                f"Interpret the question as a cross-industry comparison of average crisis duration over "
                f"{window.get('display_label')}."
            ),
            "tool_calls": [
                {
                    "name": "aggregate_industry_durations",
                    "arguments": {
                        "entity": "brand",
                        **build_window_arguments(window),
                        "limit": 25,
                    },
                }
            ],
        }

    if (
        "thought leadership" in lowered
        or "story line" in lowered
        or "storyline" in lowered
        or "article idea" in lowered
        or "article on reputational risk" in lowered
        or "interesting themes" in lowered
    ):
        arguments = {
            "entity": "brand",
            **build_window_arguments(window),
            "limit": 3,
        }
        if sector:
            arguments["sector"] = sector
        return {
            "plan_type": "storyline_scan",
            "assumption": (
                f"Interpret the question as an editorial storyline scan over {window.get('display_label')}, "
                "returning the strongest thought-leadership angles supported by search-visible crisis data."
            ),
            "tool_calls": [
                {
                    "name": "find_storylines",
                    "arguments": arguments,
                }
            ],
        }

    if sector and (
        "what brand" in lowered
        or "which brand" in lowered
        or "which brands" in lowered
        or "affected" in lowered
    ):
        days = infer_days(text, default_days=7)
        return {
            "plan_type": "sector_brand_candidates",
            "assumption": (
                f"Interpret the question as asking for the strongest current brand candidates in the {sector} "
                f"sector, ranked by negative Top Stories over the latest {days}-day window."
            ),
            "tool_calls": [
                {
                    "name": "screen_entities",
                    "arguments": {
                        "entity": "brand",
                        "metric": "top_stories_negative_count",
                        "sector": sector,
                        "days": days,
                        "limit": 3,
                        "min_value": 1,
                    },
                }
            ],
        }

    if sector and (
        "baseline" in lowered
        or "normal for" in lowered
        or "worse than peers" in lowered
        or "vs peers" in lowered
        or "versus peers" in lowered
    ):
        return {
            "plan_type": "sector_baseline",
            "assumption": (
                f"Interpret the question as a sector-baseline comparison for {sector} over the latest "
                f"{infer_days(text, default_days=30)}-day window."
            ),
            "tool_calls": [
                {
                    "name": "resolve_sector",
                    "arguments": {
                        "sector_name": sector,
                        "limit": 3,
                    },
                },
                {
                    "name": "get_sector_baseline",
                    "arguments": {
                        "entity": "brand",
                        "sector": sector,
                        "metric": "top_stories_negative_count",
                        "days": infer_days(text, default_days=30),
                        "limit": 5,
                    },
                },
            ],
        }

    if "most negative top stories" in lowered or "negative top stories today" in lowered:
        days = infer_days(text, default_days=1)
        return {
            "plan_type": "top_stories_ranking",
            "assumption": (
                f"Interpret the question as a leaderboard of brands ranked by negative Top Stories over the latest "
                f"{days}-day window."
            ),
            "tool_calls": [
                {
                    "name": "screen_entities",
                    "arguments": {
                        "entity": "brand",
                        "metric": "top_stories_negative_count",
                        "days": days,
                        "limit": 10,
                        "min_value": 1,
                    },
                }
            ],
        }

    if sector and ("most common type of crisis" in lowered or "most common crisis" in lowered):
        days = infer_days(text, default_days=90)
        return {
            "plan_type": "sector_crisis_pattern",
            "assumption": (
                f"Interpret the question as a sector-level crisis-pattern question for {sector} over the latest "
                f"{days}-day window."
            ),
            "tool_calls": [
                {
                    "name": "aggregate_crisis_patterns",
                    "arguments": {
                        "entity": "brand",
                        "sector": sector,
                        **build_window_arguments(window if window else {"days": days}),
                        "limit": 5,
                    },
                }
            ],
        }

    return None


def execute_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    executed = {"plan_type": plan.get("plan_type"), "assumption": plan.get("assumption"), "results": []}
    for call in plan.get("tool_calls") or []:
        name = call.get("name")
        arguments = call.get("arguments") or {}
        if name == "screen_entities":
            result = screen_entities(**arguments)
            summary = _summarize_screen_entities(result)
        elif name == "resolve_sector":
            result = resolve_sector(**arguments)
            summary = {
                "resolution_status": result.get("resolution_status"),
                "resolved": result.get("resolved"),
                "suggestions": (result.get("suggestions") or [])[:3],
            }
        elif name == "get_sector_baseline":
            result = get_sector_baseline(**arguments)
            summary = {
                "resolved_sector": result.get("resolved_sector"),
                "metric": result.get("metric"),
                "metric_label": result.get("metric_label"),
                "window_start": result.get("window_start"),
                "window_end": result.get("window_end"),
                "sector_summary": result.get("sector_summary"),
                "peer_entity": result.get("peer_entity"),
                "top_rows": (result.get("rows") or [])[:3],
            }
        elif name == "aggregate_crisis_patterns":
            result = aggregate_crisis_patterns(**arguments)
            summary = _summarize_crisis_patterns(result)
        elif name == "aggregate_industry_durations":
            result = aggregate_industry_durations(**arguments)
            summary = _summarize_industry_durations(result)
        elif name == "find_storylines":
            result = find_storylines(**arguments)
            summary = _summarize_storylines(result)
        elif name == "compare_entities":
            result = compare_entities(**arguments)
            summary = {
                "entity_a": result.get("entity_a"),
                "entity_b": result.get("entity_b"),
                "latest_dates": result.get("latest_dates"),
                "comparison": (result.get("comparison") or [])[:6],
            }
        else:
            continue
        executed["results"].append({"tool": name, "arguments": arguments, "summary": summary})
    return executed


def build_planner_context(user_message: str) -> str:
    plan = plan_query(user_message)
    if not plan:
        return ""
    try:
        executed = execute_plan(plan)
    except Exception as exc:
        return f"Planner note: heuristic prefetch was attempted but failed: {exc}"

    return (
        "Planner prefetch context:\n"
        f"{json.dumps(executed, indent=2, sort_keys=True)}\n\n"
        "Use this context if helpful, but you may still call tools for verification or follow-up detail."
    )
