#!/usr/bin/env python3
"""Parse natural-language time windows into rolling or calendar periods."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict


EXPLICIT_QUARTER_PATTERNS = [
    re.compile(r"\bq([1-4])\s*(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\s*q([1-4])\b", re.IGNORECASE),
    re.compile(r"\bquarter\s*([1-4])\s*(20\d{2})\b", re.IGNORECASE),
]
ROLLING_DAY_PATTERN = re.compile(r"\b(?:rolling|past|last)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    start = date(year, start_month, 1)
    if quarter == 4:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, start_month + 3, 1) - timedelta(days=1)
    return start, end


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def _calendar_period(period_label: str, display_label: str, start: date, end: date) -> Dict[str, Any]:
    return {
        "mode": "calendar",
        "period_label": period_label,
        "display_label": display_label,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": (end - start).days + 1,
    }


def _rolling_period(days: int) -> Dict[str, Any]:
    return {
        "mode": "rolling",
        "period_label": f"rolling_{days}_days",
        "display_label": f"rolling {days} days ending on the latest available date",
        "start_date": None,
        "end_date": None,
        "days": days,
    }


def resolve_time_window(user_message: str, today: date | None = None, default_days: int = 90) -> Dict[str, Any]:
    text = (user_message or "").strip().casefold()
    today = today or date.today()

    for pattern in EXPLICIT_QUARTER_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        if first > 4:
            year, quarter = first, second
        else:
            quarter, year = first, second
        start, end = _quarter_bounds(year, quarter)
        return _calendar_period(
            period_label=f"q{quarter}_{year}",
            display_label=f"Q{quarter} {year}",
            start=start,
            end=end,
        )

    if "last quarter" in text or "previous quarter" in text:
        current_quarter = ((today.month - 1) // 3) + 1
        if current_quarter == 1:
            year = today.year - 1
            quarter = 4
        else:
            year = today.year
            quarter = current_quarter - 1
        start, end = _quarter_bounds(year, quarter)
        return _calendar_period(
            period_label="last_quarter",
            display_label=f"last quarter (Q{quarter} {year})",
            start=start,
            end=end,
        )

    if "this quarter" in text or "current quarter" in text or "quarter to date" in text:
        quarter = ((today.month - 1) // 3) + 1
        start, _ = _quarter_bounds(today.year, quarter)
        return _calendar_period(
            period_label="this_quarter",
            display_label=f"this quarter to date (Q{quarter} {today.year})",
            start=start,
            end=today,
        )

    if "last month" in text:
        if today.month == 1:
            year = today.year - 1
            month = 12
        else:
            year = today.year
            month = today.month - 1
        start, end = _month_bounds(year, month)
        return _calendar_period(
            period_label="last_month",
            display_label=f"last month ({start.strftime('%B %Y')})",
            start=start,
            end=end,
        )

    if "this month" in text or "month to date" in text:
        start, _ = _month_bounds(today.year, today.month)
        return _calendar_period(
            period_label="this_month",
            display_label=f"this month to date ({start.strftime('%B %Y')})",
            start=start,
            end=today,
        )

    if "last year" in text or "previous year" in text:
        start = date(today.year - 1, 1, 1)
        end = date(today.year - 1, 12, 31)
        return _calendar_period(
            period_label="last_year",
            display_label=f"last year ({today.year - 1})",
            start=start,
            end=end,
        )

    if "this year" in text or "year to date" in text:
        start = date(today.year, 1, 1)
        return _calendar_period(
            period_label="this_year",
            display_label=f"this year to date ({today.year})",
            start=start,
            end=today,
        )

    match = ROLLING_DAY_PATTERN.search(text)
    if match:
        days = min(max(int(match.group(1)), 1), 365)
        return _rolling_period(days)

    if "today" in text:
        return _rolling_period(1)
    if "this week" in text or "past week" in text or "last 7 days" in text:
        return _rolling_period(7)
    if "this month" in text or "past month" in text or "last 30 days" in text:
        return _rolling_period(30)
    if "this quarter" in text or "past quarter" in text or "last 90 days" in text:
        return _rolling_period(90)
    if "this year" in text or "last 365 days" in text:
        return _rolling_period(365)

    return _rolling_period(default_days)
