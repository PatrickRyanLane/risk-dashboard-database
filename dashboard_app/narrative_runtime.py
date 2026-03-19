from __future__ import annotations

import os
import re

NARRATIVE_RULE_VERSION = "v3"
NARRATIVE_MIN_NEG_TOP_STORIES = max(1, int(os.environ.get("NARRATIVE_MIN_NEG_TOP_STORIES", "2") or 2))
NARRATIVE_CRISIS_TAGS = [
    "Workforce Reductions",
    "Accidents & Disasters",
    "Data Breaches",
    "Activist Investor Interest",
    "Legal & Regulatory",
    "Unforced Errors",
    "Labor Disputes",
    "CEO Departures (firings, resignations)",
    "Fraud",
    "Other",
]
NARRATIVE_NON_CRISIS_TAGS = [
    "Rebranding",
    "Mergers and acquisitions",
    "Planned Executive Turnover",
]
NARRATIVE_OTHER_MIN_SUPPORT = 2
NARRATIVE_TAG_GROUPS = {
    **{tag: "crisis" for tag in NARRATIVE_CRISIS_TAGS},
    **{tag: "non_crisis" for tag in NARRATIVE_NON_CRISIS_TAGS},
}
NARRATIVE_TAG_ORDER = {
    tag: idx
    for idx, tag in enumerate(NARRATIVE_CRISIS_TAGS + NARRATIVE_NON_CRISIS_TAGS)
}

LAYOFF_TERMS = [
    r"\blayoff(s)?\b",
    r"\blays?\s+off\b",
    r"\blaid\s+off\b",
]
WORKFORCE_REDUCTION_TERMS = [
    *LAYOFF_TERMS,
    r"\bjob cuts?\b",
    r"\bworkforce reduction(?:s)?\b",
    r"\bworkforce cuts?\b",
    r"\bheadcount reduction(?:s)?\b",
    r"\bstaff reduction(?:s)?\b",
    r"\brestructuring plan\b",
    r"\bdownsiz(?:e|ing)\b",
    r"\bright[- ]siz(?:e|ing)\b",
    r"\bredundanc(?:y|ies)\b",
    r"\bfurlough(?:s|ed|ing)?\b",
    r"\bposition eliminations?\b",
]
WORKFORCE_REDUCTION_RE = re.compile("|".join(WORKFORCE_REDUCTION_TERMS), flags=re.IGNORECASE)

LOW_PRIORITY_CRISIS_BLOCKER_RE = re.compile(
    r"\b(data breach(?:es)?|cyber(?:attack|attacks|breach|breaches)|ransomware|"
    r"hack(?:ed|s|ing)?|fraud|embezzl(?:e|ement)|briber(?:y|ies)|corruption|"
    r"indict(?:ed|ment|ments)?|guilty|convicted|subpoena(?:s)?|charge(?:d|s)?|"
    r"chapter\s+11|bankrupt(?:cy|cies)|default(?:s|ed|ing)?|insolven(?:t|cy)|"
    r"delinquen(?:t|cy)|miss(?:es|ed|ing)\s+payments?|fatal(?:ity|ities)|"
    r"death(?:s)?|killed|injur(?:y|ies)|explosion(?:s)?|fire(?:s)?|crash(?:es|ed)?|"
    r"collapse(?:d|s)?|contamination|chemical spill|oil spill|gas leak|"
    r"toxic release|hazmat|recall(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_LEGAL_ENFORCEMENT_RE = re.compile(
    r"\b(class[- ]action|lawsuit(?:s)?|legal action|attorney general|sec\b|doj\b|"
    r"ftc\b|cfpb\b|eeoc\b|nlrb\b|investigat(?:e|es|ed|ing|ion)|probe(?:s|d)?|"
    r"unlawful(?:ly)?|illegal(?:ly)?|discrimination|retaliation)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_TARIFF_CONTEXT_RE = re.compile(
    r"\b(tariff(?:s)?|trade dispute(?:s)?|trade war|trade polic(?:y|ies)|"
    r"import dut(?:y|ies)|customs dut(?:y|ies)|trade barrier(?:s)?|"
    r"import lev(?:y|ies))\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_TARIFF_LEGAL_RE = re.compile(
    r"\b(lawsuit(?:s)?|legal action|sue(?:s|d|ing)?|court challenge|"
    r"complaint(?:s)?|petition(?:s|ed|ing)?|appeal(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_TARIFF_BLOCKER_RE = re.compile(
    r"\b(class[- ]action|attorney general|sec\b|doj\b|ftc\b|cfpb\b|epa\b|fda\b|"
    r"osha\b|eeoc\b|nlrb\b|investigat(?:e|es|ed|ing|ion)|probe(?:s|d)?|"
    r"misconduct|antitrust|sanction(?:s|ed)?|penalt(?:y|ies))\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_DELAY_ACTION_RE = re.compile(
    r"\b(delay(?:s|ed|ing)?|postpon(?:e|es|ed|ing|ement)|"
    r"push(?:es|ed|ing)?\s+back|slipp(?:ed|ing|age))\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_DELAY_CONTEXT_RE = re.compile(
    r"\b(ai chip(?:s)?|chip(?:s)?|semiconductor(?:s)?|robotaxi|launch|rollout|"
    r"release|production|product roadmap|timeline|platform|model(?:s)?|program)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_DELAY_BLOCKER_RE = re.compile(
    r"\b(recall(?:s|ed|ing)?|safety|fatal(?:ity|ities)|death(?:s)?|injur(?:y|ies)|"
    r"fda\b|osha\b)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_FEE_CONTEXT_RE = re.compile(
    r"\b(commission(?: fee)?s?|app store (?:fee|fees|commission)|take rate|"
    r"developer fee(?:s)?|marketplace fee(?:s)?|platform fee(?:s)?)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_FEE_ACTION_RE = re.compile(
    r"\b(reduc(?:e|es|ed|ing)|cut(?:s|ting)?|lower(?:s|ed|ing)|"
    r"slash(?:es|ed|ing)?|trim(?:s|med|ming))\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_DEBT_CONTEXT_RE = re.compile(
    r"\b(debt|notes?|bonds?|maturit(?:y|ies)|credit facility|term loan|"
    r"capital structure|liabilit(?:y|ies) management|debt exchange|exchange offer)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_DEBT_ACTION_RE = re.compile(
    r"\b(refinanc(?:e|es|ed|ing)|exchange(?:s|d|ing)?|extend(?:s|ed|ing)?|"
    r"reduce(?:s|d|ing)?|repay(?:s|ment|ing)?|retir(?:e|es|ed|ing)|"
    r"issu(?:e|es|ed|ing)|offer(?:s|ed|ing)?|amend(?:s|ed|ing)?|swap(?:s|ped|ping)?)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_DEBT_BLOCKER_RE = re.compile(
    r"\b(default(?:s|ed|ing)?|distress(?:ed)?|delinquen(?:t|cy)|insolven(?:t|cy)|"
    r"bankrupt(?:cy|cies)|chapter\s+11|miss(?:es|ed|ing)\s+payments?|"
    r"restructuring support agreement)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_STORE_CONTEXT_RE = re.compile(
    r"\b(store(?:s)?|location(?:s)?|restaurant(?:s)?|branch(?:es)?|outlet(?:s)?|"
    r"shop(?:s)?|office(?:s)?|club(?:s)?|pharmacies|pharmacy|retail locations?)\b",
    flags=re.IGNORECASE,
)
LOW_PRIORITY_STORE_ACTION_RE = re.compile(
    r"\bclos(?:e|es|ed|ing|ure|ures)\b",
    flags=re.IGNORECASE,
)

NARRATIVE_REBRANDING_RE = re.compile(
    r"\b(rebrand(?:ing|ed|s)?|brand refresh|new logo|renam(?:e|ed|ing)|new brand identity|brand overhaul)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_MNA_RE = re.compile(
    r"\b(merger(?:s)?|acquisition(?:s)?|acquire(?:d|s|ing)?|buyout|takeover|merge(?:s|d|r|ing)?|spinoff|spin-off)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_PLANNED_EXEC_RE = re.compile(
    r"\b(retire(?:s|d|ment|ing)?|succession plan(?:ning)?|planned succession|planned transition|"
    r"step(?:ping)? down|to step down|will step down|named successor|successor)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_PLANNED_EXEC_EXCLUDE_RE = re.compile(
    r"\b(fired|firing|ousted|forced out|amid|scandal|probe|investigat(?:e|es|ed|ing|ion)|"
    r"lawsuit|indict(?:ed|ment)?|charged|fraud|misconduct)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_WORKFORCE_RE = re.compile("|".join(WORKFORCE_REDUCTION_TERMS), flags=re.IGNORECASE)
NARRATIVE_ACCIDENT_RE = re.compile(
    r"\b(accident(?:s)?|explosion(?:s)?|fire(?:s)?|disaster(?:s)?|fatal(?:ity|ities)|"
    r"injur(?:y|ies)|crash(?:es|ed)?|derailment|collapse(?:d|s)?|plant incident|"
    r"chemical spill|oil spill|gas leak|toxic release|hazmat|contamination|"
    r"industrial incident|site shutdown|evacuat(?:e|ed|ion))\b",
    flags=re.IGNORECASE,
)
NARRATIVE_DATA_BREACH_RE = re.compile(
    r"\b(data breach(?:es)?|cyber(?:attack|attacks)|ransomware|hack(?:ed|s|ing)?|"
    r"security breach(?:es)?|data leak(?:s|ed|ing)?|expos(?:e|ed|ure|ing)|"
    r"unauthori[sz]ed access|stolen data|compromised (?:accounts?|systems?|credentials)|"
    r"malware|phishing|ddos|privacy incident|zero[- ]day|vulnerabilit(?:y|ies))\b",
    flags=re.IGNORECASE,
)
NARRATIVE_ACTIVIST_INVESTOR_RE = re.compile(
    r"\b(activist investor(?:s)?|activist hedge fund(?:s)?|proxy (?:fight|battle|contest)|"
    r"dissident shareholder(?:s)?|board seat(?:s)?|board representation|"
    r"nominat(?:e|es|ed|ing) (?:director|directors)|shareholder campaign|campaign letter|"
    r"schedule 13d|13d filing|push(?:ing)? for (?:a sale|breakup|spin-?off|board changes?))\b",
    flags=re.IGNORECASE,
)
NARRATIVE_LEGAL_RE = re.compile(
    r"\b(attorney general|lawsuit(?:s)?|legal action|regulator(?:y)?|regulatory|"
    r"investigat(?:e|es|ed|ing|ion)|probe(?:s|d)?|settle(?:ment|s|d|ing)?|fine(?:d|s|ing)?|"
    r"charged|indict(?:ed|ment)?|class[- ]action|subpoena(?:s)?|consent (?:order|decree)|"
    r"injunction|violat(?:ion|ions)|non[- ]compliance|sec\b|doj\b|ftc\b|cfpb\b|"
    r"epa\b|fda\b|osha\b|eeoc\b|nlrb\b|cpsc\b)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_UNFORCED_RE = re.compile(
    r"\b(backlash|boycott(?:s|ed|ing)?|tone[- ]deaf|ad campaign|advertising campaign|"
    r"public apology|apolog(?:y|ies|ize|ized|izing)|controversial comment(?:s)?|"
    r"executive comment(?:s)?|social media post|pr disaster|gaffe|offensive (?:remark|remarks|post)|"
    r"insensitive (?:remark|remarks|post)|walked back|deleted post|viral backlash)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_LABOR_RE = re.compile(
    r"\b(strike(?:s|d|ing)?|walkout(?:s)?|labor dispute(?:s)?|union dispute(?:s)?|"
    r"picket(?:ing)?|collective bargaining|contract talks?|lockout(?:s)?|work stoppage(?:s)?|"
    r"unionization drive|organizing drive|unfair labor practice(?:s)?|nlrb charge(?:s)?|contract impasse)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_CEO_DEPART_RE = re.compile(
    r"\b(ceo\s+(?:resign(?:s|ed|ing|ation)?|step(?:s|ped)? down|depart(?:s|ed|ure)|"
    r"fired|ouste?d|removed)|chief executive\s+(?:resign(?:s|ed|ing|ation)?|step(?:s|ped)? down|"
    r"fired|ouste?d|removed)|resign(?:s|ed|ing|ation)? as ceo|ouste?d ceo|fired ceo)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_CEO_DEPART_EXCLUDE_RE = re.compile(
    r"\b(retire(?:s|d|ment|ing)?|succession plan(?:ning)?|planned succession|planned transition|"
    r"named successor|interim ceo)\b",
    flags=re.IGNORECASE,
)
NARRATIVE_FRAUD_RE = re.compile(
    r"\b(fraud|embezzl(?:e|ed|ing|ement)|briber(?:y|ies)|corruption|ponzi|accounting fraud|"
    r"falsif(?:y|ied|ication)|misappropriation|insider trading|securities fraud|wire fraud|"
    r"mail fraud|money laundering|kickback(?:s)?|tax evasion|false claims|bid rigging)\b",
    flags=re.IGNORECASE,
)


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        txt = (item or "").strip()
        if not txt:
            continue
        norm = txt.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(txt)
    return out


def _empty_narrative_result() -> dict:
    return {
        "primary_tag": None,
        "primary_group": None,
        "tags": [],
        "is_crisis": None,
        "rule_version": NARRATIVE_RULE_VERSION,
    }


def narrative_tag_gate_met(
    negative_top_stories_count: int,
    *,
    min_negative_top_stories: int = NARRATIVE_MIN_NEG_TOP_STORIES,
) -> bool:
    try:
        count = int(negative_top_stories_count or 0)
    except Exception:
        count = 0
    try:
        minimum = int(min_negative_top_stories or NARRATIVE_MIN_NEG_TOP_STORIES)
    except Exception:
        minimum = NARRATIVE_MIN_NEG_TOP_STORIES
    return count >= max(1, minimum)


def _low_priority_haystack(
    title: str = "",
    snippet: str = "",
    url: str = "",
    source: str = "",
) -> str:
    return " ".join(part for part in [title or "", snippet or "", source or "", url or ""] if part).strip()


def _matches_low_priority_tariff_story(hay: str) -> bool:
    return bool(
        LOW_PRIORITY_TARIFF_CONTEXT_RE.search(hay)
        and LOW_PRIORITY_TARIFF_LEGAL_RE.search(hay)
        and not LOW_PRIORITY_TARIFF_BLOCKER_RE.search(hay)
        and not LOW_PRIORITY_CRISIS_BLOCKER_RE.search(hay)
    )


def _matches_low_priority_workforce_story(hay: str) -> bool:
    return bool(
        WORKFORCE_REDUCTION_RE.search(hay)
        and not LOW_PRIORITY_CRISIS_BLOCKER_RE.search(hay)
        and not LOW_PRIORITY_LEGAL_ENFORCEMENT_RE.search(hay)
    )


def _matches_low_priority_delay_story(hay: str) -> bool:
    return bool(
        LOW_PRIORITY_DELAY_ACTION_RE.search(hay)
        and LOW_PRIORITY_DELAY_CONTEXT_RE.search(hay)
        and not LOW_PRIORITY_DELAY_BLOCKER_RE.search(hay)
        and not LOW_PRIORITY_CRISIS_BLOCKER_RE.search(hay)
    )


def _matches_low_priority_fee_story(hay: str) -> bool:
    return bool(
        LOW_PRIORITY_FEE_ACTION_RE.search(hay)
        and LOW_PRIORITY_FEE_CONTEXT_RE.search(hay)
        and not LOW_PRIORITY_CRISIS_BLOCKER_RE.search(hay)
    )


def _matches_low_priority_debt_story(hay: str) -> bool:
    return bool(
        LOW_PRIORITY_DEBT_ACTION_RE.search(hay)
        and LOW_PRIORITY_DEBT_CONTEXT_RE.search(hay)
        and not LOW_PRIORITY_DEBT_BLOCKER_RE.search(hay)
        and not LOW_PRIORITY_CRISIS_BLOCKER_RE.search(hay)
    )


def _matches_low_priority_store_closure_story(hay: str) -> bool:
    return bool(
        LOW_PRIORITY_STORE_ACTION_RE.search(hay)
        and LOW_PRIORITY_STORE_CONTEXT_RE.search(hay)
        and not LOW_PRIORITY_CRISIS_BLOCKER_RE.search(hay)
        and not LOW_PRIORITY_LEGAL_ENFORCEMENT_RE.search(hay)
    )


def is_low_priority_business_story(
    title: str,
    snippet: str = "",
    url: str = "",
    source: str = "",
) -> bool:
    hay = _low_priority_haystack(title, snippet, url, source)
    if not hay:
        return False
    return any((
        _matches_low_priority_tariff_story(hay),
        _matches_low_priority_workforce_story(hay),
        _matches_low_priority_delay_story(hay),
        _matches_low_priority_fee_story(hay),
        _matches_low_priority_debt_story(hay),
        _matches_low_priority_store_closure_story(hay),
    ))


def narrative_tag_group(tag: str | None) -> str | None:
    if not tag:
        return None
    return NARRATIVE_TAG_GROUPS.get(tag)


def _narrative_tag_sort_key(tag: str | None) -> tuple[int, int, str]:
    group = narrative_tag_group(tag)
    if group == "crisis":
        group_rank = 0
    elif group == "non_crisis":
        group_rank = 1
    else:
        group_rank = 2
    return (
        group_rank,
        NARRATIVE_TAG_ORDER.get(tag or "", 999),
        str(tag or "").casefold(),
    )


def classify_narrative_tags(
    title: str,
    snippet: str = "",
    *,
    url: str = "",
    source: str = "",
    sentiment: str | None = None,
    finance_routine: bool | None = None,
    allow_other_fallback: bool = True,
) -> dict:
    sentiment_l = (sentiment or "").strip().lower()
    if sentiment_l and sentiment_l != "negative":
        return _empty_narrative_result()
    if finance_routine is True:
        return _empty_narrative_result()

    hay = " ".join([title or "", snippet or "", source or "", url or ""]).strip()
    if not hay:
        return _empty_narrative_result()
    if is_low_priority_business_story(title, snippet, url=url, source=source):
        return _empty_narrative_result()

    crisis_tags: list[str] = []
    non_crisis_tags: list[str] = []

    if NARRATIVE_REBRANDING_RE.search(hay):
        non_crisis_tags.append("Rebranding")
    if NARRATIVE_MNA_RE.search(hay):
        non_crisis_tags.append("Mergers and acquisitions")
    if NARRATIVE_PLANNED_EXEC_RE.search(hay) and not NARRATIVE_PLANNED_EXEC_EXCLUDE_RE.search(hay):
        non_crisis_tags.append("Planned Executive Turnover")

    if NARRATIVE_FRAUD_RE.search(hay):
        crisis_tags.append("Fraud")
    if NARRATIVE_DATA_BREACH_RE.search(hay):
        crisis_tags.append("Data Breaches")
    if NARRATIVE_CEO_DEPART_RE.search(hay) and not NARRATIVE_CEO_DEPART_EXCLUDE_RE.search(hay):
        crisis_tags.append("CEO Departures (firings, resignations)")
    if NARRATIVE_WORKFORCE_RE.search(hay):
        crisis_tags.append("Workforce Reductions")
    if NARRATIVE_LABOR_RE.search(hay):
        crisis_tags.append("Labor Disputes")
    if NARRATIVE_ACCIDENT_RE.search(hay):
        crisis_tags.append("Accidents & Disasters")
    if NARRATIVE_ACTIVIST_INVESTOR_RE.search(hay):
        crisis_tags.append("Activist Investor Interest")
    if NARRATIVE_UNFORCED_RE.search(hay):
        crisis_tags.append("Unforced Errors")
    if NARRATIVE_LEGAL_RE.search(hay):
        crisis_tags.append("Legal & Regulatory")

    crisis_tags = _dedupe_preserve(crisis_tags)
    non_crisis_tags = _dedupe_preserve(non_crisis_tags)

    if crisis_tags:
        tags = _dedupe_preserve(crisis_tags + non_crisis_tags)
        return {
            "primary_tag": crisis_tags[0],
            "primary_group": "crisis",
            "tags": tags,
            "is_crisis": True,
            "rule_version": NARRATIVE_RULE_VERSION,
        }
    if non_crisis_tags:
        return {
            "primary_tag": non_crisis_tags[0],
            "primary_group": "non_crisis",
            "tags": non_crisis_tags,
            "is_crisis": False,
            "rule_version": NARRATIVE_RULE_VERSION,
        }

    if not allow_other_fallback:
        return _empty_narrative_result()

    return {
        "primary_tag": "Other",
        "primary_group": "crisis",
        "tags": ["Other"],
        "is_crisis": True,
        "rule_version": NARRATIVE_RULE_VERSION,
    }


def rollup_entity_day_narrative(
    items: list[dict],
    *,
    min_negative_top_stories: int = NARRATIVE_MIN_NEG_TOP_STORIES,
    other_min_support: int = NARRATIVE_OTHER_MIN_SUPPORT,
) -> dict:
    candidate_indexes: list[int] = []
    item_results: list[dict] = []
    tag_counts: dict[str, int] = {}
    unmatched_indexes: list[int] = []

    for item in items:
        sentiment_l = str(item.get("sentiment") or item.get("sentiment_label") or "").strip().lower()
        finance_routine = bool(item.get("finance_routine"))
        if sentiment_l != "negative" or finance_routine:
            item_results.append(_empty_narrative_result())
            continue

        candidate_indexes.append(len(item_results))
        result = classify_narrative_tags(
            item.get("title", "") or "",
            item.get("snippet", "") or "",
            url=item.get("url", "") or "",
            source=item.get("source", "") or "",
            sentiment=sentiment_l,
            finance_routine=finance_routine,
            allow_other_fallback=False,
        )
        primary_tag = result.get("primary_tag")
        if primary_tag:
            tag_counts[primary_tag] = tag_counts.get(primary_tag, 0) + 1
        else:
            unmatched_indexes.append(len(item_results))
        item_results.append(result)

    if not narrative_tag_gate_met(
        len(candidate_indexes),
        min_negative_top_stories=min_negative_top_stories,
    ):
        for idx in candidate_indexes:
            item_results[idx] = _empty_narrative_result()
        return {
            "gate_met": False,
            "negative_item_count": len(candidate_indexes),
            "primary_tag": None,
            "primary_group": None,
            "tags": [],
            "is_crisis": None,
            "rule_version": NARRATIVE_RULE_VERSION,
            "supporting_negative_items": 0,
            "tagged_item_count": 0,
            "unmatched_negative_items": len(candidate_indexes),
            "tag_counts": {},
            "item_results": item_results,
        }

    primary_tag = None
    primary_group = None
    is_crisis = None
    rollup_tags: list[str] = []
    supporting_negative_items = 0

    if tag_counts:
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda item: (-item[1],) + _narrative_tag_sort_key(item[0]),
        )
        primary_tag = sorted_tags[0][0]
        primary_group = narrative_tag_group(primary_tag)
        is_crisis = primary_group == "crisis"
        rollup_tags = [tag for tag, _ in sorted_tags]
        supporting_negative_items = int(sorted_tags[0][1] or 0)
    elif len(unmatched_indexes) >= max(1, int(other_min_support or NARRATIVE_OTHER_MIN_SUPPORT)):
        primary_tag = "Other"
        primary_group = "crisis"
        is_crisis = True
        rollup_tags = ["Other"]
        supporting_negative_items = len(unmatched_indexes)
        for idx in unmatched_indexes:
            item_results[idx] = {
                "primary_tag": "Other",
                "primary_group": "crisis",
                "tags": ["Other"],
                "is_crisis": True,
                "rule_version": NARRATIVE_RULE_VERSION,
            }

    return {
        "gate_met": True,
        "negative_item_count": len(candidate_indexes),
        "primary_tag": primary_tag,
        "primary_group": primary_group,
        "tags": rollup_tags,
        "is_crisis": is_crisis,
        "rule_version": NARRATIVE_RULE_VERSION,
        "supporting_negative_items": supporting_negative_items,
        "tagged_item_count": sum(1 for result in item_results if result.get("primary_tag")),
        "unmatched_negative_items": len(unmatched_indexes),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1],) + _narrative_tag_sort_key(item[0]))),
        "item_results": item_results,
    }


def rollup_crisis_event_items(
    items: list[dict],
    *,
    other_min_support: int = NARRATIVE_OTHER_MIN_SUPPORT,
) -> dict:
    candidate_indexes: list[int] = []
    item_results: list[dict] = []
    tag_counts: dict[str, int] = {}
    unmatched_indexes: list[int] = []

    for item in items:
        sentiment_l = str(item.get("sentiment") or item.get("sentiment_label") or "").strip().lower()
        finance_routine = bool(item.get("finance_routine"))
        if sentiment_l != "negative" or finance_routine:
            item_results.append(_empty_narrative_result())
            continue

        candidate_indexes.append(len(item_results))
        result = classify_narrative_tags(
            item.get("title", "") or "",
            item.get("snippet", "") or "",
            url=item.get("url", "") or "",
            source=item.get("source", "") or "",
            sentiment=sentiment_l,
            finance_routine=finance_routine,
            allow_other_fallback=False,
        )
        primary_tag = result.get("primary_tag")
        if primary_tag:
            tag_counts[primary_tag] = tag_counts.get(primary_tag, 0) + 1
        else:
            unmatched_indexes.append(len(item_results))
        item_results.append(result)

    primary_tag = None
    primary_group = None
    is_crisis = None
    rollup_tags: list[str] = []
    supporting_negative_items = 0

    if tag_counts:
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda item: (-item[1],) + _narrative_tag_sort_key(item[0]),
        )
        primary_tag = sorted_tags[0][0]
        primary_group = narrative_tag_group(primary_tag)
        is_crisis = primary_group == "crisis"
        rollup_tags = [tag for tag, _ in sorted_tags]
        supporting_negative_items = int(sorted_tags[0][1] or 0)
    elif len(unmatched_indexes) >= max(1, int(other_min_support or NARRATIVE_OTHER_MIN_SUPPORT)):
        primary_tag = "Other"
        primary_group = "crisis"
        is_crisis = True
        rollup_tags = ["Other"]
        supporting_negative_items = len(unmatched_indexes)
        for idx in unmatched_indexes:
            item_results[idx] = {
                "primary_tag": "Other",
                "primary_group": "crisis",
                "tags": ["Other"],
                "is_crisis": True,
                "rule_version": NARRATIVE_RULE_VERSION,
            }

    return {
        "negative_item_count": len(candidate_indexes),
        "primary_tag": primary_tag,
        "primary_group": primary_group,
        "tags": rollup_tags,
        "is_crisis": is_crisis,
        "rule_version": NARRATIVE_RULE_VERSION,
        "supporting_negative_items": supporting_negative_items,
        "tagged_item_count": sum(1 for result in item_results if result.get("primary_tag")),
        "unmatched_negative_items": len(unmatched_indexes),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1],) + _narrative_tag_sort_key(item[0]))),
        "item_results": item_results,
    }
