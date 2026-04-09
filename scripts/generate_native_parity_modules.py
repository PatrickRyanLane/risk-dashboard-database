#!/usr/bin/env python3
"""
Generate native parity tab modules from standalone dashboard HTML files.

Each generated module exports:
- `prefetchParityTab(...)`
- `mountParityTab(...)`

The runtime is handled by `dashboard_app/static/app/parity-runtime.js`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "dashboard_app" / "static"
OUT_DIR = STATIC_DIR / "app" / "parity-modules"

VIEWS = ("internal", "external")
TAB_FILES = (
    "brand-dashboard.html",
    "ceo-dashboard.html",
    "sectors.html",
    "crises.html",
)

SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)
STYLE_RE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.IGNORECASE | re.DOTALL)
BODY_RE = re.compile(r"<body\b[^>]*>(.*?)</body>", re.IGNORECASE | re.DOTALL)
SRC_RE = re.compile(r"""\bsrc\s*=\s*(['"])(.*?)\1""", re.IGNORECASE | re.DOTALL)


def extract_page_definition(html: str) -> dict:
    body_match = BODY_RE.search(html)
    body_html = body_match.group(1) if body_match else ""

    external_scripts = []
    inline_scripts = []
    for attrs, inline in SCRIPT_RE.findall(html):
        src_match = SRC_RE.search(attrs or "")
        if src_match:
            external_scripts.append(src_match.group(2).strip())
        else:
            inline_scripts.append((inline or "").strip())

    markup = SCRIPT_RE.sub("", body_html).strip()
    styles = "\n\n".join((style or "").strip() for style in STYLE_RE.findall(html) if (style or "").strip())
    inline_script = "\n\n".join(script for script in inline_scripts if script).strip()

    return {
        "styles": styles,
        "markup": markup,
        "externalScripts": external_scripts,
        "inlineScript": inline_script,
    }


def indent_code(code: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    if not code:
        return f"{prefix}// No inline script content."
    return "\n".join(f"{prefix}{line}" if line else "" for line in code.splitlines())


def build_inline_prelude(filename: str, inline_script: str) -> str:
    lines = []
    has_feature_order_decl = bool(
        re.search(r"\b(?:const|let|var)\s+FEATURE_ORDER_SENTIMENT\b", inline_script)
    )
    has_feature_labels_decl = bool(
        re.search(r"\b(?:const|let|var)\s+FEATURE_LABELS\b", inline_script)
    )
    if filename == "sectors.html" and (not has_feature_order_decl or not has_feature_labels_decl):
        lines.extend(
            [
                "const FEATURE_ORDER_SENTIMENT = ['organic','aio_citations','paa_items','videos_items','perspectives_items','top_stories_items'];",
                "const FEATURE_LABELS = {",
                "  organic: 'Organic',",
                "  aio_citations: 'AIO citations',",
                "  paa_items: 'PAA',",
                "  videos_items: 'Videos',",
                "  perspectives_items: 'Perspectives',",
                "  top_stories_items: 'Top stories',",
                "};",
            ]
        )
    return "\n".join(lines).strip()


def render_module(page: dict, filename: str) -> str:
    page_object = {
        "styles": page["styles"],
        "markup": page["markup"],
        "externalScripts": page["externalScripts"],
    }
    prelude = build_inline_prelude(filename, page["inlineScript"])
    composed_inline = (
        f"{prelude}\n\n{page['inlineScript']}".strip()
        if prelude
        else page["inlineScript"]
    )
    script_code = indent_code(composed_inline, spaces=2)
    return f"""import {{ mountParityPage, prefetchParityPage }} from '../../parity-runtime.js';

const page = {json.dumps(page_object, ensure_ascii=True, indent=2)};

function runInlineScript(scope) {{
  const {{
    window,
    self,
    top,
    parent,
    document,
    history,
    location,
    globalThis,
    fetch,
    localStorage,
    sessionStorage,
    navigator,
    requestAnimationFrame,
    cancelAnimationFrame,
    requestIdleCallback,
    cancelIdleCallback,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    performance,
    console,
    URL,
    URLSearchParams,
    Chart,
    Papa,
    ResizeObserver,
    MutationObserver,
    Event,
    CustomEvent,
    Node,
    HTMLElement,
    HTMLCanvasElement,
    CSS,
    getComputedStyle
  }} = scope;

{script_code}
}}

export async function prefetchParityTab({{ getDirectUrl }}) {{
  return prefetchParityPage({{ page, getDirectUrl }});
}}

export async function mountParityTab({{ host, getDirectUrl, onHistoryChange }}) {{
  return mountParityPage({{
    host,
    getDirectUrl,
    onHistoryChange,
    page,
    runInlineScript,
  }});
}}
"""


def main() -> None:
    generated = 0
    for view in VIEWS:
        for filename in TAB_FILES:
            source = STATIC_DIR / view / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing source dashboard file: {source}")
            html = source.read_text(encoding="utf-8")
            page = extract_page_definition(html)

            out_name = filename.replace(".html", ".js")
            out_path = OUT_DIR / view / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(render_module(page, filename), encoding="utf-8")
            generated += 1
            print(f"generated: {out_path.relative_to(ROOT)}")

    print(f"done: {generated} native parity modules")


if __name__ == "__main__":
    main()
