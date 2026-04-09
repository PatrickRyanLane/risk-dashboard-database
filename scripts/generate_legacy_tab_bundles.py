#!/usr/bin/env python3
"""
Generate JS legacy tab bundles from standalone dashboard HTML files.

The output is consumed by dashboard_app/static/app/legacy-tab-adapter.js so
the shell can run legacy-equivalent tabs without fetching standalone HTML pages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "dashboard_app" / "static"
BUNDLE_DIR = STATIC_DIR / "app" / "legacy-bundles"

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

    markup = SCRIPT_RE.sub("", body_html)
    styles = "\n\n".join((style or "").strip() for style in STYLE_RE.findall(html) if (style or "").strip())
    inline_script = "\n\n".join(script for script in inline_scripts if script)

    return {
        "styles": styles,
        "markup": markup.strip(),
        "externalScripts": external_scripts,
        "inlineScript": inline_script,
    }


def write_bundle(out_path: Path, page: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "export default " + json.dumps(page, ensure_ascii=True, indent=2) + ";\n",
        encoding="utf-8",
    )


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
            out_path = BUNDLE_DIR / view / out_name
            write_bundle(out_path, page)
            generated += 1
            print(f"generated: {out_path.relative_to(ROOT)}")
    print(f"done: {generated} bundles")


if __name__ == "__main__":
    main()
