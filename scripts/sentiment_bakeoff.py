#!/usr/bin/env python3
"""
Local sentiment bake-off runner for risk-dashboard data.

Compares one or more Hugging Face models on sampled records from Postgres and
reports speed + label agreement versus current stored sentiment labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MISSING_DEPS: list[str] = []
DB_DRIVER = None

try:  # pragma: no cover - runtime import guard
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - runtime import guard
    psycopg2 = None
    RealDictCursor = None
else:  # pragma: no cover - runtime import guard
    DB_DRIVER = "psycopg2"

if DB_DRIVER is None:
    try:  # pragma: no cover - runtime import guard
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:  # pragma: no cover - runtime import guard
        psycopg = None
        dict_row = None
        MISSING_DEPS.extend(["psycopg2 or psycopg"])
    else:  # pragma: no cover - runtime import guard
        DB_DRIVER = "psycopg3"

try:  # pragma: no cover - runtime import guard
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
except ImportError:  # pragma: no cover - runtime import guard
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    BertTokenizer = None
    MISSING_DEPS.extend(["torch", "transformers"])


VALID_LABELS = {"positive", "neutral", "negative"}
SOURCE_CHOICES = ("company_articles", "ceo_articles", "serp_results", "serp_features")


@dataclass
class ModelRunResult:
    model_id: str
    elapsed_seconds: float
    rows_per_second: float
    avg_ms_per_row: float
    agreement_rate: float | None
    comparable_rows: int
    baseline_distribution: dict[str, int]
    model_distribution: dict[str, int]
    confusion: dict[str, dict[str, int]]
    predictions: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local sentiment model bake-off.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yiyanghkust/finbert-tone", "ProsusAI/finbert"],
        help="Hugging Face model ids to evaluate.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_CHOICES),
        choices=SOURCE_CHOICES,
        help="Data sources to sample from.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=45,
        help="Limit sample to rows from the last N days. Use 0 to disable.",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=300,
        help="Maximum rows to sample per source table.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max_length for truncation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device.",
    )
    parser.add_argument(
        "--include-url",
        action="store_true",
        help="Include URL text in model input.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where results are written.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL"),
        help="Postgres DSN (defaults to DATABASE_URL or SUPABASE_DB_URL env var).",
    )
    return parser.parse_args()


def ensure_runtime_deps() -> None:
    if MISSING_DEPS:
        unique = sorted(set(MISSING_DEPS))
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(unique)
            + "\nInstall with:\n  pip install -r requirements-bakeoff.txt"
        )


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    raw = str(label).strip().lower()
    if not raw:
        return None
    if raw in VALID_LABELS:
        return raw
    if "positive" in raw or raw == "pos":
        return "positive"
    if "negative" in raw or raw == "neg":
        return "negative"
    if "neutral" in raw or raw == "neu":
        return "neutral"
    return None


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_conn(database_url: str | None):
    if not database_url:
        raise SystemExit("Set DATABASE_URL or pass --database-url.")
    if DB_DRIVER == "psycopg2":
        return psycopg2.connect(database_url)
    if DB_DRIVER == "psycopg3":
        return psycopg.connect(database_url)
    raise SystemExit("No postgres driver available. Install requirements-bakeoff.txt")


def query_rows(conn, sources: list[str], days_back: int, per_source_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if DB_DRIVER == "psycopg2":
        cursor = conn.cursor(cursor_factory=RealDictCursor)
    else:
        cursor = conn.cursor(row_factory=dict_row)

    with cursor as cur:
        if "company_articles" in sources:
            cur.execute(
                """
                select
                  'company_articles'::text as source,
                  cam.id::text as record_id,
                  coalesce(a.published_at, a.first_seen_at, a.last_seen_at) as item_ts,
                  a.canonical_url as url,
                  a.title,
                  a.snippet,
                  coalesce(ov.override_sentiment_label, cam.llm_sentiment_label, cam.sentiment_label) as baseline_label
                from company_article_mentions cam
                join articles a on a.id = cam.article_id
                left join company_article_overrides ov
                  on ov.company_id = cam.company_id and ov.article_id = cam.article_id
                where coalesce(a.title, '') <> ''
                  and (
                    %s <= 0
                    or coalesce(a.published_at, a.first_seen_at, a.last_seen_at) >= now() - (%s::text || ' days')::interval
                  )
                order by random()
                limit %s
                """,
                (days_back, days_back, per_source_limit),
            )
            rows.extend(dict(r) for r in cur.fetchall())

        if "ceo_articles" in sources:
            cur.execute(
                """
                select
                  'ceo_articles'::text as source,
                  cem.id::text as record_id,
                  coalesce(a.published_at, a.first_seen_at, a.last_seen_at) as item_ts,
                  a.canonical_url as url,
                  a.title,
                  a.snippet,
                  coalesce(ov.override_sentiment_label, cem.llm_sentiment_label, cem.sentiment_label) as baseline_label
                from ceo_article_mentions cem
                join articles a on a.id = cem.article_id
                left join ceo_article_overrides ov
                  on ov.ceo_id = cem.ceo_id and ov.article_id = cem.article_id
                where coalesce(a.title, '') <> ''
                  and (
                    %s <= 0
                    or coalesce(a.published_at, a.first_seen_at, a.last_seen_at) >= now() - (%s::text || ' days')::interval
                  )
                order by random()
                limit %s
                """,
                (days_back, days_back, per_source_limit),
            )
            rows.extend(dict(r) for r in cur.fetchall())

        if "serp_results" in sources:
            cur.execute(
                """
                select
                  'serp_results'::text as source,
                  r.id::text as record_id,
                  sr.run_at as item_ts,
                  r.url,
                  r.title,
                  r.snippet,
                  coalesce(ov.override_sentiment_label, r.llm_sentiment_label, r.sentiment_label) as baseline_label
                from serp_results r
                join serp_runs sr on sr.id = r.serp_run_id
                left join serp_result_overrides ov on ov.serp_result_id = r.id
                where coalesce(r.title, '') <> ''
                  and (%s <= 0 or sr.run_at >= now() - (%s::text || ' days')::interval)
                order by random()
                limit %s
                """,
                (days_back, days_back, per_source_limit),
            )
            rows.extend(dict(r) for r in cur.fetchall())

        if "serp_features" in sources:
            cur.execute(
                """
                select
                  'serp_features'::text as source,
                  sfi.id::text as record_id,
                  sfi.date::timestamp as item_ts,
                  sfi.url,
                  sfi.title,
                  sfi.snippet,
                  coalesce(ov.override_sentiment_label, uov.override_sentiment_label, sfi.llm_sentiment_label, sfi.sentiment_label) as baseline_label
                from serp_feature_items sfi
                left join serp_feature_item_overrides ov on ov.serp_feature_item_id = sfi.id
                left join serp_feature_url_overrides uov on uov.entity_type = sfi.entity_type and uov.entity_id = sfi.entity_id and uov.feature_type = sfi.feature_type and uov.url_hash = sfi.url_hash
                where coalesce(sfi.title, '') <> ''
                  and (%s <= 0 or sfi.date >= current_date - (%s::text || ' days')::interval)
                order by random()
                limit %s
                """,
                (days_back, days_back, per_source_limit),
            )
            rows.extend(dict(r) for r in cur.fetchall())
    return rows


def build_text(row: dict[str, Any], include_url: bool) -> str:
    title = (row.get("title") or "").strip()
    snippet = (row.get("snippet") or "").strip()
    url = (row.get("url") or "").strip()
    if include_url:
        return f"Title: {title}\nSnippet: {snippet}\nURL: {url}".strip()
    if title and snippet:
        return f"{title}. {snippet}"
    return title or snippet or (f"URL: {url}" if url else "")


class HFClassifier:
    def __init__(self, model_id: str, device: torch.device, max_length: int):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.label_map = self._derive_label_map()

    @staticmethod
    def _load_tokenizer(model_id: str):
        fast_exc: Exception | None = None
        slow_exc: Exception | None = None
        try:
            return AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as exc:
            fast_exc = exc
        try:
            return AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except Exception as exc:
            slow_exc = exc
        # Explicit slow BERT fallback for FinBERT-family models, bypassing fast-tokenizer conversion.
        try:
            return BertTokenizer.from_pretrained(model_id)
        except Exception as bert_exc:
            raise RuntimeError(
                "Failed to load tokenizer for "
                f"{model_id}. Install missing tokenizer deps and retry:\n"
                "  pip install protobuf sentencepiece tiktoken\n"
                f"Fast tokenizer error: {fast_exc}\n"
                f"Slow tokenizer error: {slow_exc}\n"
                f"BertTokenizer fallback error: {bert_exc}"
            ) from bert_exc

    def _predict_ids(self, texts: list[str], batch_size: int) -> tuple[list[int], list[float]]:
        pred_ids: list[int] = []
        confidences: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                conf, idx = probs.max(dim=-1)
            pred_ids.extend(idx.detach().cpu().tolist())
            confidences.extend(conf.detach().cpu().tolist())
        return pred_ids, confidences

    def _derive_label_map(self) -> dict[int, str]:
        # Prefer explicit id2label metadata when available.
        mapped: dict[int, str] = {}
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, raw in id2label.items():
            try:
                key = int(idx)
            except (TypeError, ValueError):
                continue
            norm = normalize_label(str(raw))
            if norm:
                mapped[key] = norm
        if len(mapped) == 3:
            return mapped

        # Probe mapping fallback for generic LABEL_X setups.
        probe_pos = "Record profits and strong growth beat expectations."
        probe_neg = "Fraud scandal and lawsuit trigger bankruptcy concerns."
        probe_neu = "The company announced its quarterly results schedule."
        ids, _ = self._predict_ids([probe_pos, probe_neg, probe_neu], batch_size=3)
        if len(set(ids)) == 3:
            return {ids[0]: "positive", ids[1]: "negative", ids[2]: "neutral"}

        # Last resort defaults for known model ids.
        lid = self.model_id.lower()
        if "finbert-tone" in lid:
            return {0: "neutral", 1: "positive", 2: "negative"}
        if "prosusai/finbert" in lid:
            return {0: "positive", 1: "negative", 2: "neutral"}
        return {}

    def predict(self, texts: list[str], batch_size: int) -> tuple[list[str | None], list[float]]:
        pred_ids, confidences = self._predict_ids(texts, batch_size=batch_size)
        labels: list[str | None] = []
        for pid in pred_ids:
            label = self.label_map.get(pid)
            if not label:
                raw = None
                if getattr(self.model.config, "id2label", None):
                    raw = self.model.config.id2label.get(pid)
                label = normalize_label(str(raw) if raw is not None else None)
            labels.append(label)
        return labels, confidences


def build_confusion() -> dict[str, dict[str, int]]:
    return {
        "positive": {"positive": 0, "neutral": 0, "negative": 0},
        "neutral": {"positive": 0, "neutral": 0, "negative": 0},
        "negative": {"positive": 0, "neutral": 0, "negative": 0},
    }


def evaluate_model(
    model_id: str,
    rows: list[dict[str, Any]],
    texts: list[str],
    baselines: list[str | None],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> ModelRunResult:
    clf = HFClassifier(model_id, device=device, max_length=max_length)
    start = time.perf_counter()
    pred_labels, pred_conf = clf.predict(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - start

    comparable = 0
    agree = 0
    baseline_dist: Counter[str] = Counter()
    model_dist: Counter[str] = Counter()
    confusion = build_confusion()
    row_preds: list[dict[str, Any]] = []

    for row, base, pred, conf in zip(rows, baselines, pred_labels, pred_conf):
        if pred in VALID_LABELS:
            model_dist[pred] += 1
        if base in VALID_LABELS:
            baseline_dist[base] += 1

        comparable_flag = base in VALID_LABELS and pred in VALID_LABELS
        is_agree = False
        if comparable_flag:
            comparable += 1
            confusion[base][pred] += 1
            if base == pred:
                agree += 1
                is_agree = True

        row_preds.append(
            {
                "source": row.get("source"),
                "record_id": row.get("record_id"),
                "item_ts": row.get("item_ts"),
                "url": row.get("url"),
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "baseline_label": base,
                "model_id": model_id,
                "model_label": pred,
                "model_confidence": round(float(conf), 6),
                "agree": is_agree if comparable_flag else None,
            }
        )

    total_rows = max(len(rows), 1)
    rps = len(rows) / elapsed if elapsed > 0 else 0.0
    avg_ms = (elapsed / total_rows) * 1000.0
    agreement_rate = (agree / comparable) if comparable else None

    return ModelRunResult(
        model_id=model_id,
        elapsed_seconds=elapsed,
        rows_per_second=rps,
        avg_ms_per_row=avg_ms,
        agreement_rate=agreement_rate,
        comparable_rows=comparable,
        baseline_distribution=dict(baseline_dist),
        model_distribution=dict(model_dist),
        confusion=confusion,
        predictions=row_preds,
    )


def write_outputs(
    out_dir: Path,
    run_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    results: list[ModelRunResult],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=str)

    summary = {
        "run_meta": run_meta,
        "models": [
            {
                "model_id": r.model_id,
                "elapsed_seconds": round(r.elapsed_seconds, 4),
                "rows_per_second": round(r.rows_per_second, 3),
                "avg_ms_per_row": round(r.avg_ms_per_row, 3),
                "agreement_rate": round(r.agreement_rate, 4) if r.agreement_rate is not None else None,
                "comparable_rows": r.comparable_rows,
                "baseline_distribution": r.baseline_distribution,
                "model_distribution": r.model_distribution,
                "confusion": r.confusion,
            }
            for r in results
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "sampled_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "record_id", "item_ts", "url", "title", "snippet", "baseline_label"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.get("source"),
                    "record_id": row.get("record_id"),
                    "item_ts": row.get("item_ts"),
                    "url": row.get("url"),
                    "title": row.get("title"),
                    "snippet": row.get("snippet"),
                    "baseline_label": normalize_label(row.get("baseline_label")),
                }
            )

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "record_id",
                "item_ts",
                "url",
                "title",
                "snippet",
                "baseline_label",
                "model_id",
                "model_label",
                "model_confidence",
                "agree",
            ],
        )
        writer.writeheader()
        for result in results:
            for pred in result.predictions:
                writer.writerow(pred)

    # Pairwise model disagreements to speed review.
    by_model: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for result in results:
        by_model[result.model_id] = {
            (str(p["source"]), str(p["record_id"])): p for p in result.predictions
        }
    disagreement_rows: list[dict[str, Any]] = []
    model_ids = [r.model_id for r in results]
    for i, left in enumerate(model_ids):
        for right in model_ids[i + 1 :]:
            left_map = by_model[left]
            right_map = by_model[right]
            for key in set(left_map).intersection(right_map):
                lp = left_map[key]
                rp = right_map[key]
                ll = lp.get("model_label")
                rl = rp.get("model_label")
                if ll and rl and ll != rl:
                    disagreement_rows.append(
                        {
                            "source": lp.get("source"),
                            "record_id": lp.get("record_id"),
                            "baseline_label": lp.get("baseline_label"),
                            "left_model": left,
                            "left_label": ll,
                            "left_confidence": lp.get("model_confidence"),
                            "right_model": right,
                            "right_label": rl,
                            "right_confidence": rp.get("model_confidence"),
                            "title": lp.get("title"),
                            "snippet": lp.get("snippet"),
                            "url": lp.get("url"),
                        }
                    )
    with (out_dir / "model_disagreements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "record_id",
                "baseline_label",
                "left_model",
                "left_label",
                "left_confidence",
                "right_model",
                "right_label",
                "right_confidence",
                "title",
                "snippet",
                "url",
            ],
        )
        writer.writeheader()
        writer.writerows(disagreement_rows)


def print_summary(results: list[ModelRunResult], out_dir: Path) -> None:
    print("\n=== Sentiment Bake-off Summary ===")
    for r in results:
        agree_pct = (
            f"{(r.agreement_rate * 100):.2f}%" if r.agreement_rate is not None else "n/a"
        )
        print(
            f"- {r.model_id}\n"
            f"  speed: {r.rows_per_second:.2f} rows/sec ({r.avg_ms_per_row:.2f} ms/row)\n"
            f"  agreement vs baseline: {agree_pct} on {r.comparable_rows} comparable rows\n"
            f"  label distribution: {r.model_distribution}"
        )
    print(f"\nOutputs written to: {out_dir}")


def main() -> None:
    args = parse_args()
    ensure_runtime_deps()
    device = choose_device(args.device)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"sentiment_bakeoff_{now}"

    with get_conn(args.database_url) as conn:
        rows = query_rows(
            conn=conn,
            sources=args.sources,
            days_back=args.days_back,
            per_source_limit=args.per_source_limit,
        )

    if not rows:
        raise SystemExit("No rows returned. Try larger --days-back or check your data.")

    for row in rows:
        row["baseline_label"] = normalize_label(row.get("baseline_label"))

    texts = [build_text(row, include_url=args.include_url) for row in rows]
    baselines = [row.get("baseline_label") for row in rows]

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "models": args.models,
        "sources": args.sources,
        "days_back": args.days_back,
        "per_source_limit": args.per_source_limit,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "include_url": bool(args.include_url),
        "sample_size": len(rows),
    }

    results: list[ModelRunResult] = []
    for model_id in args.models:
        print(f"Running model: {model_id}")
        result = evaluate_model(
            model_id=model_id,
            rows=rows,
            texts=texts,
            baselines=baselines,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        results.append(result)

    write_outputs(out_dir=out_dir, run_meta=run_meta, rows=rows, results=results)
    print_summary(results=results, out_dir=out_dir)


if __name__ == "__main__":
    main()
