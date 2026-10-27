#!/usr/bin/env python3
"""Fit the deployable experimental text model on all labelled samples."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from train_text_baseline import BUCKETS, features, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--prior-mix", type=float, default=0.25)
    parser.add_argument("--low-threshold", type=float, default=0.65)
    parser.add_argument("--high-threshold", type=float, default=1.55)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    class_docs: Counter[str] = Counter()
    class_features: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    vocabulary: set[str] = set()
    for row in rows:
        ratio = float(row["likes"]) / float(row["account_baseline_likes"])
        label = "low" if ratio < args.low_threshold else "high" if ratio > args.high_threshold else "medium"
        counts = features(row["transcript"])
        class_docs[label] += 1
        class_features[label].update(counts)
        totals[label] += sum(counts.values())
        vocabulary.update(counts)
    if set(class_docs) != set(BUCKETS):
        raise SystemExit(f"training data must contain all buckets; got {sorted(class_docs)}")

    payload = {
        "schema_version": "1.0",
        "model_version": "talking-head-text-pilot-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "training_samples": len(rows),
        "class_docs": dict(class_docs),
        "class_features": {label: dict(class_features[label]) for label in BUCKETS},
        "totals": dict(totals),
        "vocab_size": len(vocabulary),
        "alpha": args.alpha,
        "temperature": args.temperature,
        "prior_mix": args.prior_mix,
        "target": "likes / same-account median likes",
        "thresholds": {"low_below": args.low_threshold, "high_above": args.high_threshold},
        "validation_status": "experimental; 13 visually verified videos; not 90-percent accurate",
        "evaluation_files": [
            "runs/benchmark/pilot_v0/evaluation_v0.json",
            "runs/benchmark/pilot_v1/evaluation_v1.json"
        ]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({"ok": True, "out": str(args.out), "training_samples": len(rows), "vocab_size": len(vocabulary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
