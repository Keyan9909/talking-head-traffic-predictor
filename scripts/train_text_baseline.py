#!/usr/bin/env python3
"""Train a sealed-label-safe character n-gram Naive Bayes text baseline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BUCKETS = ("low", "medium", "high")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def features(text: str) -> Counter[str]:
    compact = "".join(str(text or "").split())
    result: Counter[str] = Counter()
    for width in (1, 2, 3):
        result.update(compact[index:index + width] for index in range(max(0, len(compact) - width + 1)))
    result["__LEN_SHORT__" if len(compact) < 120 else "__LEN_LONG__" if len(compact) > 360 else "__LEN_MID__"] += 3
    return result


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    scaled = {key: value / temperature for key, value in scores.items()}
    ceiling = max(scaled.values())
    values = {key: math.exp(value - ceiling) for key, value in scaled.items()}
    total = sum(values.values())
    return {key: values[key] / total for key in BUCKETS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prior-mix", type=float, default=0.0)
    parser.add_argument("--rubric-version", default="char-ngram-nb-v0")
    args = parser.parse_args()
    if args.temperature <= 0:
        raise SystemExit("temperature must be > 0")
    if not 0 <= args.prior_mix < 1:
        raise SystemExit("prior-mix must be in [0, 1)")

    development = read_jsonl(args.development)
    blind = read_jsonl(args.blind)
    class_docs = Counter(row["observed_bucket"] for row in development)
    class_features: dict[str, Counter[str]] = defaultdict(Counter)
    totals = Counter()
    vocabulary: set[str] = set()
    for row in development:
        label = row["observed_bucket"]
        counts = features(row["transcript"])
        class_features[label].update(counts)
        totals[label] += sum(counts.values())
        vocabulary.update(counts)
    if set(class_docs) != set(BUCKETS):
        raise SystemExit(f"development set must contain all buckets; got {sorted(class_docs)}")

    vocab_size = max(1, len(vocabulary))
    n_docs = len(development)
    predictions = []
    for row in blind:
        counts = features(row["transcript"])
        scores = {}
        for label in BUCKETS:
            score = math.log((class_docs[label] + args.alpha) / (n_docs + args.alpha * len(BUCKETS)))
            denominator = totals[label] + args.alpha * vocab_size
            for feature, count in counts.items():
                numerator = class_features[label][feature] + args.alpha
                score += count * math.log(numerator / denominator)
            scores[label] = score
        probabilities = softmax(scores, args.temperature)
        if args.prior_mix:
            empirical_prior = {label: class_docs[label] / n_docs for label in BUCKETS}
            probabilities = {
                label: (1.0 - args.prior_mix) * probabilities[label]
                + args.prior_mix * empirical_prior[label]
                for label in BUCKETS
            }
        predicted_percentile = sum(
            probabilities[label] * percentile
            for label, percentile in (("low", 17.0), ("medium", 50.0), ("high", 83.0))
        )
        predictions.append({
            "blind_id": row["blind_id"],
            "rubric_version": args.rubric_version,
            "probabilities": {key: round(probabilities[key], 8) for key in BUCKETS},
            "predicted_percentile": round(predicted_percentile, 3),
            "confidence": "experimental",
            "rationale": (
                "Blind character n-gram baseline trained only on development transcripts; "
                f"temperature={args.temperature:g}, prior_mix={args.prior_mix:g}."
            ),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(predictions, key=lambda item: item["blind_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "development": len(development), "blind": len(predictions), "vocabulary": vocab_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
