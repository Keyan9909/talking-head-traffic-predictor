#!/usr/bin/env python3
"""Predict low/medium/high relative traffic probabilities for one Chinese script."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

from train_text_baseline import BUCKETS, features, softmax


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "talking_head_text_pilot_v1.json.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file", type=Path)
    args = parser.parse_args()
    script = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    script = str(script or "").strip()
    if not script:
        raise SystemExit("script is empty")
    with gzip.open(args.model, "rt", encoding="utf-8") as handle:
        model = json.load(handle)

    counts = features(script)
    n_docs = sum(model["class_docs"].values())
    scores = {}
    for label in BUCKETS:
        score = math.log((model["class_docs"][label] + model["alpha"]) / (n_docs + model["alpha"] * len(BUCKETS)))
        denominator = model["totals"][label] + model["alpha"] * model["vocab_size"]
        stored = model["class_features"][label]
        for feature, count in counts.items():
            score += count * math.log((stored.get(feature, 0) + model["alpha"]) / denominator)
        scores[label] = score
    probabilities = softmax(scores, model["temperature"])
    mix = model["prior_mix"]
    probabilities = {
        label: (1 - mix) * probabilities[label] + mix * model["class_docs"][label] / n_docs
        for label in BUCKETS
    }
    predicted = max(BUCKETS, key=lambda label: probabilities[label])
    percentile = sum(probabilities[label] * value for label, value in (("low", 17), ("medium", 50), ("high", 83)))
    payload = {
        "model_version": model["model_version"],
        "prediction_scope": "script-only; relative to same-account normal likes; presenter and editing excluded",
        "predicted_bucket": predicted,
        "probabilities": {label: round(probabilities[label], 6) for label in BUCKETS},
        "predicted_percentile": round(percentile, 1),
        "script_characters": len(script),
        "confidence": "low",
        "validation_status": model["validation_status"],
        "warning": "This is an experimental cold-start estimate, not a guarantee of virality or exact likes."
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
