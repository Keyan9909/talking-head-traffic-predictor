#!/usr/bin/env python3
"""Build and evaluate leakage-resistant talking-head script benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUCKETS = ("low", "medium", "high")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sample {row.get('sample_id')!r}: {key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"sample {row.get('sample_id')!r}: {key} must be finite and >= 0")
    return value


def label_samples(
    samples: list[dict[str, Any]], low_threshold: float, high_threshold: float
) -> list[dict[str, Any]]:
    if not 0 < low_threshold < high_threshold:
        raise ValueError("thresholds must satisfy 0 < low < high")

    outcome_metrics: set[str] = set()
    for row in samples:
        declared = row.get("outcome_metric")
        if declared is None:
            declared = "views" if row.get("views") is not None else "likes"
        if declared not in {"views", "likes"}:
            raise ValueError("outcome_metric must be uniformly 'views' or 'likes'")
        outcome_metrics.add(declared)
        row["_outcome_metric"] = declared
    if len(outcome_metrics) != 1:
        raise ValueError("do not mix view-labelled and like-labelled samples in one run")
    outcome_metric = next(iter(outcome_metrics))

    account_values: dict[str, list[float]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in samples:
        sample_id = row.get("sample_id")
        account_id = row.get("account_id")
        transcript = row.get("transcript")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("every sample needs a non-empty sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError(f"sample {sample_id!r}: account_id is required")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"sample {sample_id!r}: transcript is required")
        account_values[account_id].append(require_number(row, outcome_metric))

    labelled: list[dict[str, Any]] = []
    for row in samples:
        account_id = row["account_id"]
        baseline_raw = row.get("account_baseline_value")
        if baseline_raw is None:
            baseline_raw = row.get(f"account_baseline_{outcome_metric}")
        if baseline_raw is None:
            if len(account_values[account_id]) < 3:
                raise ValueError(
                    f"account {account_id!r} needs an account baseline or at least 3 samples"
                )
            baseline = float(statistics.median(account_values[account_id]))
        else:
            if isinstance(baseline_raw, bool) or not isinstance(baseline_raw, (int, float)):
                raise ValueError(f"sample {row['sample_id']!r}: baseline must be numeric")
            baseline = float(baseline_raw)
        if baseline <= 0:
            raise ValueError(f"sample {row['sample_id']!r}: baseline must be > 0")
        ratio = require_number(row, outcome_metric) / baseline
        bucket = "low" if ratio < low_threshold else "high" if ratio > high_threshold else "medium"
        labelled.append({**row, "_reach_ratio": ratio, "_bucket": bucket, "_baseline": baseline})

    ordered = sorted(row["_reach_ratio"] for row in labelled)
    for row in labelled:
        lower = sum(value < row["_reach_ratio"] for value in ordered)
        equal = sum(value == row["_reach_ratio"] for value in ordered)
        row["_actual_percentile"] = 100.0 * (lower + 0.5 * equal) / len(ordered)
    return labelled


def blind_id(seed: int, sample_id: str) -> str:
    token = sha256_bytes(f"{seed}|{sample_id}".encode("utf-8"))[:16]
    return f"B-{token}"


def build_dataset(
    input_path: Path,
    out_dir: Path,
    blind_per_bucket: int,
    seed: int,
    low_threshold: float = 0.65,
    high_threshold: float = 1.55,
) -> dict[str, Any]:
    if blind_per_bucket < 1:
        raise ValueError("blind_per_bucket must be >= 1")
    samples = label_samples(read_jsonl(input_path), low_threshold, high_threshold)
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    for row in samples:
        by_bucket[row["_bucket"]].append(row)
    for bucket in BUCKETS:
        if len(by_bucket[bucket]) < blind_per_bucket + 1:
            raise ValueError(
                f"bucket {bucket!r} has {len(by_bucket[bucket])} samples; "
                f"need at least {blind_per_bucket + 1} to keep development examples"
            )
        rng.shuffle(by_bucket[bucket])

    held_ids = {
        row["sample_id"]
        for bucket in BUCKETS
        for row in by_bucket[bucket][:blind_per_bucket]
    }
    development_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []

    for row in samples:
        anonymous_id = blind_id(seed, row["sample_id"])
        if row["sample_id"] in held_ids:
            blind_rows.append({"blind_id": anonymous_id, "transcript": row["transcript"]})
            sealed_rows.append(
                {
                    "blind_id": anonymous_id,
                    "sample_id": row["sample_id"],
                    "actual_bucket": row["_bucket"],
                    "actual_percentile": round(row["_actual_percentile"], 6),
                    "performance_ratio": round(row["_reach_ratio"], 6),
                    "outcome_metric": row["_outcome_metric"],
                    "outcome_value": row[row["_outcome_metric"]],
                    "views": row.get("views"),
                    "likes": row.get("likes"),
                    "comments": row.get("comments"),
                    "shares": row.get("shares"),
                    "account_id": row["account_id"],
                    "source_url": row.get("source_url"),
                }
            )
        else:
            development_rows.append(
                {
                    "development_id": anonymous_id.replace("B-", "D-", 1),
                    "transcript": row["transcript"],
                    "observed_bucket": row["_bucket"],
                    "performance_ratio": round(row["_reach_ratio"], 6),
                    "outcome_metric": row["_outcome_metric"],
                    "actual_percentile": round(row["_actual_percentile"], 6),
                }
            )

    blind_rows.sort(key=lambda row: row["blind_id"])
    sealed_rows.sort(key=lambda row: row["blind_id"])
    development_rows.sort(key=lambda row: row["development_id"])
    blind_dir = out_dir / "blind"
    vault_dir = out_dir / "vault"
    development_dir = out_dir / "development"
    blind_path = blind_dir / "blind_inputs.jsonl"
    labels_path = vault_dir / "sealed_labels.jsonl"
    write_jsonl(blind_path, blind_rows)
    write_jsonl(labels_path, sealed_rows)
    write_jsonl(development_dir / "training_examples.jsonl", development_rows)

    manifest = {
        "schema_version": "1.0",
        "created_at": now_iso(),
        "seed": seed,
        "target": f"{samples[0]['_outcome_metric']}_at_fixed_horizon / account_baseline_{samples[0]['_outcome_metric']}",
        "thresholds": {"low_below": low_threshold, "high_above": high_threshold},
        "blind_count": len(blind_rows),
        "blind_ids": [row["blind_id"] for row in blind_rows],
        "blind_inputs_sha256": sha256_file(blind_path),
        "sealed_labels_sha256": sha256_file(labels_path),
        "isolation_warning": "Do not expose vault or raw data to the prediction agent before freeze.",
    }
    write_json(blind_dir / "manifest.json", manifest)
    return manifest


def validate_prediction(row: dict[str, Any]) -> None:
    if not isinstance(row.get("blind_id"), str):
        raise ValueError("prediction missing blind_id")
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(BUCKETS):
        raise ValueError(f"{row['blind_id']}: probabilities must contain low, medium, high")
    values = []
    for bucket in BUCKETS:
        value = probabilities[bucket]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{row['blind_id']}: probability {bucket} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{row['blind_id']}: invalid probability {bucket}")
        values.append(value)
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError(f"{row['blind_id']}: probabilities must sum to 1")
    percentile = row.get("predicted_percentile")
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise ValueError(f"{row['blind_id']}: predicted_percentile must be numeric")
    if not math.isfinite(float(percentile)) or not 0 <= float(percentile) <= 100:
        raise ValueError(f"{row['blind_id']}: predicted_percentile must be in [0, 100]")
    if not isinstance(row.get("rubric_version"), str) or not row["rubric_version"]:
        raise ValueError(f"{row['blind_id']}: rubric_version is required")


def freeze_predictions(
    predictions_path: Path, manifest_path: Path, out_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = read_jsonl(predictions_path)
    for row in predictions:
        validate_prediction(row)
    ids = [row["blind_id"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("predictions contain duplicate blind_id values")
    if set(ids) != set(manifest["blind_ids"]):
        missing = sorted(set(manifest["blind_ids"]) - set(ids))
        extra = sorted(set(ids) - set(manifest["blind_ids"]))
        raise ValueError(f"prediction IDs do not match manifest; missing={missing}, extra={extra}")
    predictions.sort(key=lambda row: row["blind_id"])
    prediction_hash = sha256_bytes(canonical_bytes(predictions))
    frozen = {
        "schema_version": "1.0",
        "frozen_at": now_iso(),
        "manifest_sha256": sha256_file(manifest_path),
        "blind_inputs_sha256": manifest["blind_inputs_sha256"],
        "predictions_sha256": prediction_hash,
        "predictions": predictions,
    }
    write_json(out_path, frozen)
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(
        sha256_file(out_path) + "\n", encoding="ascii"
    )
    return frozen


def evaluate_predictions(
    frozen_path: Path, labels_path: Path, manifest_path: Path, out_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("frozen predictions reference a different manifest")
    predictions = frozen.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("frozen file has no predictions list")
    if frozen.get("predictions_sha256") != sha256_bytes(canonical_bytes(predictions)):
        raise ValueError("frozen predictions were modified after freeze")
    if manifest.get("sealed_labels_sha256") != sha256_file(labels_path):
        raise ValueError("sealed labels do not match the blind manifest")

    labels = {row["blind_id"]: row for row in read_jsonl(labels_path)}
    if set(labels) != {row["blind_id"] for row in predictions}:
        raise ValueError("label IDs do not match frozen prediction IDs")

    confusion = {actual: {predicted: 0 for predicted in BUCKETS} for actual in BUCKETS}
    correct = 0
    log_loss = 0.0
    brier = 0.0
    percentile_errors: list[float] = []
    high_tp = high_fp = high_fn = 0
    for row in predictions:
        validate_prediction(row)
        actual = labels[row["blind_id"]]["actual_bucket"]
        probabilities = {key: float(value) for key, value in row["probabilities"].items()}
        predicted = max(BUCKETS, key=lambda bucket: probabilities[bucket])
        confusion[actual][predicted] += 1
        correct += int(actual == predicted)
        log_loss -= math.log(max(probabilities[actual], 1e-15))
        brier += sum(
            (probabilities[bucket] - (1.0 if bucket == actual else 0.0)) ** 2
            for bucket in BUCKETS
        )
        percentile_errors.append(
            abs(float(row["predicted_percentile"]) - float(labels[row["blind_id"]]["actual_percentile"]))
        )
        high_tp += int(actual == "high" and predicted == "high")
        high_fp += int(actual != "high" and predicted == "high")
        high_fn += int(actual == "high" and predicted != "high")

    n = len(predictions)
    recalls = []
    for bucket in BUCKETS:
        total = sum(confusion[bucket].values())
        if total:
            recalls.append(confusion[bucket][bucket] / total)
    report = {
        "schema_version": "1.0",
        "evaluated_at": now_iso(),
        "n": n,
        "bucket_accuracy": correct / n,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "multiclass_brier_score": brier / n,
        "log_loss": log_loss / n,
        "high_precision": high_tp / (high_tp + high_fp) if high_tp + high_fp else None,
        "high_recall": high_tp / (high_tp + high_fn) if high_tp + high_fn else None,
        "percentile_mae": sum(percentile_errors) / n,
        "confusion_matrix": confusion,
        "frozen_predictions_sha256": sha256_file(frozen_path),
    }
    write_json(out_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build development, blind, and sealed sets")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--blind-per-bucket", type=int, default=10)
    build.add_argument("--seed", type=int, default=20260802)
    build.add_argument("--low-threshold", type=float, default=0.65)
    build.add_argument("--high-threshold", type=float, default=1.55)

    freeze = subparsers.add_parser("freeze", help="validate and freeze blind predictions")
    freeze.add_argument("--predictions", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate frozen predictions")
    evaluate.add_argument("--frozen", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        result = build_dataset(
            args.input,
            args.out,
            args.blind_per_bucket,
            args.seed,
            args.low_threshold,
            args.high_threshold,
        )
    elif args.command == "freeze":
        result = freeze_predictions(args.predictions, args.manifest, args.out)
    else:
        result = evaluate_predictions(args.frozen, args.labels, args.manifest, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
