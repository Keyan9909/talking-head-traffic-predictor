# Talking-Head Traffic Predictor

An experimental, leakage-resistant benchmark for estimating the relative traffic tier of Chinese talking-head scripts.

It does **not** predict exact views or promise commercial accuracy. The project focuses on a reproducible process:

1. label outcomes relative to each account's own baseline;
2. split development, blind, and sealed-label sets;
3. freeze predictions with a SHA-256 digest;
4. reveal labels only during evaluation;
5. report accuracy, log loss, Brier score, and percentile error.

## Why this repository is data-safe

This public repository contains only source code, methodology, aggregate pilot results, and synthetic examples. It intentionally excludes real videos, creator identities, URLs, transcripts, account metrics, local paths, sealed labels from the original pilot, and the pilot model trained from that data.

## Quick start

Python 3.10+ is sufficient; the core tools use only the standard library.

```powershell
python scripts/dataset_tool.py build `
  --input examples/synthetic_samples.jsonl `
  --out runs/demo `
  --blind-per-bucket 1 `
  --seed 7

python scripts/train_text_baseline.py `
  --development runs/demo/development/training_examples.jsonl `
  --blind runs/demo/blind/blind_inputs.jsonl `
  --out runs/demo/predictions.jsonl

python scripts/dataset_tool.py freeze `
  --predictions runs/demo/predictions.jsonl `
  --manifest runs/demo/blind/manifest.json `
  --out runs/demo/frozen.json

python scripts/dataset_tool.py evaluate `
  --frozen runs/demo/frozen.json `
  --labels runs/demo/vault/sealed_labels.jsonl `
  --manifest runs/demo/blind/manifest.json `
  --out runs/demo/evaluation.json
```

The `vault` directory is generated locally for demonstration. In a real blind evaluation, the predictor must never receive it before predictions are frozen.

## Predicting one script

Train a deployable model from data you are allowed to use:

```powershell
python scripts/fit_text_model.py --input YOUR_LABELLED_DATA.jsonl --out model.json.gz
python scripts/predict_script.py --model model.json.gz --text "你的口播文稿"
```

Treat all outputs as experimental estimates. Do not use them as guarantees of reach, income, or platform performance.

## Methodology

- [Evaluation protocol](references/protocol.md)
- [Input and prediction schema](references/schema.md)
- [Text-only rubric](references/rubric-v0.md)
- [Aggregate pilot calibration report](references/calibration-pilot-v1.md)

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Privacy and research ethics

Only collect or process content you are authorized to use. Keep raw identities, URLs, media, and outcomes outside the public repository. Publish synthetic or properly licensed examples instead of creator data.

## License

MIT. The license covers the source and documentation in this repository, not third-party datasets or platform content.
