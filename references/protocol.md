# Validation protocol

## 1. Define the target before collecting data

The text-only model predicts **relative traffic performance**, not raw likes. Preferred primary target:

`reach_ratio = views_at_fixed_horizon / account_baseline_views_at_same_horizon`

If the platform collector exposes likes but not views, use a declared fallback target:

`like_ratio = likes_at_collection_time / account_baseline_likes_at_comparable_age`

Do not mix view-labelled and like-labelled samples in one benchmark run. Likes collected at different post ages are not comparable; record age or collect all outcomes at a fixed horizon.

Default buckets for the pilot:

- low: ratio below 0.65;
- medium: ratio from 0.65 through 1.55;
- high: ratio above 1.55.

Lock the horizon (prefer T+72h), platform, format, language, and account-baseline method for a run. Track like, comment, and share rates separately; do not silently merge them into the traffic label.

## 2. Cohort rules

Include only Chinese talking-head videos with limited effects and no plot-heavy or montage-dependent structure. Record excluded samples and reasons. Do not mix platforms, paid boosts, deleted/reposted videos, or materially different time horizons in one cohort.

Keep presenter identity, follower count, posting time, URL, video ID, and outcome metrics in the raw/vault layer. They are allowed for label normalization and audit, but forbidden as predictor inputs in a script-only run.

## 3. Three data layers

1. `raw`: transcript plus identity, provenance, and all outcomes. Collector/evaluator only.
2. `blind`: anonymous ID plus transcript. Predictor only.
3. `vault`: anonymous-ID mapping plus outcomes. Evaluator only after freeze.

Strong isolation means the blind-test agent cannot access raw or vault files. A folder convention is an audit aid, not a security boundary. For consequential tests, put `blind/` in a separate workspace and keep `vault/` with the evaluator or user.

## 4. Development and tests

- Development set: labels may be shown. Use it to propose rubric hypotheses.
- Backtest holdout: historical labels exist but stay sealed until predictions freeze.
- Prospective test: predict newly collected/published items before T+72h outcomes exist. Treat this as the strongest evidence.

Deduplicate near-identical scripts, reposts, recurring series templates, and the same video's edits across splits. Prefer a time-based split. If multiple creators are used, report both within-creator and held-out-creator results.

## 5. Accuracy claims

Pre-register the metric and baseline. For three balanced buckets, chance accuracy is about 33%. Always compare against:

- majority-bucket baseline;
- account-median baseline;
- previous rubric version.

Do not promise 90%. A 90% bucket score on a small or leaky historical set is more likely to indicate leakage, duplication, or an easy cohort than generalizable script understanding. Require confidence intervals and repeated prospective batches before making a strong claim.

## 6. Learning loop

1. Freeze rubric version and prediction schema.
2. Score blind inputs.
3. Freeze predictions and record hashes.
4. Reveal labels and compute metrics.
5. Cluster errors by mechanism, not by outcome alone.
6. Propose one testable rule change only when repeated evidence supports it.
7. Save a new rubric version and test on a new blind batch.
