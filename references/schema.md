# Data schemas

All files use UTF-8 JSON Lines unless noted.

## Collector input

Required identity and text fields:

```json
{"sample_id":"source-local-id","account_id":"account-a","transcript":"...","outcome_metric":"likes","likes":12000}
```

`outcome_metric` must be uniformly `views` or `likes` within one run. If omitted, the tool prefers `views` and otherwise uses `likes`.

Recommended fields: `platform`, `published_at`, `horizon_hours`, `views`, `likes`, `comments`, `shares`, `duration_seconds`, `account_baseline_value`, `account_baseline_views`, `account_baseline_likes`, `content_domain`, `human_driver`, `source_url`, `video_id`, `paid_boost`, and `format_notes`.

`content_domain` and `human_driver` are audit strata, not script-only predictor inputs. Use them to test whether a supposedly universal rule generalizes across domains.

If the matching account baseline is absent, the build tool uses the median outcome value of that account in the supplied file. Supply at least three observations per account.


## Blind input

```json
{"blind_id":"B-2c9f...","transcript":"..."}
```

No public/searchable identifiers or performance fields are allowed.

## Prediction

```json
{"blind_id":"B-2c9f...","rubric_version":"v0","probabilities":{"low":0.2,"medium":0.5,"high":0.3},"predicted_percentile":58,"dimension_scores":{"hook_immediacy":3,"audience_cost_fit":4,"curiosity_gap":3,"specificity_novelty":2,"credibility_in_text":3,"retention_progression":3,"emotional_share_impulse":3,"spoken_clarity":4},"confidence":"low","rationale":"...","failure_mode":"..."}
```

Probabilities must be numbers from 0 to 1 and sum to 1. `predicted_percentile` must be from 0 to 100. The evaluator derives `predicted_bucket` from the largest probability.
