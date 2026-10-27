# Bundled pilot calibration

The bundled `talking-head-text-pilot-v1` model was trained on 13 visually verified Chinese Douyin talking-head videos: low 4, medium 5, high 4 relative to each account's median likes.

Two six-item sealed evaluations produced 50% bucket accuracy for v0 and 33.3% for v1. V1 deliberately reduced overconfidence: log loss improved from 13.44 to 1.29 and Brier score from 1.07 to 0.78, while percentile MAE improved from 30.29 to 27.27. Different random splits and the tiny dataset mean these numbers are not directly comparable as a leaderboard.

Always label the model experimental and low confidence. Do not claim 90% accuracy. Prefer the bundled model as a weak probability anchor and use `rubric-v0.md` for evidence-based explanation. The next evidence threshold is at least 60 visually verified videos with 30 never-used blind items.
