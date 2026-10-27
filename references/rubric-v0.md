# Rubric v0: cold-start hypotheses

Score each dimension from 0 to 5 using only the transcript. Quote or point to concrete wording for every score. These dimensions begin equally weighted; the weights are not facts and must be calibrated.

| Dimension | What to inspect |
|---|---|
| Hook immediacy | In the first spoken beat, is the audience, consequence, surprise, or tension clear? |
| Audience-cost fit | Does a specific audience recognize a meaningful problem, desire, or identity stake? |
| Curiosity gap | Does the script create a concrete unanswered question without vague clickbait? |
| Specificity and novelty | Are the claim, examples, numbers, distinctions, or framing non-generic? |
| Credibility in text | Does the script contain experience, evidence, mechanism, or bounded reasoning? |
| Retention progression | Does each beat add information, tension, contrast, proof, or resolution? |
| Emotional/share impulse | Is there a safe reason to send, save, quote, debate, or self-identify? |
| Spoken clarity | Are sentences natural aloud, concrete, and easy to follow at normal speed? |

## Guardrails

- Penalize interchangeable AI phrasing, abstract lists, delayed premises, unsupported absolutes, and repeated conclusions.
- Separate broad reach from follow intent. A controversial one-off may score high on reach but low on durable follow value.
- Do not reward claims merely because they sound confident.
- Do not infer visual performance, attractiveness, authority, account distribution, or editing.

## Cold-start mapping

Convert the weighted mean `s` (0–5) to a provisional percentile with `20 × s`. Convert that percentile to a cautious probability distribution rather than a deterministic bucket. Until calibration exists, cap the largest bucket probability at 0.65 and label confidence low.
