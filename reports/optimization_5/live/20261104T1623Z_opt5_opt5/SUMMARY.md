# Experiments 20261104T1623Z_opt5_opt5

Baseline: run 20261104T1222Z_rtx3090_main (judge), paired per question on the same ids.
Population = questions the experiment's trigger selects; untriggered questions keep
the baseline answer by construction.
Estimate from the per-call medians of 1310Z: 6.1 min for the steps.

| Experiment | Questions | Replaced | Paired | McNemar p | Hallucination | False refusal | Complete compound | Lost | New hallucinations | Minutes | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2e, RAG_NLI_ANCHOR_SCOPE=document (real pipeline) | 30 | - | +1 / -0 | 1.000 | 0.0% (0/10) | 5.0% (1/20) | 10 -> 11 of 13 | - | - | 6.7 | ok |
| G2 (H5) | 58 | 1 | +1 / -0 | 1.000 | 0.0% (0/42) | 81.2% (13/16) | 0 -> 1 of 4 | - | - | 4.37 | ok |
| G2 + E1 (H5 + H2, nli_anchor_scope=document) | 58 | 2 | +2 / -0 | 0.500 | 0.0% (0/42) | 81.2% (13/16) | 0 -> 2 of 4 | - | - | 4.4 | ok |

D0 determinism: 6 of 6 first drafts identical to the log.

Projected on all questions (baseline 20261104T1222Z_rtx3090_main):
- G2: all 1307 questions 91.4% (1194/1307) (+1 / -0), hallucination 0.7% (2/269), false refusal 6.7% (70/1038)
- G2 + E1: all 1307 questions 91.4% (1195/1307) (+2 / -0), hallucination 0.7% (2/269), false refusal 6.7% (70/1038)

e2e by split:
- dev: +0 / -0, correct 81.2% (13/16), hallucination 0.0% (0/6), false refusal 10.0% (1/10)
- test: +1 / -0, correct 92.9% (13/14), hallucination 0.0% (0/4), false refusal 0.0% (0/10)

e2e changed answers:
- V3-TC-022: False -> True

Per-question results: e2e/*/e2e_guard_on, steps/<experiment>/cases.jsonl.
