# Experiments 20261103T1801Z_opt4_quick

Baseline: run 20261103T1310Z_rtx3090_main (judge), paired per question on the same ids.
Population = questions the experiment's trigger selects; untriggered questions keep
the baseline answer by construction. Estimate from the per-call medians of 1310Z:
22.5 min for the steps.

| Experiment | Questions | Replaced | Paired | McNemar p | Hallucination | False refusal | Complete compound | Lost | New hallucinations | Minutes | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G3 (H3+H5) | 151 | 5 | +4 / -0 | 0.125 | 0.0% (0/113) | 86.8% (33/38) | 0 -> 0 of 1 | - | - | 7.25 | ok |
| G4 (H1) | 140 | 0 | +0 / -0 | 1.000 | 0.0% (0/78) | 100.0% (62/62) | 0 -> 0 of 4 | - | - | 5.16 | ok |

D0 determinism: 6 of 6 first drafts identical to the log.

Projected on all questions (baseline 1184/1307):
- G3: all 1307 questions 90.9% (1188/1307) (+4 / -0), hallucination 1.1% (3/269), false refusal 7.1% (74/1038)
- G4: all 1307 questions 90.6% (1184/1307) (+0 / -0), hallucination 1.1% (3/269), false refusal 7.6% (79/1038)

Per-question results: e2e/*/e2e_guard_on, steps/<experiment>/cases.jsonl.
