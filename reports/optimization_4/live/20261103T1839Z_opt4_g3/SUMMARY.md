# Experiments 20261103T1839Z_opt4_g3

Baseline: run 20261103T1310Z_rtx3090_main (judge), paired per question on the same ids.
Population = questions the experiment's trigger selects; untriggered questions keep
the baseline answer by construction. Estimate from the per-call medians of 1310Z:
- min for the steps.

| Experiment | Questions | Replaced | Paired | McNemar p | Hallucination | False refusal | Complete compound | Lost | New hallucinations | Minutes | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2e, RAG_QUERY_TRANSLATION=refusal (real pipeline) | 171 | - | +4 / -0 | 0.125 | 0.0% (0/119) | 63.5% (33/52) | 8 -> 8 of 9 | - | - | 30.6 | ok |

e2e by split:
- dev: +2 / -0, correct 83.3% (65/78), hallucination 0.0% (0/59), false refusal 57.9% (11/19)
- test: +2 / -0, correct 76.3% (71/93), hallucination 0.0% (0/60), false refusal 66.7% (22/33)

e2e changed answers:
- V3-XQFEN-031: False -> True
- V3-XQFEN-065: False -> True
- V3-XQFEN-245: False -> True
- V3-XQFEN-291: False -> True
- V3-XQFEN-320: False -> False

Per-question results: e2e/*/e2e_guard_on, steps/<experiment>/cases.jsonl.
