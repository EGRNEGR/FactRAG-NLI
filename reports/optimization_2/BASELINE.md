# Baseline (PHASE 1): run 20261031T0508Z, RTX 3090, verifier v2

Recomputed case by case from the run logs by `python -m research.opt2.baseline` (inputs
read in place, never modified; datasets restored from git b058977 and checked against
the sha256 in the run's config.json). All six rows reproduce COMPARISON.md exactly.

Definitions. FactRAG correct: published answer correct (refusal is correct on
unanswerable questions; adversarial: refusal or the corrected fact). Vanilla: the
first LLM draft of the same call, before NLI filtering. Hallucination: on
unanswerable/adversarial questions (n = 269). False refusal: refusals on answerable
questions (n = 1038). Coverage: answerable questions with a published answer.
Selective accuracy: correct among those. Net = FactRAG correct - Vanilla correct;
correction = Vanilla wrong -> FactRAG right; damage = Vanilla right -> FactRAG wrong.
Filter metrics: the verifier's publish decision on drafts that are answers, positive =
draft correct.

## All 1307 questions

| Configuration | FactRAG | Vanilla | Halluc. | False refusal | Net | Corr. | Damage | Filter macro-F1 | MCC | Thinking calls |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen25-14b | 83.6% (1092/1307) | 85.5% (1118/1307) | 3.3% (9/269) | 8.2% (85/1038) | -26 | 14 | 40 | 0.571 | 0.173 | - |
| qwen38-27b + reranker-large | 75.2% (983/1307) | 77.0% (1007/1307) | 0.7% (2/269) | 26.9% (279/1038) | -24 | 0 | 24 | 0.559 | 0.128 | - |
| qwen38-27b (no thinking) | 84.2% (1101/1307) | 85.9% (1123/1307) | 0.4% (1/269) | 16.0% (166/1038) | -22 | 0 | 22 | 0.568 | 0.140 | - |
| qwen38-27b-think (always) | 87.8% (1147/1307) | 89.1% (1165/1307) | 1.5% (4/269) | 10.8% (112/1038) | -18 | 0 | 18 | 0.545 | 0.105 | - |
| hybrid: thinking on first-draft LLM refusal (baseline 89.0%) | 89.0% (1163/1307) | 90.8% (1187/1307) | 0.7% (2/269) | 9.2% (96/1038) | -24 | 0 | 24 | 0.564 | 0.133 | 218 |
| hybrid: thinking on any withheld answer (f888cdf default) | 89.8% (1174/1307) | 90.8% (1187/1307) | 1.1% (3/269) | 7.9% (82/1038) | -13 | 0 | 13 | 0.541 | 0.116 | 242 |

## v3

| Configuration | FactRAG | Vanilla | Halluc. | False refusal | Coverage | Selective acc. |
|---|---|---|---|---|---|---|
| qwen25-14b | 75.5% (166/220) | 77.3% (170/220) | 9.7% (9/93) | 11.8% (15/127) | 88.2% (112/127) | 73.2% (82/112) |
| qwen38-27b + reranker-large | 82.3% (181/220) | 90.0% (198/220) | 2.2% (2/93) | 13.4% (17/127) | 86.6% (110/127) | 81.8% (90/110) |
| qwen38-27b (no thinking) | 84.5% (186/220) | 89.5% (197/220) | 1.1% (1/93) | 14.2% (18/127) | 85.8% (109/127) | 86.2% (94/109) |
| qwen38-27b-think (always) | 88.2% (194/220) | 91.4% (201/220) | 4.3% (4/93) | 7.9% (10/127) | 92.1% (117/127) | 89.7% (105/117) |
| hybrid: thinking on first-draft LLM refusal (baseline 89.0%) | 85.5% (188/220) | 90.5% (199/220) | 2.2% (2/93) | 11.0% (14/127) | 89.0% (113/127) | 85.8% (97/113) |
| hybrid: thinking on any withheld answer (f888cdf default) | 86.8% (191/220) | 90.9% (200/220) | 3.2% (3/93) | 6.3% (8/127) | 93.7% (119/127) | 84.9% (101/119) |

## xquad-full

| Configuration | FactRAG | Vanilla | Halluc. | False refusal | Coverage | Selective acc. |
|---|---|---|---|---|---|---|
| qwen25-14b | 85.2% (926/1087) | 87.2% (948/1087) | 0.0% (0/176) | 7.7% (70/911) | 92.3% (841/911) | 89.2% (750/841) |
| qwen38-27b + reranker-large | 73.8% (802/1087) | 74.4% (809/1087) | 0.0% (0/176) | 28.8% (262/911) | 71.2% (649/911) | 96.5% (626/649) |
| qwen38-27b (no thinking) | 84.2% (915/1087) | 85.2% (926/1087) | 0.0% (0/176) | 16.2% (148/911) | 83.8% (763/911) | 96.9% (739/763) |
| qwen38-27b-think (always) | 87.7% (953/1087) | 88.7% (964/1087) | 0.0% (0/176) | 11.2% (102/911) | 88.8% (809/911) | 96.0% (777/809) |
| hybrid: thinking on first-draft LLM refusal (baseline 89.0%) | 89.7% (975/1087) | 90.9% (988/1087) | 0.0% (0/176) | 9.0% (82/911) | 91.0% (829/911) | 96.4% (799/829) |
| hybrid: thinking on any withheld answer (f888cdf default) | 90.4% (983/1087) | 90.8% (987/1087) | 0.0% (0/176) | 8.1% (74/911) | 91.9% (837/911) | 96.4% (807/837) |

## Dev / test halves (split used by every later experiment)

Grouped by XQuAD paragraph (both languages together), stratified by dataset and
category, 651 dev / 656 test. Parameters are fitted on dev only.

| Configuration | FactRAG dev | FactRAG test |
|---|---|---|
| qwen25-14b | 86.2% (561/651) | 80.9% (531/656) |
| qwen38-27b + reranker-large | 75.9% (494/651) | 74.5% (489/656) |
| qwen38-27b (no thinking) | 84.6% (551/651) | 83.8% (550/656) |
| qwen38-27b-think (always) | 88.6% (577/651) | 86.9% (570/656) |
| hybrid: thinking on first-draft LLM refusal (baseline 89.0%) | 89.9% (585/651) | 88.1% (578/656) |
| hybrid: thinking on any withheld answer (f888cdf default) | 90.6% (590/651) | 89.0% (584/656) |

## Paired against the baseline (FactRAG correct, same 1307 cases)

| Configuration | Gained | Lost | Delta, pp | Bootstrap 95% CI, pp | McNemar p |
|---|---|---|---|---|---|
| qwen25-14b | 55 | 126 | -5.43 | [-7.35, -3.44] | 1.37e-07 |
| qwen38-27b + reranker-large | 7 | 187 | -13.77 | [-15.76, -11.86] | 1.52e-46 |
| qwen38-27b (no thinking) | 1 | 63 | -4.74 | [-5.97, -3.60] | 7.05e-18 |
| qwen38-27b-think (always) | 25 | 41 | -1.22 | [-2.45, +0.00] | 0.064 |
| hybrid: thinking on first-draft LLM refusal (baseline 89.0%) | 0 | 0 | +0.00 | [+0.00, +0.00] | 1 |
| hybrid: thinking on any withheld answer (f888cdf default) | 12 | 1 | +0.84 | [+0.38, +1.38] | 0.00342 |

Reference for the verifier on v3: verifier v2 replay of the 5060 reference drafts
(20261027T0447Z_v3_verifier2) gives FactRAG 78.6%; it is a replay of Qwen2.5-14B drafts
and is not mixed with the RTX 3090 rows above (all of which use verifier v2).
