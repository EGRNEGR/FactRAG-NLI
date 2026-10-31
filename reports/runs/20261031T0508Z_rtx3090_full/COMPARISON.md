# FactRAG-NLI on RTX 3090: 20261031T0508Z_rtx3090_full

## End to end (profile guard_on)

Vanilla = LLM draft before NLI filtering; FactRAG = published answer (same LLM call).
Hallucination: share of answers asserting facts on unanswerable/adversarial questions;
false refusal: refusals on answerable questions. Time: median seconds per generated query.

| Configuration | Dataset | Cases | Errors | Vanilla correct | FactRAG correct | Hallucination | False refusal | Generation, s | Total, s |
|---|---|---|---|---|---|---|---|---|---|
| reference: RTX 5060 8 GB, Qwen2.5-14B (2139Z, verifier v1) | v3 | 220 | 0 | 83.2% (183/220) | 75.5% (166/220) | 4.3% (4/93) | 18.9% (24/127) | 20.5 | 21.4 |
| qwen25-14b+bge-reranker-v2-m3 | v3 | 220 | 0 | 77.3% (170/220) | 75.5% (166/220) | 9.7% (9/93) | 11.8% (15/127) | 2.6 | 2.9 |
| qwen25-14b+bge-reranker-v2-m3 | xquad-full | 1087 | 0 | 87.2% (948/1087) | 85.2% (926/1087) | 0.0% (0/176) | 7.7% (70/911) | 2.7 | 3.0 |
| qwen25-14b+bge-reranker-v2-m3 | all | 1307 | 0 | 85.5% (1118/1307) | 83.6% (1092/1307) | 3.3% (9/269) | 8.2% (85/1038) | - | - |
| qwen38-27b+bge-reranker-large | v3 | 220 | 0 | 90.0% (198/220) | 82.3% (181/220) | 2.2% (2/93) | 13.4% (17/127) | 4.4 | 4.8 |
| qwen38-27b+bge-reranker-large | xquad-full | 1087 | 0 | 74.4% (809/1087) | 73.8% (802/1087) | 0.0% (0/176) | 28.8% (262/911) | 3.9 | 4.1 |
| qwen38-27b+bge-reranker-large | all | 1307 | 0 | 77.0% (1007/1307) | 75.2% (983/1307) | 0.7% (2/269) | 26.9% (279/1038) | - | - |
| qwen38-27b+bge-reranker-v2-m3 | v3 | 220 | 0 | 89.5% (197/220) | 84.5% (186/220) | 1.1% (1/93) | 14.2% (18/127) | 4.3 | 4.6 |
| qwen38-27b+bge-reranker-v2-m3 | xquad-full | 1087 | 0 | 85.2% (926/1087) | 84.2% (915/1087) | 0.0% (0/176) | 16.2% (148/911) | 3.9 | 4.2 |
| qwen38-27b+bge-reranker-v2-m3 | all | 1307 | 0 | 85.9% (1123/1307) | 84.2% (1101/1307) | 0.4% (1/269) | 16.0% (166/1038) | - | - |
| qwen38-27b-think+bge-reranker-v2-m3 | v3 | 220 | 0 | 91.4% (201/220) | 88.2% (194/220) | 4.3% (4/93) | 7.9% (10/127) | 20.1 | 20.4 |
| qwen38-27b-think+bge-reranker-v2-m3 | xquad-full | 1087 | 0 | 88.7% (964/1087) | 87.7% (953/1087) | 0.0% (0/176) | 11.2% (102/911) | 12.6 | 12.8 |
| qwen38-27b-think+bge-reranker-v2-m3 | all | 1307 | 0 | 89.1% (1165/1307) | 87.8% (1147/1307) | 1.5% (4/269) | 10.8% (112/1038) | - | - |
| qwen38-27b+bge-reranker-v2-m3, thinking only on refusal (computed) | v3 | 220 | 0 | 90.9% (200/220) | 86.8% (191/220) | 3.2% (3/93) | 6.3% (8/127) | - | - |
| qwen38-27b+bge-reranker-v2-m3, thinking only on refusal (computed) | xquad-full | 1087 | 0 | 90.8% (987/1087) | 90.4% (983/1087) | 0.0% (0/176) | 8.1% (74/911) | - | - |
| qwen38-27b+bge-reranker-v2-m3, thinking only on refusal (computed) | all | 1307 | 0 | 90.8% (1187/1307) | 89.8% (1174/1307) | 1.1% (3/269) | 7.9% (82/1038) | - | - |

Thinking only on refusal, computed from the two logs: v3: thinking call for 77 of 220 cases (the non-thinking answer was withheld). xquad-full: thinking call for 165 of 1087 cases (the non-thinking answer was withheld).

Verifier v2 replay of the reference drafts (20261027T0447Z_v3_verifier2): FactRAG 78.6%
on v3. The 3090 runs use verifier v2, so compare v3 rows against 78.6% for the verifier
and against the vanilla column for the LLM itself.

## Retrieval (no LLM)

| Dataset + reranker | Recall@5 | MRR@10 | Gate rejects unanswerable | Gate rejects answerable |
|---|---|---|---|---|
| v3+bge-reranker-large | 0.852 | 0.777 | 45.8% (22/48) | 2.3% (4/172) |
| v3+bge-reranker-v2-m3 | 0.837 | 0.769 | 0.0% (0/48) | 0.0% (0/172) |
| xquad-full+bge-reranker-large | 0.774 | 0.655 | 59.1% (104/176) | 13.9% (127/911) |
| xquad-full+bge-reranker-v2-m3 | 0.981 | 0.966 | 81.8% (144/176) | 1.4% (13/911) |

## NLI probes, xquad-full

Catch rate of neutral probes 0.838, false rejection of entailed probes 0.117, status agreement 0.843 (n = 2174).
