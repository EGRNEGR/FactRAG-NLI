# FactRAG-NLI on RTX 3090: 20261030T1420Z_rtx3090_full

## End to end (profile guard_on)

Vanilla = LLM draft before NLI filtering; FactRAG = published answer (same LLM call).
Hallucination: share of answers asserting facts on unanswerable/adversarial questions;
false refusal: refusals on answerable questions. Time: median seconds per generated query.

| Configuration | Dataset | Cases | Errors | Vanilla correct | FactRAG correct | Hallucination | False refusal | Generation, s | Total, s |
|---|---|---|---|---|---|---|---|---|---|
| reference: RTX 5060 8 GB, Qwen2.5-14B (2139Z, verifier v1) | v3 | 220 | 0 | 83.2% (183/220) | 75.5% (166/220) | 4.3% (4/93) | 18.9% (24/127) | 20.5 | 21.4 |
| qwen25-14b+bge-reranker-large | v3 | 220 | 0 | 82.7% (182/220) | 79.1% (174/220) | 6.5% (6/93) | 16.5% (21/127) | 1.4 | 1.8 |
| qwen25-14b+bge-reranker-large | xquad | 198 | 0 | 68.2% (135/198) | 68.2% (135/198) | 0.0% (0/40) | 34.2% (54/158) | 0.9 | 1.2 |
| qwen38-27b+bge-reranker-large | v3 | 220 | 0 | 86.4% (190/220) | 78.6% (173/220) | 5.4% (5/93) | 22.8% (29/127) | 2.6 | 2.9 |
| qwen38-27b+bge-reranker-large | xquad | 198 | 0 | 65.2% (129/198) | 65.2% (129/198) | 0.0% (0/40) | 41.1% (65/158) | 1.4 | 1.7 |
| qwen38-27b+bge-reranker-v2-m3 | v3 | 220 | 152 | 79.4% (54/68) | 75.0% (51/68) | 2.7% (1/37) | 41.9% (13/31) | 2.8 | 3.2 |
| qwen38-27b+bge-reranker-v2-m3 | xquad | 198 | 145 | 73.6% (39/53) | 73.6% (39/53) | 0.0% (0/39) | 100.0% (14/14) | - | - |
| qwen38-27b-think+bge-reranker-large | v3 | 220 | 145 | 86.7% (65/75) | 82.7% (62/75) | 2.4% (1/41) | 23.5% (8/34) | 17.8 | 18.1 |
| qwen38-27b-think+bge-reranker-large | xquad | 198 | 147 | 49.0% (25/51) | 49.0% (25/51) | 0.0% (0/25) | 100.0% (26/26) | - | - |

Verifier v2 replay of the reference drafts (20261027T0447Z_v3_verifier2): FactRAG 78.6%
on v3. The 3090 runs use verifier v2, so compare v3 rows against 78.6% for the verifier
and against the vanilla column for the LLM itself.

## Retrieval (no LLM)

| Dataset + reranker | Recall@5 | MRR@10 | Gate rejects unanswerable | Gate rejects answerable |
|---|---|---|---|---|
| v3+bge-reranker-large | 0.820 | 0.760 | 45.8% (22/48) | 2.3% (4/172) |
| v3+bge-reranker-v2-m3 | 0.799 | 0.756 | 35.4% (17/48) | 1.7% (3/172) |
| xquad+bge-reranker-large | 0.671 | 0.591 | 62.5% (25/40) | 16.5% (26/158) |
| xquad+bge-reranker-v2-m3 | 0.905 | 0.899 | 97.5% (39/40) | 8.9% (14/158) |

## NLI probes, XQuAD

Catch rate of neutral probes 0.838, false rejection of entailed probes 0.098, status agreement 0.851 (n = 396).
