# FactRAG-NLI on RTX 3090: 20261030T1814Z_rtx3090_main

## End to end (profile guard_on)

Vanilla = LLM draft before NLI filtering; FactRAG = published answer (same LLM call).
Hallucination: share of answers asserting facts on unanswerable/adversarial questions;
false refusal: refusals on answerable questions. Time: median seconds per generated query.

| Configuration | Dataset | Cases | Errors | Vanilla correct | FactRAG correct | Hallucination | False refusal | Generation, s | Total, s |
|---|---|---|---|---|---|---|---|---|---|
| reference: RTX 5060 8 GB, Qwen2.5-14B (2139Z, verifier v1) | v3 | 220 | 0 | 83.2% (183/220) | 75.5% (166/220) | 4.3% (4/93) | 18.9% (24/127) | 20.5 | 21.4 |
| qwen38-27b+bge-reranker-v2-m3 | v3 | 220 | 0 | 88.6% (195/220) | 82.7% (182/220) | 1.1% (1/93) | 16.5% (21/127) | 4.3 | 4.6 |
| qwen38-27b+bge-reranker-v2-m3 | xquad-full | 1087 | 0 | 81.0% (881/1087) | 79.9% (869/1087) | 0.0% (0/176) | 21.3% (194/911) | 3.9 | 4.2 |
| qwen38-27b+bge-reranker-v2-m3 | all | 1307 | 0 | 82.3% (1076/1307) | 80.4% (1051/1307) | 0.4% (1/269) | 20.7% (215/1038) | - | - |

Verifier v2 replay of the reference drafts (20261027T0447Z_v3_verifier2): FactRAG 78.6%
on v3. The 3090 runs use verifier v2, so compare v3 rows against 78.6% for the verifier
and against the vanilla column for the LLM itself.

## Retrieval (no LLM)

| Dataset + reranker | Recall@5 | MRR@10 | Gate rejects unanswerable | Gate rejects answerable |
|---|---|---|---|---|
| v3+bge-reranker-v2-m3 | 0.837 | 0.769 | 2.1% (1/48) | 0.0% (0/172) |
| xquad-full+bge-reranker-v2-m3 | 0.958 | 0.947 | 94.3% (166/176) | 4.0% (36/911) |

## NLI probes, xquad-full

Catch rate of neutral probes 0.838, false rejection of entailed probes 0.110, status agreement 0.849 (n = 2174).
