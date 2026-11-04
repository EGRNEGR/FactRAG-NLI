# FactRAG-NLI on RTX 3090: 20261104T1222Z_rtx3090_main

## End to end (profile guard_on)

Vanilla = LLM draft before NLI filtering; FactRAG = published answer (same LLM call).
Hallucination: share of answers asserting facts on unanswerable/adversarial questions;
false refusal: refusals on answerable questions. Time: median seconds per generated query.

| Configuration | Dataset | Cases | Errors | Vanilla correct | FactRAG correct | Hallucination | False refusal | Generation, s | Total, s |
|---|---|---|---|---|---|---|---|---|---|
| reference: RTX 5060 8 GB, Qwen2.5-14B (2139Z, verifier v1) | v3 | 220 | 0 | 83.2% (183/220) | 75.5% (166/220) | 4.3% (4/93) | 18.9% (24/127) | 20.5 | 21.4 |
| qwen38-27b-hybrid+bge-reranker-v2-m3+judge | v3 | 220 | 0 | 92.7% (204/220) | 90.9% (200/220) | 2.2% (2/93) | 2.4% (3/127) | 5.5 | 6.1 |
| qwen38-27b-hybrid+bge-reranker-v2-m3+judge | xquad-full | 1087 | 0 | 85.5% (929/1087) | 91.4% (993/1087) | 0.0% (0/176) | 7.4% (67/911) | 4.0 | 4.4 |
| qwen38-27b-hybrid+bge-reranker-v2-m3+judge | all | 1307 | 0 | 86.7% (1133/1307) | 91.3% (1193/1307) | 0.7% (2/269) | 6.7% (70/1038) | - | - |

Thinking pass (RAG_LLM_THINK_RETRY, RAG_LLM_THINK_ON_TRIM) ran for: qwen38-27b-hybrid+bge-reranker-v2-m3+judge v3: 83 of 220 cases, 70 of them correct after it. qwen38-27b-hybrid+bge-reranker-v2-m3+judge xquad-full: 171 of 1087 cases, 108 of them correct after it.

Recovery pass (RAG_LLM_RECOVERY) ran for: qwen38-27b-hybrid+bge-reranker-v2-m3+judge v3: 20 cases (2 answerable: 0 answered correctly; 18 unanswerable), 1 published, 18 correct (a refusal is correct on an unanswerable question), 0 hallucinations. qwen38-27b-hybrid+bge-reranker-v2-m3+judge xquad-full: 27 cases (27 answerable: 2 answered correctly; 0 unanswerable), 3 published, 2 correct (a refusal is correct on an unanswerable question), 0 hallucinations.

Verifier v2 replay of the reference drafts (20261027T0447Z_v3_verifier2): FactRAG 78.6%
on v3. The 3090 runs use verifier v2, so compare v3 rows against 78.6% for the verifier
and against the vanilla column for the LLM itself.

## Retrieval (no LLM)

| Dataset + reranker | Recall@5 | MRR@10 | Gate rejects unanswerable | Gate rejects answerable |
|---|---|---|---|---|
| v3+bge-reranker-v2-m3 | 0.907 | 0.824 | 0.0% (0/48) | 0.0% (0/172) |
| xquad-full+bge-reranker-v2-m3 | 0.981 | 0.966 | 81.8% (144/176) | 1.4% (13/911) |

## NLI probes, xquad-full

Catch rate of neutral probes 0.983, false rejection of entailed probes 0.065, status agreement 0.891 (n = 2174).
