Dataset: contains_ai_drafts_not_expert_validated

| Metric | Mean | Defined | Undefined |
|---|---:|---:|---:|
| answer_relevance_cosine | 0.8146 | 8 | 7 |
| citation_precision | 0.8750 | 8 | 7 |
| citation_recall | 0.5833 | 12 | 3 |
| faithfulness | 0.7182 | 11 | 4 |
| hybrid_hit_at_k | 0.5000 | 12 | 3 |
| hybrid_mrr_at_k | 0.3750 | 12 | 3 |
| hybrid_ndcg_at_k | 0.3755 | 12 | 3 |
| reference_answer_cosine | 0.7022 | 8 | 7 |
| refusal_correct | 0.7333 | 15 | 0 |
| reranked_hit_at_k | 0.8333 | 12 | 3 |
| reranked_mrr_at_k | 0.6389 | 12 | 3 |
| reranked_ndcg_at_k | 0.6460 | 12 | 3 |

Errors: 0 / 15

Stage 13: 8/12 positive answers, 3/3 negative refusals, reranked Hit@5 = 83.3%. Target 9/12 NOT achieved. In-sample calibration; AI annotations require expert validation. See docs/calibration.md.
