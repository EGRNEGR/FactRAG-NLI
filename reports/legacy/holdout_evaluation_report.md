Dataset: contains_ai_drafts_not_expert_validated

| Metric | Mean | Defined | Undefined |
|---|---:|---:|---:|
| answer_relevance_cosine | 0.7985 | 5 | 5 |
| citation_precision | 1.0000 | 5 | 5 |
| citation_recall | 0.5476 | 7 | 3 |
| faithfulness | 0.7143 | 7 | 3 |
| hybrid_hit_at_k | 0.8571 | 7 | 3 |
| hybrid_mrr_at_k | 0.5286 | 7 | 3 |
| hybrid_ndcg_at_k | 0.5911 | 7 | 3 |
| reference_answer_cosine | 0.8040 | 5 | 5 |
| refusal_correct | 0.8000 | 10 | 0 |
| reranked_hit_at_k | 1.0000 | 7 | 3 |
| reranked_mrr_at_k | 0.8214 | 7 | 3 |
| reranked_ndcg_at_k | 0.7670 | 7 | 3 |

Errors: 0 / 10

Positive non-refusals: 5/7. Negative refusals: 3/3. NLI-supported claims: 11/19 (57.9%). Errors: 0/10. P50 wall latency: 15.31 seconds.

AI-reviewed held-out QA on the same documents, not expert validation. First complete run (2 timeouts) is preserved; full technical rerun uses smaller llama-server batches, unchanged QA and thresholds. See docs/holdout_validation.md.
