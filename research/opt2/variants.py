"""Research variants of the verifier (registered into experiments.VARIANTS on import)."""

from __future__ import annotations

from research.opt2.experiments import Variant, ensemble, register, single

# Alternative NLI models as the only classifier (same routes and thresholds as verifier v2).
for model, text in (
    ("bge-m3-zeroshot", "MoritzLaurer/bge-m3-zeroshot-v2.0 (XLM-R large, 2-way) alone"),
    ("xlmr-large-xnli", "joeddav/xlm-roberta-large-xnli alone"),
    ("xlmr-large-xnli-anli", "vicgalle/xlm-roberta-large-xnli-anli alone"),
):
    register(Variant(f"V10_{model}", text, single(model), models=(model,)))

# Pair-level ensembles with the production model.
for model in ("bge-m3-zeroshot", "xlmr-large-xnli", "xlmr-large-xnli-anli"):
    for mode in ("min", "mean"):
        register(
            Variant(
                f"V20_{mode}_mdeberta+{model}",
                f"pair-level {mode} of mDeBERTa and {model}",
                ensemble(("mdeberta", model), mode),
                models=("mdeberta", model),
            )
        )

# Single switches of verifier v2 (mDeBERTa).
for name, overrides, text in (
    ("V01_no_title_route", {"nli_premise_title": False}, "without the 'Title. Section.' premise route"),
    ("V01_no_anchor_windows", {"nli_anchor_windows": False}, "without the anchored-window route"),
    ("V01_no_code_span", {"nli_code_span_min_chars": 0}, "without the verbatim code-span route"),
    ("V01_no_anchor_guard", {"nli_anchor_guard": False}, "without the anchor guard"),
    ("V01_no_co_verification", {"nli_co_verification": False}, "without co-verification"),
    ("V01_window_095", {"nli_window_entailment_threshold": 0.95}, "window entailment 0.95"),
):
    register(Variant(name, text, single("mdeberta"), overrides))

register(
    Variant(
        "V02_keyword_normalization",
        "verifier v2 with RFC 2119 keywords lower-cased in the NLI inputs",
        single("mdeberta"),
        {"nli_keyword_normalization": True},
    )
)
