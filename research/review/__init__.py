"""Reviewer remarks on the journal article, answered on the final run 20261104T1222Z.

1. External baselines: the same logged first drafts filtered by other claim-level verifiers
   (single NLI filter, BGE-M3-zeroshot alone, MiniCheck) and a claim-level stress set
   (research.review.compare).
2. Agreement of the automatic scorer with the author's review of the v3 set
   (research.opt4.audit_agreement).
3. A second generator: the logged Qwen2.5-14B-Instruct drafts of the prototype run on v3
   under the same verifiers (research.review.second_generator).
4. Paired significance of the full contour against the generator's own draft
   (research.review.significance).

The run is read in place by research.opt3.data (OPT3_RUN) and never written; outputs go to
reports/review. The run folder: REVIEW_RUN if set, else reports/runs/<id> of this checkout,
else the copy on the analysis laptop.
"""

import os
from pathlib import Path

BASELINE_ID = "20261104T1222Z_rtx3090_main"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "review"
CACHE = ROOT / ".artifacts" / "review"
_IN_REPO = ROOT / "reports" / "runs" / BASELINE_ID
_LAPTOP = (
    Path(r"C:\Hakatons\20261103T1310Z_rtx3090_main\20261103T1929Z_rtx3090_main") / BASELINE_ID
)

RUN = Path(os.environ.get("REVIEW_RUN") or (_IN_REPO if _IN_REPO.is_dir() else _LAPTOP))
os.environ["OPT3_RUN"] = str(RUN)
# The replayed settings are the run's (config.json); flags of later stages stay off.
for _name in ("RAG_NLI_TABLE_ROW_WINDOWS", "RAG_NLI_ANCHOR_BIT_CELLS"):
    os.environ.pop(_name, None)
