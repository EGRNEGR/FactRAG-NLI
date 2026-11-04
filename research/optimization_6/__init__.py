"""optimization_6: NLI premises and anchors of table chunks (row windows, bit cells).

The baseline is the full run 20261104T1222Z (PARSER=v2 PLAN=main, 1193 / 1307), the measured
configuration of optimization_5. Its logs are read in place by research.opt3.data (OPT3_RUN
points there through research.opt4) and never written; outputs go to reports/optimization_6.
The run folder: OPT6_RUN if set, else reports/runs/20261104T1222Z_rtx3090_main of this
checkout (the RTX 3090), else the copy on the analysis laptop.
"""

import os
from pathlib import Path

BASELINE_ID = "20261104T1222Z_rtx3090_main"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "optimization_6"
_IN_REPO = ROOT / "reports" / "runs" / BASELINE_ID
_LAPTOP = (
    Path(r"C:\Hakatons\20261103T1310Z_rtx3090_main\20261103T1929Z_rtx3090_main") / BASELINE_ID
)

BASELINE_RUN = Path(
    os.environ.get("OPT6_RUN") or (_IN_REPO if _IN_REPO.is_dir() else _LAPTOP)
)
os.environ["OPT4_RUN"] = str(BASELINE_RUN)
os.environ["OPT4_EXPERIMENTS_OUT"] = str(OUT / "offline")
# The replayed settings are the run's (config.json); the new flags must come from overrides only.
for _name in ("RAG_NLI_TABLE_ROW_WINDOWS", "RAG_NLI_ANCHOR_BIT_CELLS"):
    os.environ.pop(_name, None)

# research.opt4 sets OPT3_RUN; it must run before research.opt3.data is first imported.
import research.opt4  # noqa: E402, F401
