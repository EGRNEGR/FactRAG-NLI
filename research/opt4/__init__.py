"""optimization_4: diagnosis of run 20261103T1310Z (PLAN=main, judge, code f083e2d).

The run is the baseline of this stage. Its logs are read in place by research.opt3.data
(OPT3_RUN points there) and never written; outputs go to reports/optimization_4.
The run folder: OPT4_RUN if set, else reports/runs/20261103T1310Z_rtx3090_main of this
checkout (the RTX 3090), else the copy on the analysis laptop.
"""

import os
from pathlib import Path

BASELINE_ID = "20261103T1310Z_rtx3090_main"
_IN_REPO = Path(__file__).resolve().parents[2] / "reports" / "runs" / BASELINE_ID
_LAPTOP = Path(r"C:\Hakatons") / BASELINE_ID

BASELINE_RUN = Path(
    os.environ.get("OPT4_RUN") or (_IN_REPO if _IN_REPO.is_dir() else _LAPTOP)
)
os.environ["OPT3_RUN"] = str(BASELINE_RUN)
