"""The research loaders' dev/test split equals the frozen split of optimization_2.

research.opt3.data.split computes the opt2 split on the run's own dataset files (no git, for
the RTX 3090 copy); it must give every question the same half as research.opt2.data.split on
the frozen files of commit b058977. Needs the run logs and git: skipped elsewhere.
"""

from __future__ import annotations

import shutil

import pytest

from research.opt3 import data as opt3
from research.opt2 import data as opt2


@pytest.mark.skipif(
    not opt3.CONFIGS or shutil.which("git") is None, reason="needs the run logs and git"
)
def test_split_on_the_run_datasets_is_the_frozen_split() -> None:
    frozen = opt2.split()
    computed = opt3.split()
    assert computed.keys() == frozen.keys()
    assert computed == frozen
