"""GPU duty cycle runner: stage parsing everywhere, freeze/resume on POSIX only."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gpu_duty_cycle", ROOT / "scripts" / "linux" / "gpu_duty_cycle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_parse_stage() -> None:
    cycle = load()
    name, command = cycle.parse_stage("xquad-e2e=python evaluate_suite.py --run-dir 'a b' e2e")
    assert name == "xquad-e2e"
    assert command == ["python", "evaluate_suite.py", "--run-dir", "a b", "e2e"]
    for bad in ("no-command=", "=python x", "python x"):
        with pytest.raises(argparse.ArgumentTypeError):
            cycle.parse_stage(bad)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP/process groups are POSIX-only")
def test_stage_is_frozen_during_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cycle = load()
    monkeypatch.setattr(cycle, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(cycle, "POLL_SECONDS", 0.2)
    marker = tmp_path / "ticks.txt"
    # The stage writes a timestamp every 0.1 s for 1.5 s of its own running time.
    script = (
        "import time\n"
        f"p = open({str(marker)!r}, 'a')\n"
        "for _ in range(15):\n"
        "    p.write(f'{time.monotonic()}\\n'); p.flush(); time.sleep(0.1)\n"
    )
    runner = cycle.DutyCycle(work=0.5, rest=1.0, temp_limit=999.0)
    started = time.monotonic()
    code = asyncio.run(runner.run([("ticks", [sys.executable, "-c", script])], keep_going=False))
    elapsed = time.monotonic() - started
    assert code == 0
    assert runner.status.rests_taken >= 1
    assert elapsed >= 1.5 + 1.0
    ticks = [float(line) for line in marker.read_text().split()]
    assert max(b - a for a, b in zip(ticks, ticks[1:])) >= 0.9, "stage kept running during rest"
    assert (tmp_path / "status.json").exists()
