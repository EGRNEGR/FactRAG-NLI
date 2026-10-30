"""Run benchmark stages with GPU rest periods: WORK minutes of work, then REST minutes idle.

Each stage is a command line started in its own process group. Work time accumulates across
stages; once it reaches --work-minutes (or the GPU reaches --temp-limit), the running stage
is frozen with SIGSTOP, the card idles for --rest-minutes and the stage resumes with SIGCONT.
llama-server is not frozen: it finishes the request in flight, so the frozen client finds
the reply in its socket on resume instead of timing out.

Usage:
  python scripts/linux/gpu_duty_cycle.py --work-minutes 45 --rest-minutes 5 \\
      --stage "xquad-e2e=python evaluate_suite.py --dataset ... e2e" --stage "..."

Status for remote monitoring: .artifacts/duty_cycle/status.json (state, stage, timers, GPU).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / ".artifacts" / "duty_cycle" / "status.json"
POLL_SECONDS = 10.0
# A temperature rest is not taken within the first minutes after the previous one.
MIN_WORK_BEFORE_THERMAL_REST = 300.0
if sys.platform != "win32":
    FREEZE, THAW = signal.SIGSTOP, signal.SIGCONT
    STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
else:  # importable for tests; running a plan needs POSIX process groups
    FREEZE = THAW = signal.SIGTERM
    STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@dataclass
class Gpu:
    temperature_c: float | None = None
    power_w: float | None = None
    utilization_pct: float | None = None
    memory_used_mib: float | None = None


@dataclass
class Status:
    state: str = "starting"
    stage: str | None = None
    stage_index: int = 0
    stages: list[str] = field(default_factory=list)
    work_seconds_since_rest: float = 0.0
    total_work_seconds: float = 0.0
    rests_taken: int = 0
    rest_until: str | None = None
    gpu: Gpu = field(default_factory=Gpu)
    updated: str = ""


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


async def read_gpu() -> Gpu:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        values: list[float | None] = []
        for raw in out.decode().strip().split(","):
            try:
                values.append(float(raw))
            except ValueError:
                values.append(None)
        return Gpu(*values[:4])
    except (OSError, asyncio.TimeoutError):
        return Gpu()


def write_status(status: Status) -> None:
    status.updated = datetime.now().isoformat(timespec="seconds")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(asdict(status), ensure_ascii=False, indent=1), encoding="utf-8")
    temp.replace(STATUS)


def gpu_line(gpu: Gpu) -> str:
    return (
        f"GPU {gpu.temperature_c}°C, {gpu.power_w} W, {gpu.utilization_pct}% util, "
        f"{gpu.memory_used_mib} MiB"
    )


class DutyCycle:
    def __init__(self, work: float, rest: float, temp_limit: float) -> None:
        self.work, self.rest, self.temp_limit = work, rest, temp_limit
        self.status = Status()
        self.current: asyncio.subprocess.Process | None = None
        self.stopping = False

    def signal_group(self, sig: signal.Signals) -> None:
        if sys.platform == "win32":
            raise RuntimeError("The duty cycle needs POSIX process groups (Linux)")
        if self.current is not None and self.current.returncode is None:
            try:
                os.killpg(self.current.pid, sig)
            except ProcessLookupError:
                pass

    async def take_rest(self, reason: str) -> None:
        self.signal_group(FREEZE)
        self.status.state = "resting"
        self.status.rests_taken += 1
        end = time.monotonic() + self.rest
        self.status.rest_until = datetime.fromtimestamp(time.time() + self.rest).isoformat(
            timespec="seconds"
        )
        log(f"REST {self.rest / 60:.0f} min ({reason}); stage frozen. {gpu_line(self.status.gpu)}")
        while not self.stopping and (left := end - time.monotonic()) > 0:
            await asyncio.sleep(min(60.0, left))
            self.status.gpu = await read_gpu()
            write_status(self.status)
            log(
                f"  resting, {max(0.0, end - time.monotonic()) / 60:.1f} min left. {gpu_line(self.status.gpu)}"
            )
        self.signal_group(THAW)
        self.status.state = "working"
        self.status.rest_until = None
        self.status.work_seconds_since_rest = 0.0
        log("WORK resumed")

    async def run_stage(self, name: str, command: Sequence[str]) -> int:
        log(f"STAGE {name}: {shlex.join(command)}")
        self.current = await asyncio.create_subprocess_exec(
            *command, cwd=ROOT, start_new_session=True
        )
        self.status.state, self.status.stage = "working", name
        last = time.monotonic()
        while True:
            try:
                code = await asyncio.wait_for(self.current.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                code = None
            now = time.monotonic()
            self.status.work_seconds_since_rest += now - last
            self.status.total_work_seconds += now - last
            last = now
            if code is not None:
                log(f"STAGE {name} finished with exit code {code}")
                return code
            self.status.gpu = await read_gpu()
            write_status(self.status)
            hot = (
                self.status.gpu.temperature_c is not None
                and self.status.gpu.temperature_c >= self.temp_limit
                and self.status.work_seconds_since_rest >= MIN_WORK_BEFORE_THERMAL_REST
            )
            if self.status.work_seconds_since_rest >= self.work:
                await self.take_rest(f"{self.work / 60:.0f} min of work")
            elif hot:
                await self.take_rest(f"GPU at {self.status.gpu.temperature_c}°C")
            last = time.monotonic()

    async def run(self, stages: list[tuple[str, list[str]]], keep_going: bool) -> int:
        loop = asyncio.get_running_loop()

        def interrupt() -> None:
            self.stopping = True
            log("Interrupted: resuming and terminating the current stage")
            self.signal_group(THAW)
            self.signal_group(signal.SIGTERM)

        for sig in STOP_SIGNALS:
            loop.add_signal_handler(sig, interrupt)
        self.status.stages = [name for name, _ in stages]
        log(
            f"Duty cycle: {self.work / 60:.0f} min work / {self.rest / 60:.0f} min rest, "
            f"thermal limit {self.temp_limit}°C; {len(stages)} stage(s)"
        )
        failed = 0
        for index, (name, command) in enumerate(stages):
            if self.stopping:
                break
            self.status.stage_index = index
            code = await self.run_stage(name, command)
            if code != 0:
                failed += 1
                if not keep_going or self.stopping:
                    break
        self.status.state = "interrupted" if self.stopping else ("failed" if failed else "done")
        self.status.stage = None
        write_status(self.status)
        log(
            f"Plan {self.status.state}: {self.status.total_work_seconds / 3600:.2f} h of work, "
            f"{self.status.rests_taken} rest(s)"
        )
        return 130 if self.stopping else (1 if failed else 0)


def parse_stage(text: str) -> tuple[str, list[str]]:
    name, sep, command = text.partition("=")
    if not sep or not name.strip() or not command.strip():
        raise argparse.ArgumentTypeError(f"stage must be NAME=COMMAND, got {text!r}")
    return name.strip(), shlex.split(command)


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--work-minutes", type=float, default=45.0)
    cli.add_argument("--rest-minutes", type=float, default=5.0)
    cli.add_argument("--temp-limit", type=float, default=83.0)
    cli.add_argument("--keep-going", action="store_true", help="Continue after a failed stage")
    cli.add_argument("--stage", type=parse_stage, action="append", required=True)
    args = cli.parse_args(argv)
    if args.work_minutes <= 0 or args.rest_minutes < 0:
        cli.error("--work-minutes must be positive and --rest-minutes non-negative")
    cycle = DutyCycle(args.work_minutes * 60, args.rest_minutes * 60, args.temp_limit)
    return asyncio.run(cycle.run(args.stage, args.keep_going))


if __name__ == "__main__":
    sys.exit(main())
