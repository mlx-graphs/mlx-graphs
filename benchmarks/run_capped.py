"""Runs a command under a hard memory cap.

    python benchmarks/run_capped.py --gb 16 -- python my_benchmark.py

The command is killed (with its children) as soon as its memory footprint
exceeds the cap. The footprint is read from ``top``, which, unlike RSS,
includes the Metal/GPU allocations of MLX and PyTorch MPS on Apple Silicon.

Exit code is the command's own, or 137 when it was killed for memory.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

_UNITS = {"B": 1 / 1024**3, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def footprint_gb(pid: int) -> float:
    """Memory footprint of ``pid`` in GB (0.0 if it cannot be read)."""
    try:
        out = subprocess.run(
            ["top", "-l", "1", "-pid", str(pid), "-stats", "mem"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return 0.0
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    if not lines:
        return 0.0
    value = lines[-1].rstrip("+-")  # e.g. "1532M", "12G+"
    if not value or value[-1] not in _UNITS:
        return 0.0
    try:
        return float(value[:-1]) * _UNITS[value[-1]]
    except ValueError:
        return 0.0


def descendants(pid: int) -> list[int]:
    """All live descendants of ``pid`` (children, grandchildren, ...)."""
    found, frontier = [], [pid]
    while frontier:
        parent = frontier.pop()
        out = subprocess.run(
            ["pgrep", "-P", str(parent)], capture_output=True, text=True
        ).stdout
        children = [int(p) for p in out.split()]
        found += children
        frontier += children
    return found


def run_capped(command: list[str], cap_gb: float, poll_s: float = 1.0) -> int:
    process = subprocess.Popen(command, start_new_session=True)
    peak = 0.0
    try:
        while process.poll() is None:
            pids = [process.pid] + descendants(process.pid)
            used = sum(footprint_gb(pid) for pid in pids)
            peak = max(peak, used)
            if used > cap_gb:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                print(
                    f"[run_capped] killed: {used:.1f} GB > cap of {cap_gb:.1f} GB",
                    file=sys.stderr,
                )
                return 137
            time.sleep(poll_s)
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGKILL)
        raise
    print(f"[run_capped] peak footprint ~{peak:.1f} GB", file=sys.stderr)
    return process.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gb", type=float, default=16.0, help="memory cap in GB")
    parser.add_argument("--poll", type=float, default=1.0, help="poll period in s")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    sys.exit(run_capped(command, args.gb, args.poll))
