"""Shared benchmark timing, process, and system utilities."""

import ctypes
import ctypes.util
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Optional

import numpy as np

from benchmarks.versions import REPO_ROOT

FRAMEWORKS = ("mlx-graphs", "pyg-mps", "pyg-cpu")


# ---------------------------------------------------------------------------
# Memory: peak footprint of a process, as shown by Activity Monitor.
# Unlike RSS it includes the Metal (GPU) allocations of MLX and PyTorch MPS.
# ---------------------------------------------------------------------------


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time",
            "ri_serviced_system_time",
            "ri_logical_writes",
            "ri_lifetime_max_phys_footprint",
            "ri_instructions",
            "ri_cycles",
            "ri_billed_energy",
            "ri_serviced_energy",
            "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


_RUSAGE_INFO_V4 = 4
_libproc = None


def _rusage(pid: int) -> Optional[_RUsageInfoV4]:
    global _libproc
    if platform.system() != "Darwin":
        return None
    if _libproc is None:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc"), use_errno=True)
    info = _RUsageInfoV4()
    if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info


def footprint_gb(pid: Optional[int] = None) -> float:
    """Current memory footprint of a process in GB (0.0 if unavailable)."""
    info = _rusage(os.getpid() if pid is None else pid)
    return 0.0 if info is None else info.ri_phys_footprint / 1024**3


def peak_footprint_gb(pid: Optional[int] = None) -> float:
    """Largest memory footprint a process has had so far, in GB."""
    info = _rusage(os.getpid() if pid is None else pid)
    return 0.0 if info is None else info.ri_lifetime_max_phys_footprint / 1024**3


def run_worker(module: str, argv: list[str], memory_gb: float, timeout_s: float):
    """Run a worker module in the checkout and read its final JSON result."""
    with subprocess.Popen(
        [sys.executable, "-m", module, *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    ) as process:
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                try:
                    out, err = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if footprint_gb(process.pid) > memory_gb:
                        return None, f"oom (> {memory_gb:.0f} GB)"
                    if time.monotonic() > deadline:
                        return None, f"timeout (> {timeout_s:.0f} s)"
        finally:
            if process.poll() is None:
                process.kill()

    if process.returncode != 0:
        lines = err.strip().splitlines() or ["?"]
        return None, f"error: {lines[-1][:200]}"
    try:
        return json.loads(out.strip().splitlines()[-1]), "ok"
    except (IndexError, json.JSONDecodeError):
        return None, "error: worker produced no JSON result"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_fn(fn: Callable[[], None], repeats: int, warmup: int) -> dict:
    """Median / quartiles of the wall time of ``fn`` in milliseconds.

    ``fn`` must block until its work is done (evaluate / synchronize inside).
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    q1, median, q3 = np.percentile(times, [25, 50, 75])
    return {"ms": float(median), "ms_q1": float(q1), "ms_q3": float(q3)}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def _version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def system_info() -> dict:
    memory = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gb": round(int(memory) / 1024**3) if memory else None,
        "os": f"macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
        "mlx": _version("mlx"),
        "mlx-graphs": _version("mlx-graphs"),
        "mlx-cluster": _version("mlx_cluster") or _version("mlx-cluster"),
        "torch": _version("torch"),
        "torch_geometric": _version("torch_geometric"),
    }


def random_graph(num_nodes: int, num_edges: int, seed: int = 0) -> np.ndarray:
    """Uniform random ``[2, num_edges]`` edge index (int64)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_nodes, (2, num_edges), dtype=np.int64)
