"""Identifies the version of mlx-graphs a benchmark measures.

Results are stored per version, in ``results/<chip>/<label>/``, so that runs
of different versions can be compared (see ``history.py``). The label comes
from the code of the library itself, not from the benchmark harness:

    <version in pyproject.toml>-<short hash of the last commit touching mlx_graphs/>

with ``-dirty`` appended when ``mlx_graphs/`` has uncommitted changes. A
checkout of main and a feature branch with the same harness therefore get
different labels as soon as their library code differs.
"""

import os
import re
import subprocess
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", REPO_ROOT, *args], capture_output=True, text=True
    ).stdout.strip()


def _pyproject_version() -> Optional[str]:
    path = os.path.join(REPO_ROOT, "pyproject.toml")
    try:
        with open(path) as f:
            match = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    except OSError:
        return None
    return match.group(1) if match else None


def _branch() -> Optional[str]:
    """Branch of the checkout; for a detached checkout, the tag it is on
    (e.g. a worktree of a release), else ``None``."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        return branch
    return _git("describe", "--tags", "--exact-match", "HEAD") or None


def library_info() -> dict:
    """Version, commit and branch of the mlx-graphs code in this checkout."""
    commit = _git("log", "-1", "--format=%H", "--", "mlx_graphs")
    dirty = bool(_git("status", "--porcelain", "--", "mlx_graphs"))
    version = _pyproject_version() or "unknown"
    short = commit[:7] if commit else "nogit"
    return {
        "label": f"{version}-{short}" + ("-dirty" if dirty else ""),
        "version": version,
        "commit": commit or None,
        "commit_date": _git("log", "-1", "--format=%cI", commit) if commit else None,
        "commit_subject": _git("log", "-1", "--format=%s", commit) if commit else None,
        "branch": _branch(),
        "dirty": dirty,
        "repo_root": REPO_ROOT,
    }


def worker_env() -> dict:
    """Environment for worker processes: this checkout's ``mlx_graphs`` comes
    first on the path, whatever version is installed in the environment."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    return env


def check_imported_library(python: str) -> str:
    """Path of the ``mlx_graphs`` package a worker imports; raises if it is
    not the one of this checkout (the measured code would not be the one
    named by the label)."""
    out = subprocess.run(
        [
            python,
            "-c",
            "import mlx_graphs, os; print(os.path.dirname(mlx_graphs.__file__))",
        ],
        capture_output=True,
        text=True,
        env=worker_env(),
        cwd=REPO_ROOT,  # as the workers: no other checkout in the current dir
    )
    path = out.stdout.strip()
    expected = os.path.join(REPO_ROOT, "mlx_graphs")
    if out.returncode != 0 or os.path.realpath(path) != os.path.realpath(expected):
        raise RuntimeError(
            f"workers would import mlx_graphs from {path or '?'} instead of "
            f"{expected}: {out.stderr.strip()[-300:]}"
        )
    return path
