"""Compare the API reference and notebooks of the Sphinx docs with this site.

    python docs-site/scripts/compare_api.py

Lists the classes and functions documented by ``docs/source/api/**/*.rst``
(autosummary entries) and by ``docs-site/api/**/*.md`` (``<!-- api -->``
blocks), and prints the names missing on either side. Exits with 1 if a name
of the Sphinx reference is missing here. Useful while both sites exist, e.g.
to port the API entries of a pull request written against the Sphinx docs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPHINX = ROOT / "docs" / "source" / "api"
TEMPLATES = ROOT / "docs-site" / "api"


def sphinx_names() -> set[str]:
    names = set()
    for rst in SPHINX.rglob("*.rst"):
        module = None
        in_autosummary = False
        for line in rst.read_text().splitlines():
            m = re.match(r"^\.\. currentmodule::\s*(\S+)", line)
            if m:
                module = m.group(1)
                in_autosummary = False
                continue
            if re.match(r"^\.\. autosummary::", line):
                in_autosummary = True
                continue
            if in_autosummary:
                s = line.strip()
                if line and not line[0].isspace():
                    in_autosummary = False
                elif s and not s.startswith(":"):
                    names.add(f"{module}.{s}")
    return names


def site_names() -> set[str]:
    names = set()
    block = re.compile(r"^<!--\s*api\s+([\w.]+)[^\n]*-->\n(.*?)^<!--\s*/api\s*-->", re.M | re.S)
    for md in TEMPLATES.rglob("*.md"):
        for m in block.finditer(md.read_text()):
            names |= {f"{m.group(1)}.{n}" for n in m.group(2).split()}
    return names


def notebook_drift() -> list[str]:
    """Notebooks of the Sphinx docs that are missing here or differ."""
    old_dir = ROOT / "docs" / "source" / "tutorials" / "notebooks"
    new_dir = ROOT / "docs-site" / "content" / "tutorials" / "notebooks"
    drift = []
    for nb in sorted(old_dir.glob("*.ipynb")):
        copy = new_dir / nb.name
        if not copy.exists():
            drift.append(f"{nb.name} (missing)")
        elif copy.read_bytes() != nb.read_bytes():
            drift.append(f"{nb.name} (differs)")
    return drift


def main() -> int:
    old, new = sphinx_names(), site_names()
    missing, extra = sorted(old - new), sorted(new - old)
    print(f"Sphinx API reference: {len(old)} names")
    print(f"This site's API templates: {len(new)} names")
    print(f"In both: {len(old & new)}")
    print(f"Missing here ({len(missing)}):", *missing, sep="\n  ")
    print(f"Only here ({len(extra)}):", *extra, sep="\n  ")
    drift = notebook_drift()
    print(f"Notebooks to copy from docs/source ({len(drift)}):", *drift, sep="\n  ")
    return 1 if missing or drift else 0


if __name__ == "__main__":
    sys.exit(main())
