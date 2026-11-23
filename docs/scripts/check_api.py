"""Check that the API reference lists every public name, and only existing ones.

    python docs/scripts/check_api.py

The API pages list their classes and functions by hand, in the
``<!-- api <module> -->`` blocks of ``docs/api/**/*.md``. This check fails
(exit code 1) when

- a public name exported by the ``__init__.py`` of a ``mlx_graphs`` subpackage
  is in no block: it prints the name and the block it most likely belongs to
  (the one that lists other names of the same module);
- a block lists a name that does not exist (any more).

Static analysis only (griffe): mlx and torch need not be installed. CI runs
it before building the site (.github/workflows/docs.yml).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import griffe

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen import _BLOCK, ROOT, TEMPLATES, _heading, get_object, load_package  # noqa: E402

# subpackages whose exports are implementation details, not API
SKIP = {"mlx_graphs.nn.kernels"}


def subpackages(pkg: griffe.Module):
    """``mlx_graphs`` and every subpackage (a directory with an __init__.py)."""
    yield pkg
    for member in pkg.members.values():
        if member.is_alias or not member.is_module or member.name.startswith("_"):
            continue
        if member.filepath and Path(member.filepath).name == "__init__.py":
            yield from subpackages(member)


def exports(module: griffe.Module) -> dict[str, str]:
    """Public classes and functions of a package: name -> canonical path."""
    names = {}
    for name, member in module.members.items():
        if name.startswith("_"):
            continue
        try:
            target = member.final_target if member.is_alias else member
        except (griffe.AliasResolutionError, griffe.CyclicAliasError):
            continue
        if not (target.is_class or target.is_function):
            continue
        if not target.path.startswith("mlx_graphs"):
            continue  # re-exported from another library
        names[name] = target.path
    return names


def main() -> int:
    pkg = load_package()

    # what the templates list
    listed: dict[str, tuple[str, str]] = {}  # canonical path -> (page, group)
    by_module: dict[str, Counter] = {}  # module -> votes for (page, group)
    stale = []
    for src in sorted(TEMPLATES.rglob("*.md")):
        text = src.read_text()
        rel = src.relative_to(ROOT).as_posix()
        title = re.search(r"^#\s+(.+)$", text, re.M)
        group = _heading(title.group(1)) if title else rel
        pos = 0
        for m in _BLOCK.finditer(text):
            headings = re.findall(r"^##\s+(.+)$", text[pos : m.start()], re.M)
            if headings:
                group = _heading(headings[-1])
            pos = m.start()
            for name in m.group("names").split():
                path = f"{m.group('module')}.{name}"
                try:
                    obj = get_object(pkg, path)
                except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
                    stale.append(f"  {path}  (listed in {rel}, section '{group}')")
                    continue
                listed[obj.path] = (rel, group)
                # every enclosing module votes for this block
                parts = obj.path.split(".")[:-1]
                for i in range(1, len(parts) + 1):
                    by_module.setdefault(".".join(parts[:i]), Counter())[(rel, group)] += 1

    missing = []
    seen = set()
    for package in subpackages(pkg):
        if package.path in SKIP:
            continue
        for name, path in sorted(exports(package).items()):
            if path in listed or path in seen:
                continue
            seen.add(path)
            # the block listing most names of the closest enclosing module
            module = path.rsplit(".", 1)[0]
            votes = by_module.get(module)
            while votes is None and "." in module:
                module = module.rsplit(".", 1)[0]
                votes = by_module.get(module)
            hint = votes.most_common(1)[0][0] if votes else None
            where = f"{hint[0]}, section '{hint[1]}'" if hint else "a page of docs/api/"
            missing.append(
                f"  {package.path}.{name}  (defined in {path.rsplit('.', 1)[0]})"
                f"\n      -> add it to an <!-- api {package.path} --> block of {where}"
            )

    if stale:
        print(f"API blocks list {len(stale)} name(s) that do not exist:", *stale, sep="\n")
    if missing:
        print(f"{len(missing)} public name(s) are in no API block:", *missing, sep="\n")
    if stale or missing:
        print("\nSee docs/README.md, 'Adding a class or function to the API reference'.")
        return 1
    print(f"API reference: all {len(listed)} listed names exist and every public name is listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
