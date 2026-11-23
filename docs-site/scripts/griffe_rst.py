"""Griffe extension: turn the reStructuredText of the docstrings into Markdown.

The docstrings of mlx-graphs are Google style, written for Sphinx (napoleon):
their prose uses reST roles and directives. mkdocstrings renders docstrings as
Markdown, so this extension rewrites, before griffe parses the sections:

- ``.. code-block:: lang`` and ``paragraph::`` literal blocks -> fenced code
- ``.. math::`` -> ``$$ ... $$`` and ``:math:`x``` -> ``$x$`` (KaTeX)
- ``.. note::`` / ``.. warning::`` / ... -> admonitions
- ``.. list-table::`` -> Markdown tables
- ``:class:`` / ``:func:`` / ``:meth:`` / ... -> cross-references to the API
  pages when the target is documented, inline code otherwise
- ````code```` -> ```code```, ```text <url>`_`` -> ``[text](url)``
- ``References:`` blocks -> a "References" admonition

The documented objects come from ``docs-site/.generated/api_objects.json``,
written by ``gen.py`` before the build. It runs on static analysis only: mlx,
torch, etc. never need to be importable.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import griffe

_INDEX_FILE = Path(__file__).resolve().parents[1] / ".generated" / "api_objects.json"
_index: dict | None = None


def _load_index() -> dict:
    global _index
    if _index is None:
        try:
            _index = json.loads(_INDEX_FILE.read_text())
        except OSError:
            _index = {"ids": [], "short": {}}
        _index["ids"] = set(_index["ids"])
    return _index


# ---------------------------------------------------------------------------
# Block-level conversion
# ---------------------------------------------------------------------------

_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([\w-]+)::\s*(.*)$")
_ADMONITIONS = {
    "note": "note",
    "notes": "note",
    "warning": "warning",
    "tip": "tip",
    "hint": "tip",
    "important": "important",
    "seealso": "info",
    "attention": "warning",
    "caution": "warning",
    "danger": "danger",
}
_SECTION_AS_ADMONITION = {"References": "quote"}


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _take_block(lines: list[str], start: int, base: int) -> tuple[list[str], int]:
    """Lines after ``start`` that are blank or indented deeper than ``base``."""
    block = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line.strip() and _indent_of(line) <= base:
            break
        block.append(line)
        i += 1
    while block and not block[-1].strip():
        block.pop()
        i -= 1
    # keep the trailing blank lines in the output stream
    return block, i


def _strip_options(block: list[str]) -> list[str]:
    """Drop the ``:option: value`` lines at the start of a directive body."""
    out = list(block)
    while out and (not out[0].strip() or re.match(r"^\s*:[\w-]+:", out[0])):
        if not out[0].strip():
            # a blank line ends the options
            out.pop(0)
            break
        out.pop(0)
    return out


def _dedent(block: list[str]) -> list[str]:
    return textwrap.dedent("\n".join(block)).split("\n")


def _list_table(block: list[str], pad: str) -> list[str]:
    header_rows = 0
    for line in block:
        m = re.match(r"^\s*:header-rows:\s*(\d+)", line)
        if m:
            header_rows = int(m.group(1))
    body = _dedent(_strip_options(block))
    rows: list[list[str]] = []
    for line in body:
        if not line.strip():
            continue
        m = re.match(r"^\*\s+-\s*(.*)$", line)
        if m:
            rows.append([m.group(1).strip()])
            continue
        m = re.match(r"^\s+-\s*(.*)$", line)
        if m and rows:
            rows[-1].append(m.group(1).strip())
            continue
        if rows:  # continuation of the last cell
            rows[-1][-1] += " " + line.strip()
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    if header_rows == 0:
        header = [""] * ncols
    else:
        header, rows = rows[0], rows[1:]
    out = [pad + "| " + " | ".join(header) + " |"]
    out.append(pad + "|" + "|".join(["---"] * ncols) + "|")
    for r in rows:
        out.append(pad + "| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
    return out


def convert_blocks(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _DIRECTIVE.match(line)
        if m:
            pad, name, arg = m.group(1), m.group(2).lower(), m.group(3).strip()
            block, end = _take_block(lines, i + 1, len(pad))
            if name in ("code-block", "code", "sourcecode"):
                code = _dedent(_strip_options(block))
                out.append(f"{pad}```{arg or 'python'}")
                out.extend(pad + c if c.strip() else "" for c in code)
                out.append(f"{pad}```")
            elif name == "math":
                body = _dedent(_strip_options(block))
                content = ([arg] if arg else []) + [b for b in body if b.strip()]
                out.append(f"{pad}$$")
                out.extend(pad + c for c in content)
                out.append(f"{pad}$$")
            elif name in _ADMONITIONS:
                body = _dedent(block)
                inner = ([arg] if arg else []) + body
                inner_text = convert_blocks("\n".join(inner))
                out.append(f"{pad}!!! {_ADMONITIONS[name]}")
                out.append("")
                out.extend(
                    pad + "    " + c if c.strip() else "" for c in inner_text.split("\n")
                )
            elif name == "list-table":
                if arg:
                    out.append(f"{pad}**{arg}**")
                    out.append("")
                out.extend(_list_table(block, pad))
            else:  # unknown directive: keep its content as text
                out.extend(_dedent(block))
            i = end
            continue

        # "References:" (not a Google section griffe knows) -> admonition
        sm = re.match(r"^(\s*)(" + "|".join(_SECTION_AS_ADMONITION) + r"):\s*$", line)
        if sm and i + 1 < len(lines) and _indent_of(lines[i + 1]) > len(sm.group(1)):
            pad, title = sm.group(1), sm.group(2)
            block, end = _take_block(lines, i + 1, len(pad))
            body = convert_blocks("\n".join(_dedent(block)))
            out.append(f'{pad}!!! {_SECTION_AS_ADMONITION[title]} "{title}"')
            out.append("")
            out.extend(pad + "    " + c if c.strip() else "" for c in body.split("\n"))
            i = end
            continue

        # "paragraph::" followed by an indented literal block
        if line.rstrip().endswith("::") and not line.strip().startswith(".."):
            pad_len = _indent_of(line)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _indent_of(lines[j]) > pad_len:
                stripped = line.rstrip()[:-2].rstrip()
                if stripped.strip():
                    out.append(stripped + ":")
                    out.append("")
                block, end = _take_block(lines, i + 1, pad_len)
                code = _dedent([b for b in block])
                while code and not code[0].strip():
                    code.pop(0)
                pad = " " * pad_len
                out.append(f"{pad}```python")
                out.extend(pad + c if c.strip() else "" for c in code)
                out.append(f"{pad}```")
                i = end
                continue

        out.append(line)
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Inline conversion
# ---------------------------------------------------------------------------

_ROLE = re.compile(r":(?:py:)?(class|func|meth|attr|obj|mod|data|exc|const):`([^`]+)`")
_MATH_ROLE = re.compile(r":math:`([^`]+)`")
_LINK = re.compile(r"`([^`<]+?)\s*<([^>`]+)>`__?", re.S)
_BARE_LINK = re.compile(r"`<([^>`]+)>`__?")
_DOUBLE_BACKTICKS = re.compile(r"``(.+?)``", re.S)


def _resolve(target: str, context: str) -> str | None:
    """Return the documented id for a reST role target, if any."""
    index = _load_index()
    ids, short = index["ids"], index["short"]
    target = target.lstrip(".~")
    if target.endswith("()"):
        target = target[:-2]
    if target in ids:
        return target
    if target in _MODULES:
        return target  # an anchor of the heading of its API page
    # relative to the current object, its module, its package...
    parts = context.split(".")
    while parts:
        candidate = ".".join(parts + [target])
        if candidate in ids:
            return candidate
        parts.pop()
    # canonical path -> documented alias
    if target in index.get("canonical", {}):
        return index["canonical"][target]
    parts = target.split(".")
    name = parts[-1]
    # a bare name, or a public path of mlx_graphs documented under another one
    if (len(parts) == 1 or parts[0] == "mlx_graphs") and name in short:
        return short[name]
    # Class.method -> the method's anchor on the class page
    if len(parts) >= 2 and parts[-2] in short:
        member = f"{short[parts[-2]]}.{name}"
        if member in ids:
            return member
    return None


# packages -> their API section page
_MODULES = {
    "mlx_graphs.data": "data",
    "mlx_graphs.datasets": "datasets",
    "mlx_graphs.loaders": "loaders",
    "mlx_graphs.transforms": "transforms",
    "mlx_graphs.nn": "nn",
    "mlx_graphs.nn.aggr": "nn",
    "mlx_graphs.nn.attention": "nn",
    "mlx_graphs.utils": "utils",
    "mlx_graphs.algorithms": "algorithms",
    "mlx_graphs.metrics": "metrics",
}


def _role_sub(context: str):
    def sub(m: re.Match) -> str:
        role, content = m.group(1), m.group(2).strip()
        title = None
        tm = re.match(r"^(.*?)\s*<([^>]+)>$", content, re.S)
        if tm:
            title, content = tm.group(1).strip(), tm.group(2).strip()
        target = content.lstrip("!")
        if title is None:
            title = target.lstrip("~.")
            if target.startswith("~"):
                title = title.rsplit(".", 1)[-1]
            if role in ("func", "meth") and not title.endswith(")"):
                title += "()"
        if role in ("obj", "data", "const") and target in ("None", "True", "False"):
            return f"`{target}`"
        ref = _resolve(target.lstrip("~"), context)
        if ref:
            return f"[`{title}`][{ref}]"
        return f"`{title}`"

    return sub


def convert_inline(text: str, context: str) -> str:
    # protect fenced code and display math
    chunks = re.split(r"(^\s*```.*?^\s*```|^\s*\$\$.*?^\s*\$\$)", text, flags=re.S | re.M)
    out = []
    for k, chunk in enumerate(chunks):
        if k % 2 == 1:
            out.append(chunk)
            continue
        chunk = _MATH_ROLE.sub(lambda m: "$" + m.group(1).strip() + "$", chunk)
        chunk = _ROLE.sub(_role_sub(context), chunk)
        chunk = _LINK.sub(
            # reST lets a long URL wrap: its whitespace is not part of it
            lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "](" + re.sub(r"\s+", "", m.group(2)) + ")",
            chunk,
        )
        chunk = _BARE_LINK.sub(lambda m: "<" + m.group(1).strip() + ">", chunk)
        chunk = _DOUBLE_BACKTICKS.sub(lambda m: "`" + m.group(1) + "`", chunk)
        chunk = _escape_citations(chunk)
        out.append(chunk)
    return "".join(out)


_CODE_OR_MATH = re.compile(r"(`+)(?:.+?)\1|\$[^$\n]+\$", re.S)


def _escape_citations(text: str) -> str:
    """``[1]`` citation markers would be read as Markdown reference links."""
    parts, pos = [], 0
    for m in _CODE_OR_MATH.finditer(text):
        parts.append(re.sub(r"(?<![\\\]])\[(\d+)\](?![(\[])", r"\\[\1\\]", text[pos : m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(re.sub(r"(?<![\\\]])\[(\d+)\](?![(\[])", r"\\[\1\\]", text[pos:]))
    return "".join(parts)


def rst_to_markdown(text: str, context: str = "") -> str:
    return convert_inline(convert_blocks(text), context)


class RstToMarkdown(griffe.Extension):
    """Convert the reST markup of every docstring to Markdown."""

    def on_instance(self, *, obj: griffe.Object, **kwargs) -> None:
        if obj.docstring is not None and obj.docstring.value:
            obj.docstring.value = rst_to_markdown(obj.docstring.value, obj.path)
