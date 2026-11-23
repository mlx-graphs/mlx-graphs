"""Generate the pages of the documentation site that come from other sources.

Run it before ``zensical build`` / ``zensical serve``::

    python docs-site/scripts/gen.py

It writes (all git-ignored):

- the API reference, from the templates in ``docs-site/api/``: one page per
  public class or function (rendered by mkdocstrings from the docstrings),
  the section pages with their summary tables, and the API sidebar
  (``content/api/SUMMARY.md``, read by the literate-nav plugin);
- the tutorial notebooks as Markdown pages with their stored outputs (they are
  never executed);
- the changelog, from ``CHANGELOG.rst``;
- the per-machine sections of the benchmarks page, from the ``results.rst``
  files written by ``benchmarks/pyg_comparison/plot.py``.

Only static analysis is used (griffe): mlx, torch, etc. need not be installed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import griffe
import yaml

HERE = Path(__file__).resolve().parent
SITE = HERE.parent  # docs-site/
ROOT = SITE.parent  # repository root
CONTENT = SITE / "content"
TEMPLATES = SITE / "api"
GENERATED = SITE / ".generated"
API_OUT = CONTENT / "api"
NB_DIR = CONTENT / "tutorials" / "notebooks"
GITHUB = "https://github.com/mlx-graphs/mlx-graphs"

sys.path.insert(0, str(HERE))
from griffe_rst import convert_inline, rst_to_markdown  # noqa: E402

# Order of the API sections in the sidebar, and of the sub-pages of a section.
# A section is a directory of docs-site/api/ with an index.md; its other pages
# are listed here (the ones not listed come after, alphabetically).
API_SECTIONS = {
    "data": ["data", "batch", "utils"],
    "datasets": [],
    "loaders": [],
    "transforms": [],
    "nn": [],
    "utils": [
        "transformations",
        "graph",
        "generators",
        "sparse",
        "topology",
        "sorting",
        "validators",
        "scatter",
        "array_ops",
        "convert",
        "spatial",
    ],
    "algorithms": [],
    "metrics": [],
}

# Packages whose exported names must all be documented: a public name that is
# in none of the templates is added to an "Other" group of its section, so new
# classes and functions show up without editing the templates.
AUTO_PACKAGES = {
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

# mkdocstrings options per kind of page (as the Sphinx templates did)
OPTIONS = {
    # classes: own public members, __call__, bases, a summary of the methods
    "class": {"filters": ["!^_", "^__call__$"]},
    # GraphData & co: also the batching hooks
    "data": {"filters": ["!^_", "^__call__$", "^__cat_dim__$", "^__inc__$"]},
    # the built-in datasets: like classes (the Sphinx guide page showed their
    # members, the Sphinx API page did not; here the API page does)
    "datasets": {"filters": ["!^_", "^__call__$"]},
    "function": {},
}

_BLOCK = re.compile(
    r"^<!--\s*api\s+(?P<module>[\w.]+)(?P<args>[^\n]*?)-->\n(?P<names>.*?)^<!--\s*/api\s*-->\n?",
    re.M | re.S,
)
_REST = re.compile(r"^<!--\s*api-rest\s*-->\n?", re.M)


@dataclass
class Entry:
    module: str
    name: str
    template: str | None
    group: str
    page: Path  # relative to content/api
    obj: griffe.Object | None = None

    @property
    def id(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class Page:
    src: Path  # template, relative to docs-site/api
    title: str = ""
    entries: list[Entry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# API reference
# ---------------------------------------------------------------------------


def load_package() -> griffe.Module:
    loader = griffe.GriffeLoader(search_paths=[str(ROOT)], allow_inspection=False)
    pkg = loader.load("mlx_graphs")
    loader.resolve_aliases(implicit=False, external=False)
    return pkg


def get_object(pkg: griffe.Module, path: str) -> griffe.Object:
    obj = pkg
    for part in path.split(".")[1:]:
        obj = obj.members[part]
    return obj.final_target if obj.is_alias else obj


def summary_of(obj: griffe.Object) -> str:
    if not obj.docstring or not obj.docstring.value.strip():
        return ""
    paragraph = obj.docstring.value.strip().split("\n\n")[0]
    paragraph = " ".join(line.strip() for line in paragraph.splitlines())
    text = convert_inline(paragraph, obj.path)
    # first sentence only, without breaking inside links/code/math
    m = re.match(r"^(.+?[.!?])(\s+[A-Z(]|$)", text)
    if m and m.group(1).count("`") % 2 == 0 and m.group(1).count("$") % 2 == 0:
        text = m.group(1)
    return text.replace("|", "\\|")


def kind_of(obj: griffe.Object) -> str:
    return "class" if obj.is_class else "function"


def _heading(text: str) -> str:
    """Heading text without its ``{ #anchor }`` attributes."""
    return re.sub(r"\s*\{[^}]*\}\s*$", "", text).strip()


def parse_templates(pkg: griffe.Module) -> tuple[list[Page], dict[str, Entry]]:
    pages: list[Page] = []
    entries: dict[str, Entry] = {}
    for src in sorted(TEMPLATES.rglob("*.md")):
        rel = src.relative_to(TEMPLATES)
        page = Page(src=rel)
        text = src.read_text()
        title = re.search(r"^#\s+(.+)$", text, re.M)
        page.title = _heading(title.group(1)) if title else rel.stem
        group = page.title
        pos = 0
        for m in _BLOCK.finditer(text):
            headings = re.findall(r"^##\s+(.+)$", text[pos : m.start()], re.M)
            if headings:
                group = _heading(headings[-1])
            pos = m.start()
            template = None
            tm = re.search(r"template=(\w+)", m.group("args"))
            if tm:
                template = tm.group(1)
            for name in m.group("names").split():
                entry = Entry(
                    module=m.group("module"),
                    name=name,
                    template=template,
                    group=group,
                    page=rel.parent / f"{name}.md",
                )
                try:
                    entry.obj = get_object(pkg, entry.id)
                except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError) as e:
                    raise SystemExit(f"{src}: cannot find {entry.id} ({e!r})")
                if entry.id in entries:
                    raise SystemExit(f"{src}: {entry.id} is listed twice")
                entries[entry.id] = entry
                page.entries.append(entry)
        pages.append(page)
    return pages, entries


def public_exports(pkg: griffe.Module, module: str) -> list[str]:
    mod = get_object(pkg, module)
    names = []
    for name, member in mod.members.items():
        if name.startswith("_") or member.is_module:
            continue
        try:
            target = member.final_target if member.is_alias else member
        except (griffe.AliasResolutionError, griffe.CyclicAliasError):
            continue
        if target.is_module or not (target.is_class or target.is_function):
            continue
        if not target.path.startswith("mlx_graphs"):
            continue  # re-exported from another library
        names.append(name)
    return names


def add_undocumented(pkg, pages, entries) -> list[Entry]:
    """Public names of AUTO_PACKAGES that no template lists."""
    documented = {e.obj.path for e in entries.values()} | {e.name for e in entries.values()}
    by_section: dict[str, Page] = {
        str(p.src.parent): p for p in pages if p.src.name == "index.md"
    }
    added = []
    for module, section in AUTO_PACKAGES.items():
        for name in public_exports(pkg, module):
            obj = get_object(pkg, f"{module}.{name}")
            if obj.path in documented or name in documented:
                continue
            page = by_section[section]
            entry = Entry(
                module=module,
                name=name,
                template=None,
                group="Other",
                page=Path(section) / f"{name}.md",
                obj=obj,
            )
            entries[entry.id] = entry
            page.entries.append(entry)
            documented |= {obj.path, name}
            added.append(entry)
    return added


def table(entries: list[Entry], page_dir: Path) -> str:
    kinds = {kind_of(e.obj) for e in entries}
    head = {"class": "Class", "function": "Function"}[kinds.pop()] if len(kinds) == 1 else "Name"
    lines = [f"| {head} | Description |", "|---|---|"]
    for e in entries:
        link = Path(*([".."] * len(page_dir.parts))) / e.page if page_dir.parts else e.page
        call = "()" if kind_of(e.obj) == "function" else ""
        lines.append(f"| [`{e.name}{call}`]({link.as_posix()}) | {summary_of(e.obj)} |")
    return "\n".join(lines) + "\n"


def render_page(page: Page, entries: dict[str, Entry], added: list[Entry]) -> str:
    text = (TEMPLATES / page.src).read_text()
    page_dir = page.src.parent

    def block(m: re.Match) -> str:
        names = m.group("names").split()
        return table([entries[f"{m.group('module')}.{n}"] for n in names], page_dir)

    text = _BLOCK.sub(block, text)
    rest = [e for e in added if e in page.entries]

    def rest_block(_: re.Match) -> str:
        if not rest:
            return ""
        return (
            "## Other\n\nPublic names not yet sorted into a group above.\n\n"
            + table(rest, page_dir)
        )

    if _REST.search(text):
        text = _REST.sub(rest_block, text)
    elif rest:
        text += "\n" + rest_block(None)

    def count(m: re.Match) -> str:
        section = m.group(1)
        return str(sum(1 for e in entries.values() if e.page.parts[0] == section))

    return re.sub(r"<!--\s*count\s+(\w+)\s*-->", count, text)


def object_page(entry: Entry) -> str:
    kind = entry.template or kind_of(entry.obj)
    options = dict(OPTIONS.get(kind, {}))
    lines = [
        "---",
        f"title: {entry.name}",
        "---",
        "",
        f"::: {entry.id}",
    ]
    if options:
        lines.append("    options:")
        for key, value in options.items():
            lines.append(f"      {key}: {json.dumps(value)}")
    source = entry.obj.relative_package_filepath.as_posix()
    lines += [
        "",
        f'<small class="mg-source">`from {entry.module} import {entry.name}` &middot; '
        f"[:fontawesome-brands-github: `{source}`]({GITHUB}/blob/main/{source}#L{entry.obj.lineno})</small>",
        "",
    ]
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    """Write only when the content changes, so the live server stays quiet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def nav(pages: list[Page]) -> str:
    by_src = {p.src.as_posix(): p for p in pages}
    lines = ["- [Overview](index.md)"]

    def objects(page: Page, indent: str) -> None:
        groups: dict[str, list[Entry]] = {}
        for e in page.entries:
            groups.setdefault(e.group, []).append(e)
        single = len(groups) == 1
        for group, items in groups.items():
            inner = indent
            if not single or group != page.title:
                lines.append(f"{indent}- {group}")
                inner = indent + "    "
            for e in items:
                lines.append(f"{inner}- [{e.name}]({e.page.as_posix()})")

    for section, subpages in API_SECTIONS.items():
        index = by_src.get(f"{section}/index.md")
        if index is None:
            continue
        lines.append(f"- [{index.title}]({section}/index.md)")
        objects(index, "    ")
        rest = sorted(
            p.src.stem for p in pages if p.src.parent.as_posix() == section and p.src.stem != "index"
        )
        for sub in subpages + [s for s in rest if s not in subpages]:
            page = by_src[f"{section}/{sub}.md"]
            lines.append(f"    - {page.title}")
            lines.append(f"        - [Overview]({section}/{sub}.md)")
            objects(page, "        ")
    return "\n".join(lines) + "\n"


def api_index(pages: list[Page], entries: dict[str, Entry]) -> dict:
    """Ids the cross-references of the docstrings may point to."""
    ids, short, canonical = set(), {}, {}
    seen_short: dict[str, int] = {}
    for e in entries.values():
        ids.add(e.id)
        canonical.setdefault(e.obj.path, e.id)
        seen_short[e.name] = seen_short.get(e.name, 0) + 1
        short[e.name] = e.id
        if e.obj.is_class:
            # the members the page renders: own, public (or __call__), documented
            for name, member in e.obj.members.items():
                if name.startswith("_") and name != "__call__":
                    continue
                try:
                    documented = bool(member.docstring and member.docstring.value.strip())
                except (griffe.AliasResolutionError, griffe.CyclicAliasError):
                    documented = False
                if documented:
                    ids.add(f"{e.id}.{name}")
    short = {k: v for k, v in short.items() if seen_short[k] == 1}
    return {"ids": sorted(ids), "short": short, "canonical": canonical}


def generate_api(check: bool = False) -> dict[str, Entry]:
    pkg = load_package()
    pages, entries = parse_templates(pkg)
    added = add_undocumented(pkg, pages, entries)
    for e in added:
        print(f"  note: {e.id} is public but in no API template; listed under 'Other'")

    # case-insensitive file systems (macOS) cannot hold Foo.md and foo.md
    seen: dict[str, Entry] = {}
    for e in entries.values():
        key = e.page.as_posix().lower()
        if key in seen:
            raise SystemExit(f"{e.id} and {seen[key].id} would share the page {e.page}")
        seen[key] = e

    GENERATED.mkdir(parents=True, exist_ok=True)
    write(GENERATED / "api_objects.json", json.dumps(api_index(pages, entries), indent=0))

    # remove pages of objects that are gone
    wanted = {API_OUT / p.src for p in pages} | {API_OUT / e.page for e in entries.values()}
    wanted.add(API_OUT / "SUMMARY.md")
    if API_OUT.exists():
        for old in API_OUT.rglob("*.md"):
            if old not in wanted:
                old.unlink()
    for page in pages:
        write(API_OUT / page.src, render_page(page, entries, added))
    for e in entries.values():
        write(API_OUT / e.page, object_page(e))
    write(API_OUT / "SUMMARY.md", nav(pages))
    print(f"api: {len(entries)} objects on {len(pages)} section pages")
    return entries


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _text(value) -> str:
    return "".join(value) if isinstance(value, list) else value


def _clean_stream(text: str) -> str:
    text = _ANSI.sub("", text)
    # progress bars: keep the last state of each carriage-returned line
    lines = [line.rsplit("\r", 1)[-1] for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n")


def _fence(text: str, lang: str = "text") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


# links of the notebooks to pages of the Sphinx site
SPHINX_LINKS = {"../../validation.rst": "../../validation.md"}


def _gh_alerts(text: str) -> str:
    """GitHub alerts (``> [!WARNING]``) -> admonitions."""

    def sub(m: re.Match) -> str:
        kind = m.group(1).lower()
        kind = {"caution": "danger"}.get(kind, kind)
        body = re.sub(r"^>\s?", "", m.group(2), flags=re.M)
        return f"!!! {kind}\n\n" + "\n".join("    " + line for line in body.split("\n")) + "\n"

    return re.sub(
        r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*(?:\n|$))*)", sub, text, flags=re.M
    )


def convert_notebook(path: Path) -> str:
    nb = json.loads(path.read_text())
    name = path.stem
    files = path.parent / f"{name}_files"
    if files.exists():
        shutil.rmtree(files)
    lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
    repo_path = f"docs-site/content/tutorials/notebooks/{path.name}"
    out = []
    n_images = 0
    first_heading_done = False
    for cell in nb["cells"]:
        source = _text(cell.get("source", ""))
        if cell["cell_type"] == "markdown":
            # images of the Sphinx site -> copies in assets/
            source = source.replace("../../_static/tutorials/", "../../assets/tutorials/")
            # links to other notebooks and to the Sphinx pages -> these pages
            source = re.sub(r"\]\((\w+)\.ipynb(#[^)]*)?\)", r"](\1.md\2)", source)
            for old, new in SPHINX_LINKS.items():
                source = source.replace(f"]({old})", f"]({new})")
            source = _gh_alerts(source)
            for att_name, att in (cell.get("attachments") or {}).items():
                for mime, data in att.items():
                    ext = mime.split("/")[-1].replace("svg+xml", "svg")
                    files.mkdir(exist_ok=True)
                    target = files / f"attachment_{n_images}.{ext}"
                    n_images += 1
                    target.write_bytes(base64.b64decode(data))
                    source = source.replace(
                        f"attachment:{att_name}", f"{name}_files/{target.name}"
                    )
            if not first_heading_done and re.match(r"^#\s", source):
                # links to the notebook, right under its title
                first_heading_done = True
                title, _, rest = source.partition("\n")
                out.append(title)
                out.append(
                    f'<div class="mg-nb-links" markdown>\n'
                    f"[:fontawesome-brands-github: View on GitHub]({GITHUB}/blob/main/{repo_path}){{ .md-button }}\n"
                    f"[:lucide-download: Download notebook]({path.name}){{ .md-button }}\n"
                    f"</div>"
                )
                source = rest
            out.append(source)
            continue
        if cell["cell_type"] == "raw":
            out.append(source)
            continue
        if not source.strip():
            continue
        out.append(f'<div class="mg-nb-input" markdown>\n\n{_fence(source, lang)}\n\n</div>')
        outputs = []
        for o in cell.get("outputs", []):
            kind = o["output_type"]
            if kind == "stream":
                text = _clean_stream(_text(o["text"]))
                if text.strip():
                    cls = "mg-nb-stderr" if o.get("name") == "stderr" else ""
                    outputs.append((cls, _fence(text)))
            elif kind == "error":
                tb = _ANSI.sub("", "\n".join(o.get("traceback", [])))
                outputs.append(("mg-nb-stderr", _fence(tb)))
            elif kind in ("display_data", "execute_result"):
                data = o.get("data", {})
                if "image/png" in data or "image/jpeg" in data or "image/svg+xml" in data:
                    mime = next(m for m in ("image/svg+xml", "image/png", "image/jpeg") if m in data)
                    files.mkdir(exist_ok=True)
                    ext = {"image/png": "png", "image/jpeg": "jpg", "image/svg+xml": "svg"}[mime]
                    target = files / f"output_{n_images}.{ext}"
                    n_images += 1
                    if ext == "svg":
                        target.write_text(_text(data[mime]))
                    else:
                        target.write_bytes(base64.b64decode(_text(data[mime])))
                    alt = _text(data.get("text/plain", "output")).strip().replace("\n", " ")
                    alt = re.sub(r"[\[\]<>]", "", alt)[:120] or "output"
                    outputs.append(("mg-nb-image", f"![{alt}]({name}_files/{target.name})"))
                elif "text/html" in data:
                    html = _text(data["text/html"]).strip()
                    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.S)
                    outputs.append(("mg-nb-html", html))
                elif "text/markdown" in data:
                    outputs.append(("", _text(data["text/markdown"])))
                elif "text/latex" in data:
                    outputs.append(("", _text(data["text/latex"])))
                elif "text/plain" in data:
                    outputs.append(("", _fence(_ANSI.sub("", _text(data["text/plain"])))))
        for cls, body in outputs:
            if cls == "mg-nb-html":
                out.append(f'<div class="mg-nb-output {cls}">\n{body}\n</div>')
            else:
                out.append(f'<div class="mg-nb-output {cls}" markdown>\n\n{body}\n\n</div>')
    front = (
        "---\n"
        f"edit_url: {GITHUB}/edit/main/{repo_path}\n"
        "---\n\n"
    )
    return front + "\n\n".join(out) + "\n"


def tutorial_cards(tutorials: list[dict], page: str) -> str:
    """Cards linking the notebooks, for a snippet included by ``page``."""
    base = Path(page).parent

    def rel(target: str) -> str:
        return Path(os.path.relpath(target, base)).as_posix()

    cards = []
    for t in tutorials:
        if t.get("thumbnail"):
            thumb = f'![]({rel("assets/tutorials/" + t["thumbnail"])}){{ loading=lazy }}'
        else:
            icon = t.get("icon", "lucide/notebook").replace("/", "-")
            thumb = f":{icon}:"
        tags = "".join(f"<span>{tag}</span>" for tag in t.get("tags", []))
        cards.append(
            '<div class="mg-tutorial" markdown>\n\n'
            f'<div class="mg-tutorial__thumb" markdown>{thumb}</div>\n\n'
            f'<div class="mg-tutorial__body" markdown>\n\n'
            f'[{t["title"]}]({rel("tutorials/notebooks/" + t["notebook"] + ".md")}){{ .mg-tutorial__title }}\n\n'
            f'{t["text"]}\n\n'
            f'<div class="mg-tags">{tags}</div>\n\n'
            "</div>\n\n</div>"
        )
    return '<div class="mg-tutorials" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>\n"


def generate_notebooks() -> None:
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    for nb in notebooks:
        write(nb.with_suffix(".md"), convert_notebook(nb))
    print(f"notebooks: {len(notebooks)} converted (not executed)")
    tutorials = yaml.safe_load((SITE / "tutorials.yml").read_text())
    missing = {t["notebook"] for t in tutorials} - {nb.stem for nb in notebooks}
    if missing:
        raise SystemExit(f"tutorials.yml lists missing notebooks: {sorted(missing)}")
    write(GENERATED / "tutorial_cards.md", tutorial_cards(tutorials, "tutorials/index.md"))
    home = [t for t in tutorials if t.get("home")]
    write(GENERATED / "tutorial_cards_home.md", tutorial_cards(home, "index.md"))


# ---------------------------------------------------------------------------
# Changelog and benchmarks (reST written by other tools)
# ---------------------------------------------------------------------------

_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def simple_rst_to_md(text: str, image_dir: str | None = None, image_prefix: str = "") -> str:
    """reST of the changelog / benchmark sections -> Markdown.

    Handles headings (levels in order of appearance), bullet lists (re-indented
    for Python-Markdown), images (with the only-light / only-dark classes) and
    the inline markup handled by griffe_rst.
    """
    lines = text.split("\n")
    levels: list[str] = []
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 1:
            char = nxt.strip()[0]
            if char not in levels:
                levels.append(char)
            out.append("#" * (levels.index(char) + 1) + " " + line.strip())
            i += 2
            continue
        m = re.match(r"^\.\.\s+image::\s*(\S+)", line)
        if m:
            src, alt, cls = m.group(1), "", ""
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")) and lines[i].strip():
                om = re.match(r"^\s+:(\w+):\s*(.*)$", lines[i])
                if om and om.group(1) == "alt":
                    alt = om.group(2)
                elif om and om.group(1) == "class":
                    cls = om.group(2)
                i += 1
            name = src.rsplit("/", 1)[-1]
            suffix = "#only-light" if "only-light" in cls else "#only-dark" if "only-dark" in cls else ""
            out.append(f"![{alt}]({image_prefix}{name}{suffix})")
            continue
        if re.match(r"^\.\.\s", line) or (line.startswith("..") and not line.strip(".")):
            # comments and other directives: skip with their body
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                if not lines[i].strip() and i + 1 < len(lines) and not lines[i + 1].startswith((" ", "\t")):
                    break
                i += 1
            continue
        indent = len(line) - len(line.lstrip())
        out.append(" " * (indent * 2) + line.lstrip() if indent else line)
        i += 1
    return convert_inline("\n".join(out), "mlx_graphs")


def generate_changelog() -> None:
    md = simple_rst_to_md((ROOT / "CHANGELOG.rst").read_text())
    front = (
        "---\n"
        f"edit_url: {GITHUB}/edit/main/CHANGELOG.rst\n"
        "---\n\n"
        "<!-- Generated by docs-site/scripts/gen.py from CHANGELOG.rst -->\n\n"
    )
    write(CONTENT / "changelog.md", front + md)
    version = latest_release(md)
    write(
        GENERATED / "version.md",
        f'<a class="mg-badge" href="{GITHUB}/releases/tag/v{version}">'
        f'<span class="mg-badge__dot"></span>Latest release: v{version}</a>\n',
    )
    print(f"changelog: converted from CHANGELOG.rst (latest release {version})")


def latest_release(changelog_md: str) -> str:
    """The last release tag (``git describe``), else the changelog's last version."""
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return tag.lstrip("v")
    except (OSError, subprocess.CalledProcessError):
        m = re.search(r"^## (\d+\.\d+\.\d+)", changelog_md, re.M)
        return m.group(1) if m else "dev"


def generate_benchmarks() -> None:
    """Sections and figures of the results the pages refer to.

    A page includes a machine's section with
    ``--8<-- "benchmarks/<machine>/<version>.md"`` or shows one of its figures
    from ``assets/benchmarks/<machine>/<version>/``.
    """
    results = ROOT / "benchmarks" / "pyg_comparison" / "results"
    used = set()
    for page in CONTENT.rglob("*.md"):
        text = page.read_text()
        used |= set(re.findall(r'--8<--\s+"benchmarks/([^/"]+)/([^/"]+)\.md"', text))
        used |= set(re.findall(r"assets/benchmarks/([^/\s)]+)/([^/\s)]+)/", text))
    count = 0
    for machine, version in sorted(used):
        rst = results / machine / version / "results.rst"
        if not rst.exists():
            raise SystemExit(f"no benchmark results in {rst.parent}")
        figures = CONTENT / "assets" / "benchmarks" / machine / version
        figures.mkdir(parents=True, exist_ok=True)
        for png in (rst.parent / "figures").glob("*.png"):
            target = figures / png.name
            if not target.exists() or target.stat().st_mtime < png.stat().st_mtime:
                shutil.copy2(png, target)
        # included by benchmarks.md, so paths are relative to it
        prefix = f"assets/benchmarks/{machine}/{version}/"
        md = simple_rst_to_md(rst.read_text(), image_prefix=prefix)
        md = re.sub(r"^(#+) ", r"#\1 ", md, flags=re.M)  # sections are h2 on the page
        write(GENERATED / "benchmarks" / machine / f"{version}.md", md)
        count += 1
    print(f"benchmarks: {count} result sections, figures copied to assets/benchmarks/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only",
        choices=["api", "notebooks", "changelog", "benchmarks"],
        action="append",
        help="generate only these parts (default: all)",
    )
    args = parser.parse_args()
    parts = args.only or ["api", "notebooks", "changelog", "benchmarks"]
    if "api" in parts:
        generate_api()
    if "notebooks" in parts:
        generate_notebooks()
    if "changelog" in parts:
        generate_changelog()
    if "benchmarks" in parts:
        generate_benchmarks()
    if args.only is None:
        check_nav()


def check_nav() -> None:
    """Zensical does not report nav entries whose page is missing: check them."""
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        return
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]

    def pages(item):
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                yield from pages(value)
        elif isinstance(item, list):
            for value in item:
                yield from pages(value)

    missing = [
        p for p in pages(config["nav"])
        if not p.startswith("http") and not (CONTENT / p).exists()
    ]
    if missing:
        raise SystemExit(f"zensical.toml nav lists missing pages: {missing}")
    print("nav: every page of zensical.toml exists")


if __name__ == "__main__":
    main()
