"""Generate the pages of the documentation site that come from other sources.

Run it before ``zensical build`` / ``zensical serve``::

    python docs/scripts/gen.py

It writes (all git-ignored):

- the API reference, from the templates in ``docs/api/``: one page per
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
SITE = HERE.parent  # docs/
ROOT = SITE.parent  # repository root
CONTENT = SITE / "content"
TEMPLATES = SITE / "api"
GENERATED = SITE / ".generated"
API_OUT = CONTENT / "api"
NB_DIR = CONTENT / "tutorials" / "notebooks"
GITHUB = "https://github.com/mlx-graphs/mlx-graphs"

sys.path.insert(0, str(HERE))
from griffe_rst import convert_inline  # noqa: E402

# Order of the API sections in the sidebar, and of the sub-pages of a section.
# A section is a directory of docs/api/ with an index.md; its other pages
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
    src: Path  # template, relative to docs/api
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
                except (
                    KeyError,
                    griffe.AliasResolutionError,
                    griffe.CyclicAliasError,
                ) as e:
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
    documented = {e.obj.path for e in entries.values()} | {
        e.name for e in entries.values()
    }
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
    head = (
        {"class": "Class", "function": "Function"}[kinds.pop()]
        if len(kinds) == 1
        else "Name"
    )
    lines = [f"| {head} | Description |", "|---|---|"]
    for e in entries:
        link = (
            Path(*([".."] * len(page_dir.parts))) / e.page if page_dir.parts else e.page
        )
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

    text = re.sub(r"<!--\s*count\s+(\w+)\s*-->", count, text)
    # the edit button opens the template, not this generated page
    edit = f"edit_url: {GITHUB}/edit/main/docs/api/{page.src.as_posix()}"
    # the browser title: the page's heading, not its "Overview" label in the sidebar
    meta = f"{edit}\ntitle: {json.dumps(page.title)}"
    if text.startswith("---\n"):
        return text.replace("---\n", f"---\n{meta}\n", 1)
    return f"---\n{meta}\n---\n\n{text}"


DUPLICATE_NAMES: set[str] = set()


def object_page(entry: Entry) -> str:
    kind = entry.template or kind_of(entry.obj)
    options = dict(OPTIONS.get(kind, {}))
    source = entry.obj.relative_package_filepath.as_posix()
    lines = [
        "---",
        # the module too when another section has an object of the same name
        f"title: {entry.name}"
        + (f" ({entry.module})" if entry.name in DUPLICATE_NAMES else ""),
        f"edit_url: {GITHUB}/edit/main/{source}",
        "---",
        "",
        f"::: {entry.id}",
    ]
    if options:
        lines.append("    options:")
        for key, value in options.items():
            lines.append(f"      {key}: {json.dumps(value)}")
    lines += [
        "",
        f'<small class="mg-source">`from {entry.module} import {entry.name}` &middot; '
        f"[:fontawesome-brands-github: `{source}`]"
        f"({GITHUB}/blob/main/{source}#L{entry.obj.lineno})</small>",
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
            p.src.stem
            for p in pages
            if p.src.parent.as_posix() == section and p.src.stem != "index"
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
                    documented = bool(
                        member.docstring and member.docstring.value.strip()
                    )
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
    write(
        GENERATED / "api_objects.json", json.dumps(api_index(pages, entries), indent=0)
    )

    # remove pages of objects that are gone
    wanted = {API_OUT / p.src for p in pages} | {
        API_OUT / e.page for e in entries.values()
    }
    wanted.add(API_OUT / "SUMMARY.md")
    if API_OUT.exists():
        for old in API_OUT.rglob("*.md"):
            if old not in wanted:
                old.unlink()
    for page in pages:
        write(API_OUT / page.src, render_page(page, entries, added))
    names = [e.name for e in entries.values()]
    DUPLICATE_NAMES.update(n for n in names if names.count(n) > 1)
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
        return (
            f"!!! {kind}\n\n"
            + "\n".join("    " + line for line in body.split("\n"))
            + "\n"
        )

    return re.sub(
        r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*(?:\n|$))*)",
        sub,
        text,
        flags=re.M,
    )


def convert_notebook(path: Path) -> str:
    nb = json.loads(path.read_text())
    name = path.stem
    files = path.parent / f"{name}_files"
    if files.exists():
        shutil.rmtree(files)
    lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
    repo_path = f"docs/content/tutorials/notebooks/{path.name}"
    out = []
    n_images = 0
    first_heading_done = False
    for cell in nb["cells"]:
        source = _text(cell.get("source", ""))
        if cell["cell_type"] == "markdown":
            # figures referenced the Sphinx way (notebooks of older branches)
            source = source.replace(
                "../../_static/tutorials/", "../../assets/tutorials/"
            )
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
                    f"[:fontawesome-brands-github: View on GitHub]"
                    f"({GITHUB}/blob/main/{repo_path}){{ .md-button }}\n"
                    f"[:lucide-download: Download notebook]({path.name})"
                    "{ .md-button }\n"
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
        out.append(
            f'<div class="mg-nb-input" markdown>\n\n{_fence(source, lang)}\n\n</div>'
        )
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
                if (
                    "image/png" in data
                    or "image/jpeg" in data
                    or "image/svg+xml" in data
                ):
                    mime = next(
                        m
                        for m in ("image/svg+xml", "image/png", "image/jpeg")
                        if m in data
                    )
                    files.mkdir(exist_ok=True)
                    ext = {
                        "image/png": "png",
                        "image/jpeg": "jpg",
                        "image/svg+xml": "svg",
                    }[mime]
                    target = files / f"output_{n_images}.{ext}"
                    n_images += 1
                    if ext == "svg":
                        target.write_text(_text(data[mime]))
                    else:
                        target.write_bytes(base64.b64decode(_text(data[mime])))
                    alt = (
                        _text(data.get("text/plain", "output"))
                        .strip()
                        .replace("\n", " ")
                    )
                    alt = re.sub(r"[\[\]<>]", "", alt)[:120] or "output"
                    outputs.append(
                        ("mg-nb-image", f"![{alt}]({name}_files/{target.name})")
                    )
                elif "text/html" in data:
                    html = _text(data["text/html"]).strip()
                    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.S)
                    outputs.append(("mg-nb-html", html))
                elif "text/markdown" in data:
                    outputs.append(("", _text(data["text/markdown"])))
                elif "text/latex" in data:
                    outputs.append(("", _text(data["text/latex"])))
                elif "text/plain" in data:
                    outputs.append(
                        ("", _fence(_ANSI.sub("", _text(data["text/plain"]))))
                    )
        for cls, body in outputs:
            if cls == "mg-nb-html":
                out.append(f'<div class="mg-nb-output {cls}">\n{body}\n</div>')
            else:
                out.append(
                    f'<div class="mg-nb-output {cls}" markdown>\n\n{body}\n\n</div>'
                )
    heading = next((re.sub(r"^#\s+", "", s) for s in out if re.match(r"^#\s", s)), name)
    title = re.sub(r"[`*_]", "", heading.strip()) + " · Tutorials"
    front = (
        "---\n"
        f"edit_url: {GITHUB}/edit/main/{repo_path}\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
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
            thumb = (
                f'![]({rel("assets/tutorials/" + t["thumbnail"])}){{ loading=lazy }}'
            )
        else:
            icon = t.get("icon", "lucide/notebook").replace("/", "-")
            thumb = f":{icon}:"
        tags = "".join(f"<span>{tag}</span>" for tag in t.get("tags", []))
        cards.append(
            '<div class="mg-tutorial" markdown>\n\n'
            f'<div class="mg-tutorial__thumb" markdown>{thumb}</div>\n\n'
            f'<div class="mg-tutorial__body" markdown>\n\n'
            f'[{t["title"]}]({rel("tutorials/notebooks/" + t["notebook"] + ".md")})'
            "{ .mg-tutorial__title }\n\n"
            f'{t["text"]}\n\n'
            f'<div class="mg-tags">{tags}</div>\n\n'
            "</div>\n\n</div>"
        )
    return (
        '<div class="mg-tutorials" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>\n"
    )


def generate_notebooks() -> None:
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    for nb in notebooks:
        write(nb.with_suffix(".md"), convert_notebook(nb))
    print(f"notebooks: {len(notebooks)} converted (not executed)")
    tutorials = yaml.safe_load((SITE / "tutorials.yml").read_text())
    missing = {t["notebook"] for t in tutorials} - {nb.stem for nb in notebooks}
    if missing:
        raise SystemExit(f"tutorials.yml lists missing notebooks: {sorted(missing)}")
    write(
        GENERATED / "tutorial_cards.md", tutorial_cards(tutorials, "tutorials/index.md")
    )
    home = [t for t in tutorials if t.get("home")]
    write(GENERATED / "tutorial_cards_home.md", tutorial_cards(home, "index.md"))


# ---------------------------------------------------------------------------
# Changelog and benchmarks (reST written by other tools)
# ---------------------------------------------------------------------------

_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def simple_rst_to_md(
    text: str, image_dir: str | None = None, image_prefix: str = ""
) -> str:
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
        if (
            line.strip()
            and _UNDERLINE.match(nxt)
            and len(nxt.strip()) >= len(line.strip()) - 1
        ):
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
            while (
                i < len(lines) and lines[i].startswith((" ", "\t")) and lines[i].strip()
            ):
                om = re.match(r"^\s+:(\w+):\s*(.*)$", lines[i])
                if om and om.group(1) == "alt":
                    alt = om.group(2)
                elif om and om.group(1) == "class":
                    cls = om.group(2)
                i += 1
            name = src.rsplit("/", 1)[-1]
            suffix = (
                "#only-light"
                if "only-light" in cls
                else "#only-dark"
                if "only-dark" in cls
                else ""
            )
            out.append(f"![{alt}]({image_prefix}{name}{suffix})")
            continue
        if re.match(r"^\.\.\s", line) or (
            line.startswith("..") and not line.strip(".")
        ):
            # comments and other directives: skip with their body
            i += 1
            while i < len(lines) and (
                not lines[i].strip() or lines[i].startswith((" ", "\t"))
            ):
                if (
                    not lines[i].strip()
                    and i + 1 < len(lines)
                    and not lines[i + 1].startswith((" ", "\t"))
                ):
                    break
                i += 1
            continue
        indent = len(line) - len(line.lstrip())
        out.append(" " * (indent * 2) + line.lstrip() if indent else line)
        i += 1
    return convert_inline("\n".join(out), "mlx_graphs")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


_STABLE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_CONVENTIONAL = re.compile(
    r"^(?P<type>\w+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<title>.+)$"
)
_GROUPS = [
    ("breaking", "Breaking changes"),
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
    ("docs", "Documentation"),
    ("other", "Other"),
]


def stable_tags() -> list[tuple[str, str]]:
    """``(tag, date)`` of every vX.Y.Z tag, newest first."""
    out = git(
        "for-each-ref", "refs/tags", "--format=%(refname:short)\t%(creatordate:short)"
    )
    tags = []
    for line in out.splitlines():
        name, date = line.split("\t")
        if _STABLE.match(name):
            tags.append((name, date))
    return sorted(
        tags,
        key=lambda t: tuple(int(x) for x in _STABLE.match(t[0]).groups()),
        reverse=True,
    )


def release_commits(rev_range: str) -> list[str]:
    """Titles of the commits of main (first parent) in ``rev_range``."""
    out = git("log", "--first-parent", "--format=%s%x1f%b%x1e", rev_range)
    titles = []
    for record in out.split("\x1e"):
        if not record.strip():
            continue
        subject, _, body = record.strip("\n").partition("\x1f")
        m = re.match(r"^Merge pull request #(\d+) from \S+", subject)
        if m:  # older history: the PR title is the merge commit's body
            first = next((line for line in body.splitlines() if line.strip()), "")
            subject = f"{first.strip()} (#{m.group(1)})" if first else subject
        elif subject.startswith("Merge "):
            continue
        titles.append(subject)
    return titles


def release_entry(title: str) -> tuple[str, str]:
    """``(group, Markdown line)`` of a commit title."""
    m = _CONVENTIONAL.match(title)
    if m:
        kind, scope, text = m.group("type").lower(), m.group("scope"), m.group("title")
        group = (
            "breaking"
            if m.group("bang")
            else kind
            if kind in ("feat", "fix", "perf", "docs")
            else "other"
        )
        if group == "other":  # keep the type: ci, build, refactor, test...
            label = kind + (f"({scope})" if scope else "")
            text = f"**{label}:** {text}"
        else:
            text = (f"**{scope}:** " if scope else "") + text
    else:
        group, text = "other", title
    text = text.replace("<", "&lt;")
    text = re.sub(r"\(#(\d+)\)", rf"([#\1]({GITHUB}/pull/\1))", text)
    return group, f"- {text}"


def entries_md(titles: list[str]) -> str:
    groups: dict[str, list[str]] = {}
    for title in titles:
        group, line = release_entry(title)
        groups.setdefault(group, []).append(line)
    parts = []
    for key, label in _GROUPS:
        if key in groups:
            parts.append(f"**{label}**\n\n" + "\n".join(groups[key]))
    return "\n\n".join(parts) if parts else "*No pull request.*"


def changelog_sections() -> (
    tuple[dict[str, tuple[str, str]], list[tuple[str, str, int]]]
):
    """Hand-written notes of CHANGELOG.rst.

    Returns ``{version: (name, reST body)}`` for the version sections, and the
    items of the "Unreleased" section as ``(subsection, reST item, line)``.
    """
    lines = (ROOT / "CHANGELOG.rst").read_text().split("\n")
    heads = [
        i
        for i in range(len(lines) - 1)
        if lines[i].strip() and re.match(r"^-{3,}\s*$", lines[i + 1])
    ]
    versions, unreleased = {}, []
    for k, start in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(lines)
        title = lines[start].strip()
        body = lines[start + 2 : end]
        m = re.match(r"^v?(\d+\.\d+\.\d+)\s*:?\s*(.*)$", title)
        if m:
            versions[m.group(1)] = (m.group(2), "\n".join(body))
        elif title.lower() == "unreleased":
            sub = ""
            item_start = None
            for j, line in enumerate(body + ["<end>"]):
                if (
                    j + 1 < len(body)
                    and line.strip()
                    and re.match(r"^~{3,}\s*$", body[j + 1])
                ):
                    sub = line.strip()
                if item_start is not None and (
                    line.startswith("- ")
                    or line == "<end>"
                    or (
                        line.strip()
                        and not line.startswith(" ")
                        and not line.startswith("-")
                    )
                ):
                    text = "\n".join(body[item_start:j]).rstrip()
                    unreleased.append((sub, text, start + 2 + item_start + 1))
                    item_start = None
                if line.startswith("- "):
                    item_start = j
    return versions, unreleased


def first_release_of(line: int, tags: list[tuple[str, str]]) -> str | None:
    """The first stable tag containing the commit that wrote a CHANGELOG.rst line."""
    try:
        blame = git("blame", "--porcelain", "-L", f"{line},{line}", "CHANGELOG.rst")
    except subprocess.CalledProcessError:
        return None
    sha = blame.split()[0]
    if set(sha) == {"0"}:
        return None  # not committed yet
    containing = set(git("tag", "--contains", sha).split())
    for tag, _ in reversed(tags):  # oldest first
        if tag in containing:
            return tag
    return None


def rst_chunk_to_md(rst: str, top: int) -> str:
    """reST notes -> Markdown, with their headings starting at level ``top``."""
    md = simple_rst_to_md(rst)
    levels = sorted({len(m) for m in re.findall(r"^(#+) ", md, re.M)})
    for i, level in enumerate(levels):
        md = re.sub(rf"^{'#' * level} ", f"{'#' * (top + i)}\x00 ", md, flags=re.M)
    return md.replace("\x00", "")


API_RELEASES = "https://api.github.com/repos/mlx-graphs/mlx-graphs/releases"


def github_releases() -> dict[str, dict] | None:
    """Published GitHub releases by tag, or None when the API is unreachable.

    Uses GITHUB_TOKEN / GH_TOKEN when set (CI); works without one locally,
    within the rate limit of unauthenticated requests.
    """
    import urllib.error
    import urllib.request

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "mlx-graphs-docs"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    releases: dict[str, dict] = {}
    page = 1
    try:
        while True:
            req = urllib.request.Request(
                f"{API_RELEASES}?per_page=100&page={page}", headers=headers
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                batch = json.load(r)
            if not batch:
                break
            for rel in batch:
                if not rel.get("draft"):
                    releases[rel["tag_name"]] = rel
            page += 1
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(
            f"  note: GitHub releases unavailable ({e});"
            " release notes from the git history only"
        )
        return None
    return releases


def url_ok(url: str) -> bool:
    import urllib.request

    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "mlx-graphs-docs"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status < 400
    except Exception:  # noqa: BLE001
        return False


def release_body_md(body: str, broken_images: list[str]) -> str:
    """A GitHub release body (GitHub-flavoured Markdown) -> Python-Markdown."""
    text = body.replace("\r\n", "\n").strip()
    # GitHub's generated heading: a label, not an entry of the table of contents
    text = re.sub(r"^#+\s+What's Changed\s*$", "**What's Changed**", text, flags=re.M)
    # the other headings go under the version's h2: the top level becomes h3
    levels = [len(h) for h in re.findall(r"^(#{1,6})\s", text, re.M)]
    shift = 3 - min(levels) if levels else 0
    text = re.sub(
        r"^(#{1,6})\s+",
        lambda m: "#" * min(6, len(m.group(1)) + shift) + " ",
        text,
        flags=re.M,
    )
    # bare links of pull requests, compare views and users, as GitHub renders them
    text = re.sub(
        rf"(?<![(<\"\[]){re.escape(GITHUB)}/pull/(\d+)",
        rf"[#\1]({GITHUB}/pull/\1)",
        text,
    )
    text = re.sub(
        rf"(?<![(<\"\[]){re.escape(GITHUB)}/compare/([\w.\-]+)",
        rf"[\1]({GITHUB}/compare/\1)",
        text,
    )
    text = re.sub(r"(?<![(<\"\[\w/])(https?://[^\s)<>\]]+)", r"<\1>", text)
    text = re.sub(
        r"(?<![\w/@`\[])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\b",
        r"[@\1](https://github.com/\1)",
        text,
    )
    # lists need a blank line before them, and nested items 4 spaces
    lines, out = text.split("\n"), []
    for i, line in enumerate(lines):
        is_item = re.match(r"^\s*([-*+]|\d+\.)\s", line)
        if (
            is_item
            and out
            and out[-1].strip()
            and not re.match(r"^\s*([-*+]|\d+\.)\s", out[-1])
        ):
            if not out[-1].startswith(" "):
                out.append("")
        indent = len(line) - len(line.lstrip(" "))
        if is_item and indent:
            line = " " * (indent * 2) + line.lstrip(" ")
        out.append(line)
    text = "\n".join(out)
    for url in re.findall(
        r"!\[[^\]]*\]\((https?://[^)\s]+)\)|<img[^>]+src=\"(https?://[^\"]+)\"", text
    ):
        url = url[0] or url[1]
        if not url_ok(url):
            broken_images.append(url)
    return text


def generate_changelog() -> None:
    """The changelog page.

    One section per stable tag vX.Y.Z, newest first, rendered from

    - the published GitHub release of the tag: its body, which is the release
      notes (hand-written, or GitHub's generated "What's Changed");
    - otherwise (a tag whose release is not published yet, as in release.yml,
      which builds the docs before it publishes the release; or no network):
      the titles of the pull requests merged since the previous tag, from the
      git history, grouped by conventional-commit type.

    Hand-written notes of CHANGELOG.rst for a version are shown above. Commits
    after the last tag are listed as Unreleased; items of CHANGELOG.rst's
    "Unreleased" section go under the release that shipped the commit that
    wrote them (git blame + git tag --contains).
    """
    tags = stable_tags()
    releases = github_releases()
    versions, unreleased_items = changelog_sections()
    moved: dict[str | None, list[tuple[str, str]]] = {}
    for sub, item, line in unreleased_items:
        moved.setdefault(first_release_of(line, tags), []).append((sub, item))

    def notes_md(tag: str | None) -> str:
        """Hand-written notes of CHANGELOG.rst for a version."""
        parts = []
        version = tag[1:] if tag else None
        if version in versions and versions[version][1].strip():
            notes = rst_chunk_to_md(versions[version][1], 3)
            if len(notes) > 3000:  # long notes (0.2.0): folded under the release
                notes = "\n".join(
                    "    " + line if line.strip() else "" for line in notes.split("\n")
                )
                notes = re.sub(r"^    (#+) (.+)$", r"    **\2**", notes, flags=re.M)
                notes = f'??? info "Detailed notes (CHANGELOG.rst)"\n\n{notes}'
            parts.append(notes)
        items = moved.get(tag, [])
        if items:
            rst = []
            for sub in dict.fromkeys(s for s, _ in items):
                rst += [sub, "~" * max(3, len(sub)), ""] if sub else []
                rst += [item for s, item in items if s == sub] + [""]
            parts.append(rst_chunk_to_md("\n".join(rst), 3))
        return "\n\n".join(parts)

    intro = (
        "Every release of mlx-graphs, newest first, with the notes of its"
        f" [GitHub release]({GITHUB}/releases) and, for some versions, hand-written"
        f" notes from [`CHANGELOG.rst`]({GITHUB}/blob/main/CHANGELOG.rst)."
        " The packages are on [PyPI](https://pypi.org/project/mlx-graphs/#history)."
    )
    out = [
        "---",
        f"edit_url: {GITHUB}/edit/main/CHANGELOG.rst",
        "toc_depth: 2  # the table of contents lists the versions only",
        "---",
        "",
        "<!-- Generated by docs/scripts/gen.py from the GitHub releases, the git tags"
        " and CHANGELOG.rst -->",
        "",
        "# Changelog",
        "",
        intro,
        "",
    ]
    if releases is None:
        out += [
            '!!! warning "Built offline"',
            "",
            "    The GitHub releases could not be fetched when this page was built:"
            " every version lists the pull requests of the git history instead of"
            " its release notes.",
            "",
        ]
    latest = tags[0][0] if tags else None
    # squash merges of pull requests only: not the commits of a local branch
    pending = (
        [t for t in release_commits(f"{latest}..HEAD") if re.search(r"\(#\d+\)$", t)]
        if latest
        else []
    )
    if pending or moved.get(None):
        out += [
            "## Unreleased { #unreleased }",
            "",
            f'<p class="mg-release-meta" markdown>Merged after {latest} &middot; '
            f"[changes since {latest}]({GITHUB}/compare/{latest}...main)</p>",
            "",
        ]
        if moved.get(None):
            out += [notes_md(None), ""]
        out += [entries_md(pending), ""]

    sources, broken_images = {}, []
    for i, (tag, date) in enumerate(tags):
        version = tag[1:]
        previous = tags[i + 1][0] if i + 1 < len(tags) else None
        release = (releases or {}).get(tag)
        name = versions.get(version, ("", ""))[0]
        anchor = "v" + version.replace(".", "-")
        out += [f"## {version}{': ' + name if name else ''} {{ #{anchor} }}", ""]
        meta = [(release.get("published_at") or "")[:10] if release else date]
        if release:
            meta.append(f"[GitHub release]({release['html_url']})")
        if previous:
            meta.append(f"[Full diff]({GITHUB}/compare/{previous}...{tag})")
        meta.append(f"[PyPI](https://pypi.org/project/mlx-graphs/{version}/)")
        out += [
            '<p class="mg-release-meta" markdown>' + " &middot; ".join(meta) + "</p>",
            "",
        ]
        notes = notes_md(tag)
        if notes:
            out += [notes, ""]
        body = (release or {}).get("body") or ""
        if body.strip():
            out += [
                f"<!-- source: github-release {tag} -->",
                release_body_md(body, broken_images),
                "",
            ]
            sources[tag] = "GitHub release"
        else:
            if notes:
                out += [f"### Pull requests {{ #{anchor}-prs }}", ""]
            out += [
                entries_md(release_commits(f"{previous}..{tag}" if previous else tag)),
                "",
            ]
            sources[tag] = "git history" + (
                " (release not published)" if releases is not None else " (offline)"
            )
        if notes:
            sources[tag] += " + CHANGELOG.rst"
    md = "\n".join(out)

    # every stable tag has its section, with its release notes when published
    missing = [t for t, _ in tags if f"{{ #v{t[1:].replace('.', '-')} }}" not in md]
    if missing:
        raise SystemExit(f"changelog: no section for {missing}")
    if releases is not None:
        no_body = [
            t
            for t, _ in tags
            if (releases.get(t) or {}).get("body", "").strip()
            and f"<!-- source: github-release {t} -->" not in md
        ]
        if no_body:
            raise SystemExit(
                f"changelog: the release notes of {no_body} are not on the page"
            )
    write(CONTENT / "changelog.md", md)
    write(GENERATED / "changelog_sources.json", json.dumps(sources, indent=1))

    version = latest[1:] if latest else "dev"
    write(
        GENERATED / "version.md",
        f'<a class="mg-badge" href="changelog/#v{version.replace(".", "-")}">'
        f'<span class="mg-badge__dot"></span>Latest release: v{version}</a>\n',
    )
    from_releases = sum(1 for s in sources.values() if s.startswith("GitHub"))
    print(
        f"changelog: {len(tags)} releases (latest {latest}):"
        f" {from_releases} from GitHub releases,"
        f" {len(tags) - from_releases} from the git history;"
        f" {len(pending)} unreleased PRs;"
        f" CHANGELOG.rst notes for {sorted(versions)}"
    )
    for url in broken_images:
        print(f"  note: an image of a GitHub release does not load: {url}")


# Sphinx pages -> pages of this site (paths without .md; "" is the home page)
SPHINX_PAGES = {
    "install.html": "getting-started/installation",
    "tutorials/quickstart.html": "getting-started/quickstart",
    "tutorials/from_pyg.html": "getting-started/from-pyg",
    "tutorials/examples/index.html": "tutorials/examples",
    "guide/message_passing.html": "getting-started/writing-layers",
    "datasets.html": "getting-started/datasets",
    # linked from the README, not a page of the Sphinx site
    "tutorials/graph_classification.html": "tutorials/notebooks/graph_classification",
    "benchmarks.html": "benchmarks",
    "validation.html": "validation",
    "changelog.html": "changelog",
    "genindex.html": "api",
    # pages of this site that moved: the User guide is now in Getting started
    "guide/message-passing/index.html": "getting-started/writing-layers",
    "guide/datasets/index.html": "getting-started/datasets",
    "search.html": "",
}


def generate_redirects(entries: dict[str, Entry]) -> None:
    """A redirect page at every URL of the Sphinx site (docs/sphinx-urls.txt).

    The README, PyPI and other sites link to pages such as
    ``tutorials/quickstart.html`` or
    ``api/nn/_autosummary/mlx_graphs.nn.GCNConv.html``; each one now sends the
    reader to the page that replaces it.
    """
    by_id = {e.id: e for e in entries.values()}
    by_path = {e.obj.path: e for e in entries.values()}
    by_name: dict[str, list[Entry]] = {}
    for e in entries.values():
        by_name.setdefault(e.name, []).append(e)

    def target(old: str) -> tuple[str, str] | None:
        """(page path, anchor) replacing an old URL."""
        if old in SPHINX_PAGES:
            return SPHINX_PAGES[old], ""
        m = re.match(r"^tutorials/notebooks/(\w+)\.html$", old)
        if m and (NB_DIR / f"{m.group(1)}.ipynb").exists():
            return f"tutorials/notebooks/{m.group(1)}", ""
        m = re.match(r"^api/(\w+)/_autosummary[^/]*/([\w.]+)\.html$", old)
        if m:
            section, dotted = m.groups()
            parts = dotted.split(".")
            # the object itself, or a method / attribute of a documented class
            for cut in range(len(parts), 1, -1):
                path = ".".join(parts[:cut])
                entry = by_id.get(path) or by_path.get(path)
                if entry is None:
                    same = [
                        e
                        for e in by_name.get(parts[cut - 1], [])
                        if e.page.parts[0] == section
                    ]
                    entry = same[0] if len(same) == 1 else None
                if entry is not None:
                    anchor = (
                        f"{entry.id}.{'.'.join(parts[cut:])}"
                        if cut < len(parts)
                        else ""
                    )
                    return f"api/{entry.page.with_suffix('').as_posix()}", anchor
            return f"api/{section}", ""
        m = re.match(r"^api/([\w/]+)\.html$", old)
        if m:
            page = m.group(1)
            page = page[: -len("/index")] if page.endswith("/index") else page
            if (TEMPLATES / f"{page}.md").exists() or (
                TEMPLATES / page / "index.md"
            ).exists():
                return f"api/{page}", ""
        return None

    lines = (SITE / "sphinx-urls.txt").read_text().splitlines()
    count, unmapped = 0, []
    for old in (line.strip() for line in lines):
        if not old or old.startswith("#"):
            continue
        found = target(old)
        if found is None:
            unmapped.append(old)
            continue
        page, anchor = found
        if (CONTENT / old).with_suffix(".md").exists() and old.endswith("index.html"):
            continue  # same URL in both sites (e.g. api/nn/index.html)
        depth = old.count("/")
        rel = "../" * depth + (page + "/" if page else "")
        rel = rel or "./"
        url = f"https://mlx-graphs.github.io/mlx-graphs/{page + '/' if page else ''}"
        if anchor:
            rel += f"#{anchor}"
            url += f"#{anchor}"
        write(
            CONTENT / old,
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            "<title>Redirecting...</title>\n"
            f'<link rel="canonical" href="{url}">\n'
            '<meta name="robots" content="noindex">\n'
            f'<meta http-equiv="refresh" content="0; url={rel}">\n'
            f"<script>location.replace({json.dumps(rel)}"
            ' + (location.hash || ""))</script>\n'
            "</head>\n<body>\n"
            f'<p>This page has moved to <a href="{rel}">{url}</a>.</p>\n'
            "</body>\n</html>\n",
        )
        count += 1
    if unmapped:
        raise SystemExit(f"redirects: no page for the old URLs {unmapped}")
    print(f"redirects: {count} pages of the Sphinx site redirect to this one")


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
        entries = generate_api()
        generate_redirects(entries)
        generate_datasets_table()
    if "notebooks" in parts:
        generate_notebooks()
    if "changelog" in parts:
        generate_changelog()
    if "benchmarks" in parts:
        generate_benchmarks()
    if args.only is None:
        check_nav()
        check_lists()


def generate_datasets_table() -> None:
    """The "Available datasets" table, from docs/datasets.yml.

    Each row links the class that loads the dataset and shows the call, after
    checking that the class is exported by mlx_graphs.datasets and accepts the
    arguments (static analysis of its __init__). The dataset classes that no
    row uses are listed below the table.
    """
    import ast

    rows = yaml.safe_load((SITE / "datasets.yml").read_text())
    pkg = load_package()
    exported = set(public_exports(pkg, "mlx_graphs.datasets"))
    problems, used = [], set()
    lines = [
        "| Dataset | Load with | Graphs | Nodes | Edges | Avg degree"
        " | Features | Labels |",
        "|---|---|--:|--:|--:|--:|---|---|",
    ]
    for row in rows:
        cls, args = row["class"], row.get("args", "")
        if cls not in exported:
            problems.append(
                f"datasets.yml: {row['name']}: {cls}"
                " is not exported by mlx_graphs.datasets"
            )
            continue
        used.add(cls)
        obj = get_object(pkg, f"mlx_graphs.datasets.{cls}")
        init = next(
            (
                c.members["__init__"]
                for c in [obj, *obj.mro()]
                if "__init__" in c.members
            ),
            None,
        )
        params = [p for p in (init.parameters if init else []) if p.name != "self"]
        call = ast.parse(f"f({args})", mode="eval").body
        names = {p.name for p in params}
        if len(call.args) > len(
            [p for p in params if p.kind is None or "positional" in str(p.kind)]
        ):
            problems.append(
                f"datasets.yml: {row['name']}: too many arguments for {cls}"
            )
        for kw in call.keywords:
            if kw.arg not in names:
                problems.append(
                    f"datasets.yml: {row['name']}: {cls} has no argument {kw.arg!r}"
                )
        stats = [
            str(row.get(k, "–")).replace("~", "\\~")
            for k in ("graphs", "nodes", "edges", "avg_degree")
        ]
        lines.append(
            f"| [{row['name']}][mlx_graphs.datasets.{cls}] | `{cls}({args})` | "
            + " | ".join(stats)
            + f" | {row.get('features', '–')} | {row.get('labels', '–')} |"
        )
    if problems:
        raise SystemExit("\n".join(problems))
    others = sorted(exported - used, key=str.lower)
    other_line = (
        "Other dataset classes: "
        + ", ".join(f"[`{name}`][mlx_graphs.datasets.{name}]" for name in others)
        + "."
    )
    write(GENERATED / "datasets_table.md", "\n".join(lines) + "\n")
    write(GENERATED / "datasets_other.md", other_line + "\n")
    print(
        f"datasets: {len(rows)} rows, {len(used)} classes in the table,"
        f" {len(others)} others listed"
    )


def check_lists() -> None:
    """Hand-written lists that must follow the code."""
    problems = []
    examples = (CONTENT / "tutorials" / "examples.md").read_text()
    for script in sorted((ROOT / "examples").glob("*.py")):
        if f"examples/{script.name})" not in examples:
            problems.append(
                f"tutorials/examples.md does not list examples/{script.name}"
            )
    # the datasets page: every dataset class is in its table or its list
    datasets_text = (GENERATED / "datasets_table.md").read_text() + (
        GENERATED / "datasets_other.md"
    ).read_text()
    for name in public_exports(load_package(), "mlx_graphs.datasets"):
        if f"[mlx_graphs.datasets.{name}]" not in datasets_text:
            problems.append(
                f"getting-started/datasets.md does not link mlx_graphs.datasets.{name}"
            )
    if problems:
        raise SystemExit("\n".join(problems))
    print("lists: examples and datasets pages match examples/ and mlx_graphs.datasets")


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
        p
        for p in pages(config["nav"])
        if not p.startswith("http") and not (CONTENT / p).exists()
    ]
    if missing:
        raise SystemExit(f"zensical.toml nav lists missing pages: {missing}")
    print("nav: every page of zensical.toml exists")


if __name__ == "__main__":
    main()
