import os
import re
import sys
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath("../.."))
autodoc_mock_imports = ["mlx", "networkx", "mlx_cluster", "torch", "torch_geometric"]
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mlx-graphs"
copyright = "2024, mlx-graphs contributors"
author = "mlx-graphs contributors"
release = package_version("mlx-graphs")
# Shown in the navbar: drop the local "+g<sha>" part of development versions
version = release.split("+")[0]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.opengraph",
]

napoleon_use_param = True
napoleon_google_docstring = True
# napoleon_preprocess_types = True
napoleon_attr_annotations = True
typehints_use_signature = True
typehints_use_signature_return = True
autosummary_generate = True

# API readability: short names in signatures and in the page TOC, one
# parameter per line once a signature gets long, and Literal["a", "b"]
# shown as "a" | "b"
add_module_names = False
toc_object_entries_show_parents = "hide"
python_maximum_signature_line_length = 88
python_display_short_literal_types = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

source_suffix = ".rst"
master_doc = "index"
highlight_language = "python"

# Notebooks are rendered with their stored outputs, never executed
nbsphinx_execute = "never"
# A download link at the top of every notebook page ("Edit on GitHub" is in
# the page header)
nbsphinx_prolog = r"""
{% set nbname = env.docname.split('/') | last %}

.. container:: nb-links

   :download:`Download notebook <{{ nbname }}.ipynb>`
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "mlx-graphs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo_dark.svg",
    },
    "navbar_start": ["navbar-logo", "navbar-version"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 7,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mlx-graphs/mlx-graphs",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mlx-graphs/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    # No right-hand sidebar: the content gets the width. "Edit on GitHub"
    # moves next to the breadcrumbs; long guides keep their own
    # ".. contents::" lists
    "secondary_sidebar_items": [],
    "article_header_end": ["edit-this-page"],
    "navigation_with_keys": False,
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "tango",
    "pygments_dark_style": "github-dark",
}
html_context = {
    "github_user": "mlx-graphs",
    "github_repo": "mlx-graphs",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}
# The home page has no left sidebar
html_sidebars = {"index": []}

# sphinx-copybutton: don't copy prompts and notebook output markers
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True
copybutton_exclude = ".linenos, .gp, .go"

# Open Graph metadata for link previews
ogp_site_url = "https://mlx-graphs.github.io/mlx-graphs/"
ogp_image = "_static/og-image.png"
ogp_description_length = 200
ogp_social_cards = {"enable": False}

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mlx": ("https://ml-explore.github.io/mlx/build/html", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}


# -- API reference tweaks ------------------------------------------------------

from sphinx.ext.autosummary import Autosummary  # noqa: E402


class _SummaryTable(Autosummary):
    """Autosummary tables list names and one-line summaries, without signatures:
    the full signature is one click away, and the tables stay easy to scan."""

    def run(self):
        self.options.setdefault("nosignatures", True)
        return super().run()


def _drop_object_base(app, name, obj, options, bases):
    # "Bases: object" says nothing
    bases[:] = [base for base in bases if base is not object]


# -- API sections with one page per category ----------------------------------
#
# api/nn/index.rst lists every layer of mlx_graphs.nn on one page, grouped by
# section headings. At build time, each section becomes its own page in
# api/nn/_autosummary_sections/ (generated, git-ignored like _autosummary/),
# and the index renders as cards linking to them, so the sidebar lists the
# categories. index.rst stays the single place where entries are added.

# index docname -> title of the entries listed before the first heading
_SPLIT_API_INDEXES = {"api/nn/index": "Base classes and linear layers"}
_SECTIONS_DIR = "_autosummary_sections"
_split_indexes = {}  # docname -> (header lines, [(title, slug, names)])


def _slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _parse_api_index(text, first_title):
    """Split an API index into its header and its sections.

    Returns the lines up to the page title, and (title, body lines, module)
    for the entries before the first heading and for every section.
    """
    lines = text.splitlines()
    # the page title: a line underlined with "="
    title_end = next(
        i + 1
        for i in range(len(lines) - 1)
        if lines[i].strip() and re.fullmatch(r"=+", lines[i + 1].strip())
    )
    header = lines[: title_end + 1]
    sections = [[first_title, [], None]]
    module = None
    i = title_end + 1
    while i < len(lines):
        line = lines[i]
        is_heading = (
            i + 1 < len(lines)
            and line.strip()
            and not line.startswith((" ", "\t", ".."))
            and re.fullmatch(r"-{3,}", lines[i + 1].strip() or "x")
        )
        if is_heading:
            sections.append([line.strip(), [], module])
            i += 2
            continue
        m = re.match(r"\.\. currentmodule:: (\S+)", line)
        if m:
            module = m.group(1)
        sections[-1][1].append(line)
        i += 1
    return header, [s for s in sections if any(x.strip() for x in s[1])]


def _autosummary_names(body):
    names, in_block = [], False
    for line in body:
        if line.startswith(".. autosummary::"):
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if line and not line[0].isspace():
                in_block = False
            elif stripped and not stripped.startswith(":"):
                names.append(stripped.lstrip("~").split(".")[-1])
    return names


def _write_if_changed(path, content):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            if f.read() == content:
                return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _generate_api_sections(app):
    for docname, first_title in _SPLIT_API_INDEXES.items():
        index_path = os.path.join(app.srcdir, docname + ".rst")
        with open(index_path, encoding="utf-8") as f:
            header, sections = _parse_api_index(f.read(), first_title)
        out_dir = os.path.join(os.path.dirname(index_path), _SECTIONS_DIR)
        os.makedirs(out_dir, exist_ok=True)
        pages = []
        for title, body, module in sections:
            slug = _slugify(title)
            body = [re.sub(r"(:toctree:\s*)(\S+)", r"\1../\2", line) for line in body]
            page = [title, "=" * len(title), ""]
            if module:
                page += [f".. currentmodule:: {module}", ""]
            page += body
            _write_if_changed(
                os.path.join(out_dir, slug + ".rst"), "\n".join(page).strip() + "\n"
            )
            pages.append((title, slug, _autosummary_names(body)))
        _split_indexes[docname] = (header, pages)


def _render_split_index(app, docname, source):
    # The index page becomes a card per category, plus the toctree that puts
    # the categories in the sidebar
    if docname not in _split_indexes:
        return
    header, pages = _split_indexes[docname]
    out = header + ["", ".. grid:: 1 2 2 3", "   :gutter: 3", ""]
    for title, slug, names in pages:
        shown = ", ".join(f"``{n}``" for n in names[:4])
        more = f", and {len(names) - 4} more" if len(names) > 4 else ""
        out += [
            f"   .. grid-item-card:: {title}",
            f"      :link: {_SECTIONS_DIR}/{slug}",
            "      :link-type: doc",
            "",
            f"      {shown}{more}",
            "",
        ]
    out += [".. toctree::", "   :hidden:", ""]
    out += [f"   {_SECTIONS_DIR}/{slug}" for _, slug, _ in pages]
    source[0] = "\n".join(out) + "\n"


def setup(app):
    app.add_directive("autosummary", _SummaryTable, override=True)
    app.connect("autodoc-process-bases", _drop_object_base)
    app.connect("builder-inited", _generate_api_sections, priority=100)
    app.connect("source-read", _render_split_index)
