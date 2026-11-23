# The documentation site

Built with [Zensical](https://zensical.org) from `zensical.toml` (repository
root) and this directory. Published at <https://mlx-graphs.github.io/mlx-graphs/>.

## Preview locally

```shell
pip install -e '.[docs]'
python docs/scripts/gen.py      # generated pages (see below)
zensical serve                       # http://localhost:8000/mlx-graphs/, live reload
```

Re-run `gen.py` after changing an API template, a notebook, `tutorials.yml`,
`CHANGELOG.rst` or the benchmark results; edits to the hand-written pages
reload on their own. CI builds with `python docs/scripts/check_api.py`,
`gen.py` and `zensical build --strict --clean`
(`.github/workflows/docs.yml`); run the same before opening a PR.

## Where things are

| Path | What |
|---|---|
| `content/` | hand-written pages (Markdown): home, getting started (with writing layers and datasets), tutorials index, benchmarks, validation |
| `content/tutorials/notebooks/*.ipynb` | the tutorial notebooks, rendered with their stored outputs (never executed) |
| `content/assets/` | CSS, JavaScript, images, tutorial figures (`assets/tutorials/`) |
| `api/` | templates of the API reference: one Markdown page per section, with `<!-- api -->` blocks |
| `tutorials.yml` | the tutorial cards of the home and Tutorials pages |
| `overrides/` | theme overrides: header (home link, Discord), edit buttons, extra icons |
| `scripts/gen.py` | writes the generated pages |
| `scripts/griffe_rst.py` | griffe extension: the reST of the docstrings to Markdown |
| `scripts/check_api.py` | fails when a public name is missing from the API templates |
| `sphinx-urls.txt` | URLs of the old Sphinx site, which get a redirect page |

`gen.py` writes, all git-ignored: the API pages (`content/api/`), the notebook
pages, the changelog (from the git tags and `CHANGELOG.rst`), the benchmark
sections (from `benchmarks/pyg_comparison/results/*/*/results.rst`), the
tutorial cards and the redirect pages.

## Adding a class or function to the API reference

Add its name to an `<!-- api <module> -->` block of the page of `api/` where it
belongs, e.g. a new convolution layer in the "Convolution layers" block of
`api/nn/index.md`:

```markdown
<!-- api mlx_graphs.nn -->
GCNConv
MyNewConv
<!-- /api -->
```

The module is the one the name is imported from (`from mlx_graphs.nn import
MyNewConv`). `gen.py` then writes its page (rendered by mkdocstrings from the
docstring), its row in the page's table and its sidebar entry. A block may set
`template=data` (also render `__cat_dim__` and `__inc__`) or
`template=datasets`. Write docstrings in Google style; the reST of Sphinx
(`:class:`, `.. code-block::`, `.. math::`, ``` ``code`` ```) is converted.

`check_api.py` fails CI when a public name exported by a `mlx_graphs`
subpackage's `__init__.py` is in no block (it names the page and section it
most likely belongs to), and when a block lists a name that no longer exists.
A name in no block still gets a page, under "Other", so the local preview works
in the meantime.

## Adding a tutorial notebook

1. Put the notebook, with its outputs, in `content/tutorials/notebooks/`.
   It is never executed: run it before committing.
2. Make a pipeline figure in the style of the others (SVG, 960 px wide) in
   `content/assets/tutorials/`, and show it in a markdown cell after the title:
   `![description](../../assets/tutorials/my_tutorial.svg)`.
3. Add a card to `tutorials.yml` (title, one-line text, thumbnail, tags;
   `home: true` to show it on the home page too).
4. Add the page to the "Notebooks" entry of `nav` in `zensical.toml`.

## Release notes

The Changelog page has one section per `vX.Y.Z` tag, rendered from:

1. the body of the tag's published GitHub release (fetched from the REST API;
   `GITHUB_TOKEN` / `GH_TOKEN` is used when set, CI sets it);
2. when the release is not published yet (`release.yml` builds the docs before
   it publishes the release) or the API cannot be reached: the titles of the
   pull requests merged since the previous tag, from the git history, grouped
   by Conventional Commit type. An offline build says so at the top of the page.

Commits merged after the last tag are listed under Unreleased. `CHANGELOG.rst`
holds optional hand-written notes in a section titled with the version number,
shown above that version's release notes (long ones folded). `gen.py` fails if
a stable tag has no section, or if a published release body is missing. The
build needs the git tags (`fetch-depth: 0` in CI).
