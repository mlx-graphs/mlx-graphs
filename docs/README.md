# Documentation
The Python dependencied needed to build the documentation can be installed via `pip install -e '.[docs]'`.

`pandoc` is also required to build the docs - it can be installed via `brew install pandoc`.

The site uses [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io), with
[sphinx-design](https://sphinx-design.readthedocs.io) for the cards, grids and buttons. The
notebooks in `source/tutorials/notebooks` are rendered with their stored outputs and are
never executed by the build.

## Build
```
cd docs
make html
```
You can see the docs locally by
```
open build/html/index.html
```

## Live preview
To rebuild on every change and reload the browser:
```
pip install sphinx-autobuild
cd docs
sphinx-autobuild source build/html -a --ignore "*/_autosummary*"
```
`-a` makes static files (CSS, images) update too.
