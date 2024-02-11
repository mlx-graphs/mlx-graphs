# Documentation
The Python dependencied needed to build the documentation can be installed via `pip install -e '.[docs]'`.

`pandoc` is also required to build the docs - it can be installed via `brew install pandoc`.

## Build
```
cd docs
make html
```
You can see the docs locally by
```
open build/html/index.html
```
