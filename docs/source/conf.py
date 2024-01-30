import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mlx-graphs"
copyright = "2024, mlx-graphs contributors"
author = "mlx-graphs contributors"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
]

napoleon_use_param = True
napoleon_google_docstring = True
# napoleon_preprocess_types = True
napoleon_attr_annotations = True
typehints_use_signature = True
typehints_use_signature_return = True
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = ".rst"
master_doc = "index"
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    # "show_toc_level": 2,
    "repository_url": "https://github.com/TristanBilot/mlx-graphs",
    "use_repository_button": True,
    "navigation_with_keys": False,
}

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mlx": ("https://ml-explore.github.io/mlx/build/html", None),
}

autodoc_mock_imports = ["mlx"]
