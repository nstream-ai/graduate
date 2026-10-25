"""Sphinx configuration for the Graduate docs (built on Read the Docs from .readthedocs.yaml)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

project = "Graduate"
author = "The Graduate Authors"
copyright = "The Graduate Authors"
try:
    release = _version("graduate")
except PackageNotFoundError:  # building from a source tree that is not installed
    release = "unknown"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

myst_heading_anchors = 3  # README links to its own sections, e.g. #trace-formats
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "furo"
html_title = f"Graduate {release}"
html_theme_options = {
    "source_repository": "https://github.com/nstream-ai/graduate/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {"name": "PyPI", "url": "https://pypi.org/project/graduate/", "html": "PyPI"},
        {"name": "GitHub", "url": "https://github.com/nstream-ai/graduate", "html": "GitHub"},
    ],
}
