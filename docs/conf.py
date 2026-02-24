from __future__ import annotations

project = "pathsig"
author = ""

extensions = [
    "numpydoc",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
}
html_show_sourcelink = False

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
