from __future__ import annotations

project = "ChangePointLab"
author = "Diogo Ribeiro"
release = "0.1.6"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
