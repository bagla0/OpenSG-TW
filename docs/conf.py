"""Sphinx configuration for the OpenSG-TW documentation site.

Built with `sphinx-book-theme` + `myst-nb` (the Jupyter-Book look, cf. the dolfinx-tutorial),
deployed to GitHub Pages. Tutorial notebooks are committed pre-executed and rendered as-is.
"""
import datetime

project = "OpenSG-TW"
author = "Akshat Bagla (bagla0)"
copyright = "%d, Akshat Bagla" % datetime.date.today().year
release = "0.1.0"

# ---------------------------------------------------------------- extensions
extensions = [
    "myst_nb",            # MyST markdown + executed Jupyter notebooks (.ipynb)
    "sphinx_design",      # grids / cards / tabs on the landing page
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline", "substitution",
]
myst_dmath_double_inline = True

# Notebooks are committed already executed -> just render the stored outputs (never re-run on CI).
nb_execution_mode = "off"
nb_merge_streams = True

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "tutorials/_img/*",
    # internal dev notes kept in docs/ for reference but not part of the rendered site
    # (their content is folded into theory/reissner_mindlin.md and theory/msg_structure_genome.md)
    "MITC_transverse_shear.md", "MSG_TW_Beam_Formulation.md",
]

# ---------------------------------------------------------------- HTML / theme
html_theme = "sphinx_book_theme"
html_title = "OpenSG-TW"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "repository_url": "https://github.com/bagla0/OpenSG-TW",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {"text": "OpenSG-TW"},
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/bagla0/OpenSG-TW", "icon": "fa-brands fa-github"},
        {"name": "OpenSG_io", "url": "https://github.com/bagla0/OpenSG_io", "icon": "fa-solid fa-cube"},
    ],
    "announcement": "OpenSG-TW &mdash; JAX Mechanics-of-Structure-Genome beam homogenization (KL / RM shell &amp; 2-D solid).",
}

# MathJax v3 macros for the recurring symbols
mathjax3_config = {
    "tex": {
        "macros": {
            "bA": r"\mathbf{A}", "bB": r"\mathbf{B}", "bD": r"\mathbf{D}",
            "veps": r"\boldsymbol{\varepsilon}", "bkappa": r"\boldsymbol{\kappa}",
        }
    }
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None),
                       "numpy": ("https://numpy.org/doc/stable", None)}
autodoc_mock_imports = ["jax", "jaxlib", "basix", "pypardiso", "numba", "matplotlib", "scipy", "yaml"]
