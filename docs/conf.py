"""Sphinx configuration for the OpenSG-TW documentation site.

Built with `sphinx-book-theme` + `myst-nb` (the Jupyter-Book look, cf. the dolfinx-tutorial),
deployed to GitHub Pages. Tutorial notebooks are committed pre-executed and rendered as-is.
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # repo root, so autodoc can import opensg_jax.fe_jax.*

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
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True

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
# Read-the-Docs theme, matching the upstream OpenSG documentation (wenbinyugroup.github.io/OpenSG),
# of which OpenSG-TW is the thin-walled / JAX extension.
html_theme = "sphinx_rtd_theme"
html_title = "OpenSG-TW"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
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
# The docs build installs the JAX runtime (jax, fenics-basix, pypardiso, ...) so autodoc imports the real
# modules and renders true signatures. Only the optional FEniCS back-end is mocked (the JAX package guards
# it in try/except and it is never installed for the docs build).
autodoc_mock_imports = ["dolfinx", "ufl", "mpi4py", "petsc4py", "opensg"]
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
# The free-form source docstrings use plain-text math bars (|S_ij|) and indented blocks that the RST
# parser flags; these are cosmetic in the rendered API page, so don't let them fail the -W build.
# docutils/myst: free-form docstring math bars; autodoc: if an optional dep (e.g. libigl) is missing on a
# given runner, skip that module's autodoc rather than failing the whole -W build.
suppress_warnings = ["docutils", "myst.substitution", "autodoc"]
