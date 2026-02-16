# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
from pathlib import Path

import west.manifest

manifest = west.manifest.Manifest.from_topdir()

EDGE_AI_BASE = Path(manifest.repo_abspath)
NRF_BASE = Path(manifest.get_projects(['nrf'])[0].abspath)
ZEPHYR_BASE = Path(manifest.get_projects(['zephyr'])[0].abspath)

sys.path.insert(0, str(NRF_BASE / 'doc' / '_extensions'))
sys.path.insert(0, str(ZEPHYR_BASE / 'doc' / '_extensions'))

# Needed by options_from_kconfig extension which is not self contained
sys.path.insert(0, str(ZEPHYR_BASE / 'scripts'))

# -- Project information -----------------------------------------------------

project = 'nRF Connect SDK - Edge AI Add-on'
copyright = '2026, Nordic Semiconductor'
author = 'Nordic Semiconductor'

# The full version, including alpha/beta/rc tags
release = '2026'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'options_from_kconfig',
    'table_from_rows',
    'zephyr.doxyrunner',
    'zephyr.doxybridge',
    'zephyr.external_content',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Options for external_content -------------------------------------------------

external_content_contents = [
    (EDGE_AI_BASE / "doc", "[!_]*"),
    (EDGE_AI_BASE, "applications/**/*.rst"),
    (EDGE_AI_BASE, "samples/**/*.rst"),
    (EDGE_AI_BASE, "tests/**/*.rst"),
]

# -- Options for doxyrunner plugin ---------------------------------------------

_doxyrunner_outdir = Path(sys.argv[4]) / "html" / "doxygen"

doxyrunner_doxygen = os.environ.get("DOXYGEN_EXECUTABLE", "doxygen")
doxyrunner_projects = {
    "edge-ai": {
        "doxyfile": EDGE_AI_BASE / "doc" / "doxyfile.in",
        "outdir": _doxyrunner_outdir,
        "fmt": True,
        "fmt_vars": {
            "NRF_BASE": str(NRF_BASE),
            "EDGE_AI_BASE": str(EDGE_AI_BASE),
            "DOCSET_SOURCE_BASE": str(EDGE_AI_BASE),
            "DOCSET_BUILD_DIR": str(_doxyrunner_outdir),
            "DOCSET_VERSION": release,
        },
    }
}

# -- Options for zephyr.doxybridge plugin ---------------------------------

doxybridge_projects = {"edge-ai": doxyrunner_projects["edge-ai"]["outdir"]}

# Options for table_from_rows --------------------------------------------------

table_from_rows_base_dir = EDGE_AI_BASE
table_from_sample_yaml_board_reference = "/includes/sample_board_rows.txt"

# Options for options_from_kconfig ---------------------------------------------

options_from_kconfig_base_dir = EDGE_AI_BASE
options_from_kconfig_zephyr_dir = ZEPHYR_BASE

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_ncs_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static'] (currently not in use)

rst_epilog = """
.. include:: /links.txt
.. include:: /shortcuts.txt
"""
