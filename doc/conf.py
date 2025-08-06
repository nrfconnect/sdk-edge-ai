# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import sys
from pathlib import Path

import west.manifest

manifest = west.manifest.Manifest.from_topdir()

EDGE_AI_BASE = Path(manifest.repo_abspath)
ZEPHYR_BASE = Path(manifest.get_projects(["zephyr"])[0].abspath)

sys.path.extend(map(
    lambda project: str(Path(project.abspath).absolute() / "doc" / "_extensions"),
    manifest.get_projects(["zephyr", "nrf"])
))

# -- Project information -----------------------------------------------------

project = "nRF Connect SDK - Edge AI add-on"
copyright = "2025, Nordic Semiconductor"
author = "Nordic Semiconductor"

# -- General configuration ---------------------------------------------------
extensions = [
    "options_from_kconfig",
    "sphinx_tabs.tabs",
    "table_from_rows",
    "zephyr.domain",
    "zephyr.external_content",
    "zephyr.kconfig",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The root document.
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["venv", "shortcuts.rst", "links.rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_ncs_theme"
html_theme_options = {"docsets": {}}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

plantuml = "java -jar /usr/local/bin/plantuml.jar"

rst_epilog = """
.. include:: /links.rst
.. include:: /shortcuts.rst
"""


# Options for options_from_kconfig ---------------------------------------------

options_from_kconfig_base_dir = EDGE_AI_BASE
options_from_kconfig_zephyr_dir = ZEPHYR_BASE


# Options for table_from_rows --------------------------------------------------

table_from_rows_base_dir = EDGE_AI_BASE
table_from_sample_yaml_board_reference = "/includes/sample_board_rows.txt"


# Options for external_content -------------------------------------------------

external_content_contents = [
    (EDGE_AI_BASE / "doc", "[!_]*"),
    (EDGE_AI_BASE, "applications/**/*.rst"),
    (EDGE_AI_BASE, "samples/**/*.rst"),
]
