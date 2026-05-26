# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

import west.manifest

manifest = west.manifest.Manifest.from_topdir()

EDGE_AI_BASE = Path(manifest.repo_abspath)
NRF_BASE = Path(manifest.get_projects(['nrf'])[0].abspath)
ZEPHYR_BASE = Path(manifest.get_projects(['zephyr'])[0].abspath)

sys.path.insert(0, str(EDGE_AI_BASE / 'doc' / '_extensions'))
sys.path.insert(0, str(NRF_BASE / 'doc' / '_extensions'))
sys.path.insert(0, str(ZEPHYR_BASE / 'doc' / '_extensions'))

# Needed by options_from_kconfig extension which is not self contained
sys.path.insert(0, str(ZEPHYR_BASE / 'scripts'))

from edge_ai_project_info import (
    get_manifest_revision,
    get_version_string,
    strip_v,
    build_rst_epilog,
)

# -- Project information -----------------------------------------------------

project = 'nRF Connect SDK - Edge AI Add-on'
copyright = '2026, Nordic Semiconductor'
author = 'Nordic Semiconductor'

# The full version, including alpha/beta/rc tags.
# Read from edge-ai/VERSION via the canonical Zephyr-format parser.
# 'release' is a special symbol used by Sphinx for the version string in docs.
release = get_version_string(EDGE_AI_BASE / 'VERSION')


# -- Version information from west manifest ---------------------------------

NCS_VERSION              = get_manifest_revision(manifest, "nrf")
NCS_VERSION_NUMBER       = strip_v(NCS_VERSION)
EDGE_IMPULSE_SDK_VERSION = get_manifest_revision(manifest, "edge-impulse-sdk-zephyr")
TOOLCHAIN_NCS_ID         = NCS_VERSION
ADDON_RELEASE            = release

# Zephyr version — read from zephyr/VERSION (same format as edge-ai/VERSION).
ZEPHYR_VERSION_NUMBER    = get_version_string(ZEPHYR_BASE / "VERSION")
ZEPHYR_VERSION           = f"v{ZEPHYR_VERSION_NUMBER}"


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
    'sphinxcontrib.plantuml',
    'edge_ai_project_info',
]

plantuml = 'plantuml'
plantuml_output_format = 'svg_img'

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
    (EDGE_AI_BASE, "lib/**/Kconfig"),
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
html_theme_options = {
    'docsets': {},
}

html_extra_path = ['versions.json']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [str(EDGE_AI_BASE / "doc" / "_static")]
html_css_files = ['custom.css']

# -- Project-wide string substitutions ---------------------------------------
#
# Single source of truth for project-wide string substitutions (the tokens
# referenced in RST source as ``|name|``).
#
# Each entry is a ``(name, value)`` pair consumed by the
# ``edge_ai_project_info`` extension, which:
#
#   1. Validates uniqueness of names at builder-init time.
#
#   2. Pre-expands ``|name|`` markers in raw RST source *before* docutils
#      parses it (via a ``source-read`` hook).  This makes substitutions
#      work uniformly everywhere, including inside ``code-block`` and
#      ``literalinclude`` bodies.
#
#   3. Is used here via ``build_rst_epilog`` to pre-expand the same
#      markers inside ``links.txt`` / ``shortcuts.txt`` URI targets before
#      the text is handed to Sphinx as ``rst_epilog``.  (Docutils does not
#      expand substitution references inside hyperlink target URIs.)
#
# List longer names before any name that is a substring of another to
# ensure the longer match is applied first (the surrounding ``|``
# delimiters already prevent overlap today, but the convention keeps the
# file robust against future marker-syntax changes).
#
# Add new substitutions here only — do not redefine them elsewhere.

SUBSTITUTIONS = [
    ("release_version",             ADDON_RELEASE),
    ("ncs_version_number",          NCS_VERSION_NUMBER),
    ("ncs_version",                 NCS_VERSION),
    ("toolchain_ncs_id",            TOOLCHAIN_NCS_ID),
    ("edge_impulse_sdk_version",    EDGE_IMPULSE_SDK_VERSION),
    ("zephyr_version_number",       ZEPHYR_VERSION_NUMBER),
    ("zephyr_version",              ZEPHYR_VERSION),
]

# Consumed by the edge_ai_project_info extension (source-read handler).
edge_ai_substitutions = SUBSTITUTIONS

# -- rst_epilog: pre-expanded links.txt + shortcuts.txt ----------------------

rst_epilog = build_rst_epilog(EDGE_AI_BASE / "doc", SUBSTITUTIONS)
