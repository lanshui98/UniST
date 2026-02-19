# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'UniST_API'
copyright = '2026, Lan Shui'
author = 'Lan Shui'

release = 'latest'
version = 'latest'

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

autosummary_generate = True

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx_design",
    "nbsphinx",
    "myst_nb",
]

nb_execution_mode = "off"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'

def setup(app):
    app.add_css_file("custom.css")
