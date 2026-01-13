# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Unitree-Go2-Control'
copyright = '2026, Sachit Raheja'
author = 'Sachit Raheja'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Core autodoc functionality
    'sphinx.ext.napoleon',   # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',   # Add links to source code
    'sphinx.ext.todo',       # Optional: if you want to include TODOs
    'sphinx.ext.autosummary',# Optional: generate summary tables automatically
]
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

