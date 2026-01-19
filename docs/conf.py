# Configuration file for the Sphinx documentation builder. AI Generated.

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Unitree-Go2-Control'
copyright = '2026, Sachit Raheja'
author = 'Sachit Raheja'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',          # Core autodoc functionality
    'sphinx.ext.napoleon',         # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',         # Add links to source code
    'sphinx.ext.autosummary',      # Generate summary tables automatically
    'sphinx_copybutton',           # Adds "copy" buttons to code blocks
]

# Mock imports for modules that may not be available
autodoc_mock_imports = [
    "aruko_helpers",
    "pyrealsense2",
    "unitree_sdk2py"
]

# Autosummary settings
autosummary_generate = True
napoleon_numpy_docstring = True
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# Use a modern, responsive theme
html_theme = 'furo'  # Recommended: 'furo' or 'sphinx_rtd_theme'
html_static_path = ['_static']

# Optional: theme-specific options
html_theme_options = {
    "light_logo": "logo-light.png",  # if you have a logo
    "dark_logo": "logo-dark.png",
    "navigation_with_keys": True,     # allow keyboard nav
}

# -- Custom CSS tweaks -------------------------------------------------------
# Create _static/custom.css with any tweaks you want
html_css_files = [
    'custom.css',
]

# Suggested custom.css content (create _static/custom.css):
"""
/* Limit table widths and allow scrolling */
table {
    max-width: 100%;
    overflow-x: auto;
}

/* Make code blocks wrap nicely */
.highlight pre {
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Improve body readability */
body {
    line-height: 1.6;
    font-size: 14px;
}
"""

# Make copy button ignore Python REPL prompts (>>> and ...)
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True