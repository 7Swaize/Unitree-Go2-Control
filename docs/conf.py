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
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
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
html_theme = 'furo'
html_static_path = ['_static']

html_last_updated_fmt = "%Y-%m-%d"

# Optional: theme-specific options
html_theme_options = {
    "light_logo": "logo-light.png", 
    "dark_logo": "logo-dark.png",
    "navigation_with_keys": True,    
}

# -- Custom CSS tweaks -------------------------------------------------------
html_css_files = [
    'custom.css',
]

# Make copy button ignore Python REPL prompts (>>> and ...)
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True