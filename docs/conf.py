import os
import sys
from datetime import date
import toml

# Load the version from pyproject.toml
pyproject_path = os.path.abspath('/Users/joshuamoore/Dropbox/Ox_PostDoc/Method_development/netPCF_paper/netPCF/spacenet/pyproject.toml')
with open(pyproject_path, 'r') as f:
    pyproject_data = toml.load(f)

project = 'spacenet'
copyright = '2026, Joshua W. Moore'
author = 'Joshua W. Moore'
release = pyproject_data['project']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_favicon",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "texext",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinxcontrib.collections",
    "sphinx_gallery.load_style",
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinx-prompt",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinx-jsonschema",
    "sphinx_carousel.carousel"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = 'SpaceNet'
html_short_title = 'SpaceNet'

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

# -- Options for sphinxcontrib-bibtex ---------------------------------------
bibtex_bibfiles = ['spacenet_citations.bib']



# Drop out the sidebar on specific pages
html_sidebars = {
    "references": [],
    "install": [],
}

# auto doc settings
autodoc_default_options = {
    'inherited-members': None,
}
autodoc_typehints = 'none'
copybutton_only_copy_prompt_lines = False
html_show_sourcelink = False
html_copy_source = False

html_favicon = '_static/images/fav.svg'


# theme options
html_theme_options = {
    "primary_sidebar_end": ["indices.html"],
    "logo": {
        "image_light": "_static/images/logo_full_color.png",
        "image_dark": "_static/images/logo_full_color.png",
    },
    "navbar_end": ["navbar-icon-links", "theme-switcher", "version-switcher"],
    "navbar_start": ["navbar-logo"],
    "collapse_navigation": True,
    "navigation_depth": 2,
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "header_links_before_dropdown": 7,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/joshwillmoore1/spacenet",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
    "analytics": {"google_analytics_id": "G-HQ98Y5N0C9"},
    "switcher": {
        "json_url": 'https://www.spacenet-python.com/version_switcher.json',
        "version_match": release,
    },
}



collections = {
    'basic_tutorials': {
        'driver': 'copy_folder',
        'source': '../../spacenet_docs_tutorials/getting_started',
        'target': 'getting_started/',
        'ignore': ['*.py', '.sh'],
    },
    
    
    'misc_tutorials': {
        'driver': 'copy_folder',
        'source': '../../spacenet_docs_tutorials/misc',
        'target': 'misc/',
        'ignore': ['*.py', '.sh'],
    },
    
    'pp_tutorials': {
        'driver': 'copy_folder',
        'source': '../../spacenet_docs_tutorials/point_patterns',
        'target': 'point_patterns/',
        'ignore': ['*.py', '.sh'],
    },
    
    'part_tutorials': {
        'driver': 'copy_folder',
        'source': '../../spacenet_docs_tutorials/partition',
        'target': 'partition/',
        'ignore': ['*.py', '.sh'],
    },

    'rel_notes': {
        'driver': 'copy_folder',
        'source': '../release_notes',
        'target': 'release_notes/',
        'ignore': ['*.py', '.sh'],
    },

    
    
}

