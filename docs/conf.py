import os
import jaxoplanet

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if os.environ.get("READTHEDOCS", "") == "True":
    html_context = {"READTHEDOCS": True}

language = "en"
master_doc = "index"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
    "autoapi.extension",
]

autoapi_dirs = ["../src"]
autoapi_ignore = ["*_version*", "*/types*"]
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # "imported-members",
]
# autoapi_add_toctree_entry = False
autoapi_template_dir = "_autoapi_templates"

suppress_warnings = ["autoapi.python_import_resolution"]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]

# General information about the project.
project = "jaxoplanet"
copyright = "2021-2024 Simons Foundation, Inc."
version = jaxoplanet.__version__
release = jaxoplanet.__version__

exclude_patterns = ["_build", "_autoapi_templates"]
html_theme = "sphinx_book_theme"
html_title = "jaxoplanet documentation"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/exoplanet-dev/jaxoplanet",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
}
nb_execution_mode = "cache"
nb_execution_excludepatterns = []
nb_execution_timeout = -1
