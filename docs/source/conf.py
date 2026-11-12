"""Sphinx configuration for the USAS Evaluation Framework documentation."""

import logging

import usas_evaluation_framework

project = "USAS Evaluation Framework"
copyright = "2026, UCREL Research Centre"
author = "UCREL Research Centre"
release = usas_evaluation_framework.__version__
version = usas_evaluation_framework.__version__

extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "show-inheritance": True,
    "undoc-members": True,
    "special-members": "__init__",
    "exclude-members": "model_post_init, model_config",
}
autodoc_member_order = "bysource"
typehints_defaults = "comma"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    '.txt': 'markdown',
}

# Building fetches the inventories of these projects, so it needs network
# access. If offline, Sphinx emits warnings and leaves the cross-references to
# external docs unresolved (these warnings fail the build only with ``-W``).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}


class FilterSphinxAutodocTypehintsFilter(logging.Filter):
    """Hide noisy forward-reference warnings raised for pydantic models."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            text in message
            for text in ("Cannot resolve forward reference", "Failed guarded type import")
        )


# Logger filters do not apply to records propagated from child loggers (e.g.
# ``sphinx.sphinx_autodoc_typehints``), so attach the filter to the handlers
# Sphinx installs on its ``sphinx`` logger, which every child record reaches.
for _handler in logging.getLogger("sphinx").handlers:
    _handler.addFilter(FilterSphinxAutodocTypehintsFilter())
