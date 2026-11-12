"""Sphinx configuration for the USAS Evaluation Framework documentation."""

from enum import Enum
from typing import Any

from sphinx.application import Sphinx

# Private module, but this is the sentinel autodoc passes for annotation-only
# attributes (pydantic fields, TypedDict keys). Re-check on Sphinx upgrades.
from sphinx.ext.autodoc._sentinels import INSTANCE_ATTR

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


# Hide noisy sphinx_autodoc_typehints warnings raised for pydantic models (e.g.
# unresolvable forward references inside pydantic's own ``Field``). Unlike a
# logging filter, ``suppress_warnings`` also stops them counting towards the
# build's warning total.
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
]


def skip_attributes_documented_by_napoleon(
    app: Sphinx, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool | None:
    """
    Skip autodoc's own entries for fields already documented by napoleon.

    Napoleon turns each class's Google-style ``Attributes:`` section into an
    ``.. attribute::`` entry with its type and description. Autodoc would also
    document the same pydantic fields, TypedDict keys and Enum members, without
    a description, giving duplicate entries and "duplicate object description"
    warnings. So every such member must be listed in its class's ``Attributes:``
    section, or it will not appear in the documentation.

    Returns:
        bool | None: True to skip the member, None to leave autodoc's decision.
    """
    if what == "class" and (obj is INSTANCE_ATTR or isinstance(obj, Enum)):
        return True
    return None


def setup(app: Sphinx) -> None:
    """Register the documentation's event handlers with Sphinx."""
    app.connect("autodoc-skip-member", skip_attributes_documented_by_napoleon)
