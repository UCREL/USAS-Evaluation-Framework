"""Example Google style docstrings.
This was adapted from: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ uv run coding_style_format_example.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1: Module level variables are documented in
        the ``Attributes`` section of the module docstring. The type is not
        repeated here as it is given by the type annotation.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

from collections.abc import Iterator

module_level_variable1: int = 12345

module_level_variable2: int = 98765
"""Module level variables may be documented in either the ``Attributes``
section of the module docstring, or in an inline docstring immediately
following the variable.

Either form is acceptable, but the two should not be mixed. This codebase
uses the ``Attributes`` section of the module docstring, as shown for
`module_level_variable1`, so use that convention rather than this one.
"""


def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True for success, False otherwise.

    """

    return True


def module_level_function(param1: int,
                          param2: str | None = None,
                          *args: int,
                          **kwargs: str) -> bool:
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter and its description is required.

    If *args or **kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::
        name: description
            The description may span multiple lines. Following
            lines should be indented.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1: The first parameter.
        param2: The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        True if successful, False otherwise.

        The return type is not repeated in the ``Returns`` section as it is
        given by the type annotation, only its description is required.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                "param1": param1,
                "param2": param2
            }

    Raises:
        ValueError: If `param2` is the string form of `param1`. The
            ``Raises`` section is a list of all exceptions that the function
            can raise and that are relevant to the interface.

    Examples:
        Exceptions can be shown in doctests with a traceback. The lines
        between the first and last line of the traceback are replaced
        with ``...``.

        >>> module_level_function(1, "2")
        True
        >>> module_level_function(1, "1")
        Traceback (most recent call last):
            ...
        ValueError: param2 may not be the string form of param1

    """
    if param2 == str(param1):
        raise ValueError("param2 cannot be the string form of param1")
    return True


def example_generator(n: int) -> Iterator[int]:
    """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Args:
        n: The upper limit of the range to generate, from 0 to `n` - 1.

    Yields:
        The next number in the range of 0 to `n` - 1.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]

    """
    for i in range(n):
        yield i


def describe_tag(tag: str) -> str:
    """Use match-case, rather than if/elif/else, when branching on a value.

    match-case suits comparing a value against literal values or
    structural patterns, as each case reads as one of the possible values.

    Args:
        tag: A USAS semantic tag.

    Returns:
        A short description of the kind of tag.

    Examples:
        >>> describe_tag("")
        'no label'
        >>> describe_tag("PUNCT")
        'punctuation'
        >>> describe_tag("F2/O2")
        'multi-tag'
        >>> describe_tag("Z5")
        'single tag'

    """
    match tag:
        case "":
            return "no label"
        case "PUNCT" | "Z9":
            return "punctuation"
        case _ if "/" in tag:
            return "multi-tag"
        case _:
            return "single tag"


def confidence_level(score: float) -> str:
    """Use if/elif/else when match-case would be harder to read.

    match-case cannot express range comparisons such as ``score >= 0.9``
    directly, so every case would need a guard (``case _ if score >= 0.9:``).
    In cases like this, where the branches are conditions rather than
    values or patterns, if/elif/else is easier to read.

    Args:
        score: A confidence score between 0 and 1 inclusive.

    Returns:
        "high", "medium" or "low" depending on the score.

    Raises:
        ValueError: If `score` is not between 0 and 1 inclusive.

    Examples:
        >>> confidence_level(0.95)
        'high'
        >>> confidence_level(0.5)
        'medium'
        >>> confidence_level(0.1)
        'low'
        >>> confidence_level(1.5)
        Traceback (most recent call last):
            ...
        ValueError: score must be between 0 and 1 inclusive, got 1.5

    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be between 0 and 1 inclusive, got {score}")

    if score >= 0.9:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return "low"


class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented as a docstring on the __init__
    method itself.

    Attributes:
        msg: Human readable string describing the exception.
        code: Exception error code.

    """

    def __init__(self, msg: str, code: int = 0) -> None:
        """Initialise the exception with a message and an error code.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            msg: Human readable string describing the exception.
            code: Error code. Defaults to 0.
        """
        self.msg = msg
        self.code = code


class ExampleClass:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. The types are not repeated here as they
    are given by the type annotations.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1: Description of `attr1`.
        attr2: Description of `attr2`. Defaults to `None`.
        attr3: Description of `attr3`.
        attr4: Description of `attr4`.
        attr5: Description of `attr5`. Defaults to `None`.

    Examples:
        >>> example = ExampleClass("value", 2)
        >>> example.attr1
        'value'
        >>> example.readwrite_property = ["new value"]
        >>> example.readwrite_property
        ['new value']

    """

    def __init__(self,
                 param1: str,
                 param2: int | None = None,
                 param3: list[str] | None = None) -> None:
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: Description of `param1`.
            param2: Description of `param2`. Multiple
                lines are supported.
            param3: Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3

        self.attr4: list[str] = ["attr4"]

        self.attr5: str | None = None

        self._readwrite_property: list[str] = ["readwrite_property"]

    @property
    def readonly_property(self) -> str:
        """Properties should be documented in their getter method."""
        return "readonly_property"

    @property
    def readwrite_property(self) -> list[str]:
        """Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return self._readwrite_property

    @readwrite_property.setter
    def readwrite_property(self, value: list[str]) -> None:
        self._readwrite_property = value

    def example_method(self, param1: bool, param2: bool) -> bool:
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True

    def __special__(self) -> None:
        """By default special members with docstrings are not included.

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self) -> None:
        pass

    def _private(self) -> None:
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self) -> None:
        pass
