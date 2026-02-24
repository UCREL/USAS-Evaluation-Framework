# Agents.md

## Python Guidelines

- Use match-case syntax instead of if/elif/else for pattern matching.
- Employ modern type hints using built-in generics (list, dict) and the union pipe (|) operator, rather than deprecated types from the typing module (e.g., Optional, Union, Dict, List).
- Ensure code adheres to strong static typing practices compatible with static analyzers like mypy, to note we use [ty](https://docs.astral.sh/ty/) in this project.
- Favour pathlib.Path methods for file system operations over older os.path functions.
- Additional best practices including f-string formatting, comprehensions, context managers, and overall PEP 8 compliance.
- Use Match-Case-Syntax where possible over if/elif/else statements.


## Development

- Use uv for all commands.We use uv to manage our python environment. You should nevery try to run a bare python commands. Always run commands using uv instead of invoking python or pip directly. For example, use uv add package and uv run script.py rather than pip install package or python script.py. This practice helps avoid environment drift and leverages modern Python packaging best practices. Useful uv commands are:
  - uv add/remove to manage dependencies. NOTE when adding/removing a developer dependency use --dev, e.g. uv add --dev
  - uv run script.py to run a script within the uv environment
- Use make to run linting and tests:
  - make lint - this will run the linter.
  - make test - this will run the tests.
- The code should be documented in Google-Style format for the Python code, with the addition that all doc string examples should be in the doctest style. An example of this style format can be found in the local file [coding_style_format_example.py](./coding_style_format_example.py).

# General coding guidelines

- Priorities readability, for example do not use deep nested if statements or list comprehensions.