# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository. `AGENTS.md` is a symlink to this file, so edit this file only.

## Project Overview

An evaluation framework for USAS semantic tagging, from UCREL at Lancaster University. Parsers turn each annotated corpus into an `EvaluationDataset`, and metrics are computed by comparing a gold dataset with a predicted one. So far only top-N accuracy (micro and macro) is implemented. The other metrics and splits listed in the README are planned but not yet built.

## Layout

- `src/usas_evaluation_framework/dataset.py` holds the Pydantic v2 data models: `EvaluationTexts` (one text) and `EvaluationDataset` (a collection, with `merge()` and `stats()`).
- USAS tag parsing (`parse_usas_token_group`), the USAS tag map (`load_usas_mapper`), the `USASTag`/`USASTagGroup` models and MWE index helpers come from the [USAS-Validator](https://github.com/UCREL/USAS-Validator) package (`usas_validator.utils`, `usas_validator.usas_tag`). Don't reimplement them here. Always call `parse_usas_token_group(..., strict=True)`, because the default `strict=False` silently drops invalid tags instead of raising `ValueError`, and the parsers' validation relies on that error.
- `src/usas_evaluation_framework/data_utils.py` holds small general helpers (`create_inner_list`).
- `src/usas_evaluation_framework/parsers/` has one parser per corpus, each subclassing `BaseParser` (`parsers/base.py`). `parse(dataset_path, label_validation, label_filter, dataset_name, language)` is a static method that returns an `EvaluationDataset`. New corpora should follow this pattern.
- `src/usas_evaluation_framework/metrics/` holds the evaluation metrics.
- `scripts/` holds Typer CLIs that convert annotated Excel/TSV files to CSV. They need the `excel-conversion` extra, and the README explains how to use them.
- `notebooks/` holds notebooks that evaluate PyMUSAS models on the Hugging Face `ucrelnlp/USAS-WSD` dataset and explore dataset statistics. They need the `notebooks` extra.
- `docs/source/` holds the Sphinx site (`conf.py`, `index.rst`, MyST Markdown pages in `overview/`, and API pages in `api/`). The API pages list each module with its own `automodule` directive, so when you add a module (for example a new parser or metric), also add it to the matching `docs/source/api/*.rst` file. `docs/build/` is gitignored build output.

## Domain Conventions

- `EvaluationTexts.semantic_tags` is `list[list[str]]`: one list per token, ordered from most to least likely. Gold data has a single tag per token. In `mwe_indexes`, an empty `frozenset()` means the token is not part of a multi-word expression.
- An empty string `""` as a tag means "no label". Metrics such as `top_n_accuracy` ignore it.
- Parsers apply label filtering **before** label validation. A filtered tag becomes `""`, so validation skips it.
- Filtering matches the whole tag string, so filtering `F2` does not remove `F2/O2`.
- `Z9` (punctuation) is validated like any other tag, so `label_validation` must include it. Parsers must not exempt it from validation. `PUNCT` is the legacy USAS tag name and should no longer be produced by parsers; `PUNCT` as a POS/UPOS value is a different tagset and is unaffected (see the README).
- Validation raises `ValueError` for any tag not in `label_validation`.
- Parsed datasets record `labels_removed=label_filter`. `EvaluationDataset.merge()` requires every dataset to have the same `labels_removed`.
- Some parsers hard-code fixes for known annotation errors at specific row indexes (for example `parsers/torch.py`). Keep these fixes.

## Data

- `Data/` is gitignored and holds local, unpublished annotated data. Never commit corpus data.
- Some full corpora under `tests/data/parsers/` are also gitignored. The NAACL 2015 Chinese and Portuguese corpora and the Irish ICC corpora can't be redistributed. Tests that need missing corpus files call `pytest.skip`, so a skipped test is expected, not a failure.
- Dataset licences are listed in the README.

## Python Guidelines

- Use match-case syntax instead of if/elif/else statements wherever possible.
- Use modern type hints with built-in generics (`list`, `dict`, `type`) and the union pipe (`|`) operator, rather than deprecated types from the `typing` module (e.g. `Optional`, `Union`, `Dict`, `List`, `Type`). Non-deprecated imports such as `Any`, `Literal`, `TypeVar` and `cast` are fine. The project supports Python 3.10+, so don't use PEP 695 generic syntax.
- Write strongly, statically typed code that passes static analysers such as mypy. This project uses [ty](https://docs.astral.sh/ty/).
- Use `pathlib.Path` methods for file system operations rather than the older `os.path` functions.
- Follow general best practices, including f-string formatting, comprehensions, context managers and PEP 8 compliance.

## Development

- Use `uv` for all commands. Never run bare `python` or `pip` commands; always go through `uv` to avoid environment drift. For example, use `uv add package` and `uv run script.py` rather than `pip install package` or `python script.py`. Useful commands:
  - `uv sync --all-extras` to set up the full environment. Both the `dev` and `docs` groups are default groups (`[tool.uv] default-groups` in `pyproject.toml`), so they are installed without `--all-groups`. The `docs` group is a default because `make lint` type-checks `docs/source/conf.py`, which imports Sphinx. `uv sync` is exact and removes anything not requested, so don't pass `--no-default-groups` or `--no-group docs` unless you mean to drop them.
  - `uv add` / `uv remove` to manage dependencies. For developer dependencies add `--dev`, e.g. `uv add --dev package`.
  - `uv run script.py` to run a script within the uv environment.
- Use `make` to run linting and tests (the targets are in `makefile`):
  - `make lint` first **auto-fixes files in place** with ruff (`--fix-only`), then runs `ruff check` and `ty check` over `src`, `tests`, `scripts`, `coding_style_format_example.py` and `docs/source/conf.py`.
  - `make test` runs pytest under coverage. Pytest runs `tests/` and the doctests in `src/` and `coding_style_format_example.py` (`--doctest-modules` with `NORMALIZE_WHITESPACE`).
  - `make docs` builds the HTML documentation into `docs/build` (it deletes any previous build first). It runs with `-W`, so any Sphinx warning fails the build, and it needs network access for the intersphinx inventories. `make docs-serve` serves it with live reload at http://127.0.0.1:8000 and rebuilds when `src` changes. Both use the `docs` dependency group.
- Document Python code with Google-style docstrings, and write all docstring examples in doctest style. They are executed by `make test`, so they must pass. Sphinx also renders these docstrings into the API pages, so formatting mistakes show up on the documentation site as well as in the doctests. Docstrings are reStructuredText: put a blank line before a list, use double backticks for literals containing spaces (``` ``"; "`` ```), and write maths with `.. math::` / `:math:` rather than `$$` or `\( \)`. List every Pydantic field, TypedDict key and Enum member in its class's `Attributes:` section: `docs/source/conf.py` skips autodoc's own entries for these to avoid duplicates, so an unlisted one is missing from the docs. See [coding_style_format_example.py](./coding_style_format_example.py) for an example of this format.
- Testing:
  - Tests mirror the `src` layout: `tests/parsers/`, `tests/metrics/`.
  - There is no `conftest.py`. Shared fixtures such as `get_test_data_directory` live in `tests/utils_test.py` and are imported explicitly.
  - Test data goes in `tests/data/parsers/<corpus>/`.
  - Parser tests are class-based (one class per parser) and use parametrised fixtures to cover each parse option, e.g. a fixture with `params=[None, {"T1"}]` that returns the expected dataset for each label filter, and one with `params=[False, True]` for whether label validation should raise. Other tests are plain functions, often using `@pytest.mark.parametrize`.

## General Coding Guidelines

- Prioritise readability. For example, avoid deeply nested if statements and nested list comprehensions.
