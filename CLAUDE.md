# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository. `AGENTS.md` is a symlink to this file, so edit this file only.

## Project Overview

An evaluation framework for USAS semantic tagging, from UCREL at Lancaster University. Parsers turn each annotated corpus into an `EvaluationDataset`, and metrics are computed by comparing a gold dataset with a predicted one. So far only top-N accuracy (micro and macro) is implemented. The other metrics and splits listed in the README are planned but not yet built.

## Layout

- `src/usas_evaluation_framework/dataset.py` holds the Pydantic v2 data models: `EvaluationTexts` (one text) and `EvaluationDataset` (a collection, with `merge()` and `stats()`).
- `src/usas_evaluation_framework/data_utils.py` parses USAS tags (`parse_usas_token_group`) and loads the packaged tag map at `data/usas/usas_mapper.yaml` (`load_usas_mapper`).
- `src/usas_evaluation_framework/parsers/` has one parser per corpus, each subclassing `BaseParser` (`parsers/base.py`). `parse(dataset_path, label_validation, label_filter, dataset_name, language)` is a static method that returns an `EvaluationDataset`. New corpora should follow this pattern.
- `src/usas_evaluation_framework/metrics/` holds the evaluation metrics.
- `scripts/` holds Typer CLIs that convert annotated Excel/TSV files to CSV. They need the `excel-conversion` extra, and the README explains how to use them.
- `notebooks/` holds notebooks that evaluate PyMUSAS models on the Hugging Face `ucrelnlp/USAS-WSD` dataset and explore dataset statistics. They need the `notebooks` extra.

## Domain Conventions

- `EvaluationTexts.semantic_tags` is `list[list[str]]`: one list per token, ordered from most to least likely. Gold data has a single tag per token. In `mwe_indexes`, an empty `frozenset()` means the token is not part of a multi-word expression.
- An empty string `""` as a tag means "no label". Metrics such as `top_n_accuracy` ignore it.
- Parsers apply label filtering **before** label validation. A filtered tag becomes `""`, so validation skips it.
- Filtering matches the whole tag string, so filtering `F2` does not remove `F2/O2`.
- `PUNCT` always passes validation. Punctuation is moving to `Z9`, and `PUNCT` is the legacy name (see the README).
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
  - `uv sync --all-extras --all-groups` to set up the full environment. Only the dev group is installed by default, so `--all-groups` is needed for the docs group. `uv sync` is exact and removes anything not requested.
  - `uv add` / `uv remove` to manage dependencies. For developer dependencies add `--dev`, e.g. `uv add --dev package`.
  - `uv run script.py` to run a script within the uv environment.
- Use `make` to run linting and tests (the targets are in `makefile`):
  - `make lint` first **auto-fixes files in place** with ruff (`--fix-only`), then runs `ruff check` and `ty check` over `src`, `tests`, `scripts` and `coding_style_format_example.py`.
  - `make test` runs pytest under coverage. Pytest runs `tests/` and the doctests in `src/` and `coding_style_format_example.py` (`--doctest-modules` with `NORMALIZE_WHITESPACE`).
- Document Python code with Google-style docstrings, and write all docstring examples in doctest style. They are executed by `make test`, so they must pass. See [coding_style_format_example.py](./coding_style_format_example.py) for an example of this format.
- Testing:
  - Tests mirror the `src` layout: `tests/parsers/`, `tests/metrics/`.
  - There is no `conftest.py`. Shared fixtures such as `get_test_data_directory` live in `tests/utils_test.py` and are imported explicitly.
  - Test data goes in `tests/data/parsers/<corpus>/`.
  - Parser tests are class-based (one class per parser) and use parametrised fixtures to cover each parse option, e.g. a fixture with `params=[None, {"T1"}]` that returns the expected dataset for each label filter, and one with `params=[False, True]` for whether label validation should raise. Other tests are plain functions, often using `@pytest.mark.parametrize`.

## General Coding Guidelines

- Prioritise readability. For example, avoid deeply nested if statements and nested list comprehensions.
