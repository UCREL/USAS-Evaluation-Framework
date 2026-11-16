.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only src tests coding_style_format_example.py scripts docs/source/conf.py
	@uv run ruff check src tests coding_style_format_example.py scripts docs/source/conf.py
	@echo "Type checking with Ty"
	@uv run ty check src tests coding_style_format_example.py scripts docs/source/conf.py

# The GitHub workflow linters run as isolated, version-pinned uvx tools rather
# than dev dependencies, so they never constrain the project's Python version
# or dependency resolution. actionlint is upstream rhysd/actionlint; if it stays
# unmaintained, see the README about switching to the kjanat/actionlint fork.
ZIZMOR_VERSION ?= 1.32.0
ACTIONLINT_PY_VERSION ?= 1.7.12.25
SHELLCHECK_PY_VERSION ?= 0.11.0.1

.PHONY: lint-workflows
lint-workflows:
	@echo "Linting GitHub workflows with actionlint (and ShellCheck):"
	@uvx --from actionlint-py==$(ACTIONLINT_PY_VERSION) --with shellcheck-py==$(SHELLCHECK_PY_VERSION) actionlint
	@echo "Auditing GitHub workflows with zizmor:"
	@uvx zizmor@$(ZIZMOR_VERSION) .github/workflows

.PHONY: test
test:
	@echo "Testing with coverage"
	@uv run coverage run
	@uv run coverage report

.PHONY: remove-built-docs
remove-built-docs:
	@echo "Removing any pre-built documentation from ./docs/build"
	@rm -rf docs/build

# The command used to build the Sphinx documentation website.
# The GitHub docs workflow overrides it with
# "uv run --no-sync" after installing only the docs dependencies.
UV_RUN_DOCS ?= uv run --group docs

.PHONY: docs
docs: remove-built-docs
	@echo "Building the documentation into docs/build"
	@$(UV_RUN_DOCS) sphinx-build -W --keep-going -b html docs/source docs/build

.PHONY: docs-serve
docs-serve: remove-built-docs
	@echo "Serving the documentation locally with live reload at http://127.0.0.1:8000"
	@uv run --group docs sphinx-autobuild --host 127.0.0.1 --port 8000 --watch src --ignore docs/build docs/source docs/build
