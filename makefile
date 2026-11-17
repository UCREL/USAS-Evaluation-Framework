# The files that are linted and type checked.
LINT_PATHS := src tests coding_style_format_example.py scripts docs/source/conf.py

# Ty type checks once for each of these Python versions: the oldest and newest
# the project supports. The oldest finds features that are too new, and the
# newest finds APIs that have since been removed. --python-version only changes
# which type definitions ty uses, so no extra Python installs are needed. Keep
# in sync with requires-python in pyproject.toml and the CI test matrix.
TY_PYTHON_VERSIONS ?= 3.10 3.14

# Type checking needs every import in the linted files to resolve, but that
# doesn't need every extra:
# - excel-conversion is needed, because the scripts import it.
# - notebooks is not needed, because notebooks/ isn't linted.
# The docs group, which docs/source/conf.py needs, is a default group, so uv
# run includes it anyway. Keep in sync with the CI lint job's install step.
UV_RUN_LINT := uv run --extra excel-conversion

# Auto-fixes files in place with Ruff, then runs the same checks as CI.
.PHONY: lint
lint:
	@echo "Auto-fixing with Ruff:"
	@$(UV_RUN_LINT) ruff check --fix-only $(LINT_PATHS)
	@$(MAKE) --no-print-directory lint-check

# Checks only, without changing any files. The CI workflow uses this, because
# auto-fixing first would hide fixable errors instead of failing on them.
.PHONY: lint-check
lint-check:
	@echo "Linting with Ruff:"
	@$(UV_RUN_LINT) ruff check $(LINT_PATHS)
	@for version in $(TY_PYTHON_VERSIONS); do \
		echo "Type checking with Ty for Python $$version:"; \
		$(UV_RUN_LINT) ty check --python-version "$$version" $(LINT_PATHS) || exit 1; \
	done

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

.PHONY: docs
docs: remove-built-docs
	@echo "Building the documentation into docs/build"
	@uv run --group docs sphinx-build -W --keep-going -b html docs/source docs/build

.PHONY: docs-serve
docs-serve: remove-built-docs
	@echo "Serving the documentation locally with live reload at http://127.0.0.1:8000"
	@uv run --group docs sphinx-autobuild --host 127.0.0.1 --port 8000 --watch src --ignore docs/build docs/source docs/build
