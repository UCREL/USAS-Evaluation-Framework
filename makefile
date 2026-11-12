.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only src tests coding_style_format_example.py scripts docs/source/conf.py
	@uv run ruff check src tests coding_style_format_example.py scripts docs/source/conf.py
	@echo "Type checking with Ty"
	@uv run ty check src tests coding_style_format_example.py scripts docs/source/conf.py

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
