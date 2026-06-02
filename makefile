.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only src tests coding_style_format_example.py scripts
	@uv run ruff check src tests coding_style_format_example.py scripts
	@echo "Type checking with Ty"
	@uv run ty check src tests coding_style_format_example.py scripts

.PHONY: test
test:
	@echo "Testing with coverage"
	@uv run coverage run
	@uv run coverage report
