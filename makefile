.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only src tests
	@uv run ruff check src tests
	@echo "Type checking with Ty"
	@uv run ty check src tests

.PHONY: test
test:
	@echo "Testing with coverage"
	@uv run coverage run
	@uv run coverage report
