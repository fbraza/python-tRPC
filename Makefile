.PHONY : install
install:
	uv sync
	pre-commit install

.PHONY : upgrade
upgrade:
	uv sync

# --- CI instructions

.PHONY : lint
lint:
	pre-commit run --all-files

.PHONY : test
test:
	coverage erase; coverage run --source . -m pytest tests/ -svv --durations=5; coverage report --skip-empty --omit "tests/*" -m
