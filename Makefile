.PHONY: help venv install install-full test lint fmt doctor demo clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:            ## Create .venv
	python3 -m venv .venv

install: venv    ## Install Jarvis (base)
	.venv/bin/pip install -e .

install-full: venv ## Install with Google + document extras
	.venv/bin/pip install -e ".[all]"

test:            ## Run the test suite (offline, no model needed)
	.venv/bin/python -m pytest -q

lint:            ## Lint with ruff
	.venv/bin/python -m ruff check jarvis_local tests

fmt:             ## Format with ruff
	.venv/bin/python -m ruff format jarvis_local tests

doctor:          ## Check this machine's setup
	.venv/bin/jarvis doctor

demo:            ## Guided tour with the mock backend
	.venv/bin/jarvis demo

clean:           ## Remove build artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
