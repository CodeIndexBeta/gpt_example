PYTHON := python3
REQUIREMENTS := requirements.txt

PACKAGE_NAME := openAI_gpt
TEST_RUNNER := pytest

LINTER := flake8
LINT_FLAGS := --max-line-length=80

VENV := myenv

.PHONY: create-venv activate-venv deactivate-venv

install:
	pip install $(REQUIREMENTS)
	pip install -e

create-venv:
	$(PYTHON) -m venv $(VENV)

delete-venv:
	rm -rf $(VENV)

test:
	python -m $(TEST_RUNNER) tests/

clean:
	rm -rf build/ dist/ $(PACKAGE_NAME).egg-info/

lint:
	$(LINTER) $(LINT_FLAGS) $(PACKAGE_NAME) tests