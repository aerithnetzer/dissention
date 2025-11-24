#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = dissention
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	

.PHONY: analysis
analysis:
	uv run python -m cudf.pandas dissent/modeling/analysis.py

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Make dataset
.PHONY: data
dataset:
	if [ ! -f data/raw/opinions-2024-12-31.csv.bz2 ]; then \
		echo Opinions file not found! Downloading the file. \(This could take a while.\); \
		wget --directory-prefix=data/raw https://storage.courtlistener.com/bulk-data/opinions-2024-12-31.csv.bz2; \
		echo Opinions file downloaded!; \
	fi
	if [ ! -f data/raw/opinion-clusters-2024-12-31.csv.bz2 ]; then \
		echo Opinions cluster file not found! Downloading the file. \(This could take a while.\);\
		wget --directory-prefix=data/raw https://storage.courtlistener.com/bulk-data/opinion-clusters-2024-12-31.csv.bz2; \
		echo Opinions cluster file downloaded!; \
	fi
	if [ ! -f data/raw/dockets-2024-12-31.csv.bz2 ]; then \
		echo Opinions cluster file not found! Downloading the file. \(This could take a while.\); \
		wget --directory-prefix=data/raw  https://storage.courtlistener.com/bulk-data/dockets-2024-12-31.csv.bz2; \
		echo Opinions cluster file downloaded!; \
	fi
	uv run python -m cudf.pandas dissent/dataset.py

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
