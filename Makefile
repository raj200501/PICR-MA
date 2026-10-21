PYTHON ?= python
WORKERS ?= 4

.PHONY: install test lint experiments main lambda crossplay analyze paper reproduce

install:
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check picr_ma scripts tests
	$(PYTHON) -m ruff format --check picr_ma scripts tests

# Train every run in a suite (finished runs are skipped).
main:
	$(PYTHON) scripts/run_experiments.py --suite main --workers $(WORKERS)
lambda:
	$(PYTHON) scripts/run_experiments.py --suite lambda --workers $(WORKERS)

crossplay:
	$(PYTHON) scripts/crossplay.py --suite main --workers $(WORKERS)
	$(PYTHON) scripts/crossplay.py --suite lambda --workers $(WORKERS)

# Regenerate every table, figure and number in the paper from results/.
analyze:
	$(PYTHON) scripts/analyze.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

experiments: main lambda crossplay

reproduce: experiments analyze paper
