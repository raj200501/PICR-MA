# Full pipeline:  make reproduce   (about 130 training runs; see README for timing)
# Steps run in order; finished training runs are skipped on re-runs.
PYTHON ?= python
WORKERS ?= 4

.NOTPARALLEL:
.PHONY: install test lint lambda select main crossplay-main order-shift reference overhead analyze figures paper reproduce

install:
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check picr_ma scripts tests
	$(PYTHON) -m ruff format --check picr_ma scripts tests

# 1. Select the PICR weight on seeds 100-104 (writes results/lambda/selection.json;
#    copy the chosen values into picr_lambda_by_env in configs/experiments.yaml).
lambda:
	$(PYTHON) scripts/run_experiments.py --suite lambda --workers $(WORKERS)
	$(PYTHON) scripts/crossplay.py --suite lambda --workers $(WORKERS)

select: lambda
	$(PYTHON) scripts/select_lambda.py

# 2. Main comparison on seeds 0-9 and all-pairs cross-play.
main: select
	$(PYTHON) scripts/run_experiments.py --suite main --workers $(WORKERS)

crossplay-main: main
	$(PYTHON) scripts/crossplay.py --suite main --workers $(WORKERS)

# Self-play when every agent sees its teammates in a different order.
order-shift: main
	$(PYTHON) scripts/order_shift.py --suite main --workers $(WORKERS)

# 3. Reference points and controlled cost measurements.
reference:
	$(PYTHON) scripts/reference_returns.py

overhead:
	$(PYTHON) scripts/benchmark_overhead.py

# 4. Every table, figure and number in the paper.
figures:
	$(PYTHON) scripts/make_method_figure.py

analyze: crossplay-main order-shift reference overhead figures
	$(PYTHON) scripts/analyze.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

reproduce: analyze paper
