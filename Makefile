# make analyze / make paper   rebuild every table, figure, number and the PDF from the committed results/
# make reproduce              re-run the whole study: training, evaluation, analysis and paper (see README for timing)
# Training and evaluation steps skip work that is already finished.
PYTHON ?= python
WORKERS ?= 4

.NOTPARALLEL:
.PHONY: install test lint tuning lambda select main crossplay-main order-shift long reference positive-control \
	env-speed overhead evaluate figures analyze paper reproduce

install:
	$(PYTHON) -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check picr_ma scripts tests
	$(PYTHON) -m ruff format --check picr_ma scripts tests

# 0. The IPPO hyperparameter sweep behind docs/tuning.md (not part of reproduce).
tuning:
	$(PYTHON) scripts/tuning_sweep.py --workers $(WORKERS)
	$(PYTHON) scripts/tuning_table.py

# 1. Select the PICR weight on seeds 100-104, scored on validation episodes.
#    Writes results/lambda/selection.json; configs/experiments.yaml must name the
#    selected values in picr_lambda_by_env, and the main suite refuses to run otherwise.
lambda:
	$(PYTHON) scripts/run_experiments.py --suite lambda --workers $(WORKERS)
	$(PYTHON) scripts/crossplay.py --suite lambda --workers $(WORKERS)

select: lambda
	$(PYTHON) scripts/select_lambda.py

# 2. Main comparison on seeds 0-9, scored on test episodes.
main: select
	$(PYTHON) scripts/run_experiments.py --suite main --workers $(WORKERS)

crossplay-main: main
	$(PYTHON) scripts/crossplay.py --suite main --workers $(WORKERS)

order-shift: main
	$(PYTHON) scripts/order_shift.py --suite main --workers $(WORKERS)

# 3. Reference controllers, instrument checks with scripted policies, and cost
#    measurements (env-speed and overhead depend on the machine; run them on an idle one).
reference:
	$(PYTHON) scripts/reference_returns.py

positive-control:
	$(PYTHON) scripts/positive_control.py

env-speed:
	$(PYTHON) scripts/benchmark_env.py

overhead:
	$(PYTHON) scripts/benchmark_overhead.py

# Supplementary training-budget check: twice the steps on simple_spread, new seeds.
long: select
	$(PYTHON) scripts/run_experiments.py --suite long --workers $(WORKERS)
	$(PYTHON) scripts/crossplay.py --suite long --workers $(WORKERS)
	$(PYTHON) scripts/order_shift.py --suite long --workers $(WORKERS)

evaluate: crossplay-main order-shift long reference positive-control env-speed overhead

# 4. Tables, figures and numbers from results/ (no training or evaluation).
figures:
	$(PYTHON) scripts/make_method_figure.py

analyze: figures
	$(PYTHON) scripts/analyze.py

paper: analyze
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

reproduce: evaluate paper
