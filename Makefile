PYTHON ?= ./.venv/bin/python
ENV = PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp

install:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt

preflight:
	$(ENV) $(PYTHON) scripts/preflight.py

smoke:
	$(ENV) $(PYTHON) scripts/install_check.py
	$(ENV) $(PYTHON) scripts/smoke_test.py

stage1:
	$(ENV) $(PYTHON) scripts/run_stage1.py

stage2:
	$(ENV) $(PYTHON) scripts/run_stage2.py

plots:
	$(ENV) $(PYTHON) scripts/make_plots.py

tables:
	$(ENV) $(PYTHON) scripts/make_tables.py

paper:
	$(ENV) $(PYTHON) scripts/make_paper.py

clean_artifacts:
	$(ENV) $(PYTHON) scripts/cleanup_artifacts.py

report:
	$(ENV) $(PYTHON) scripts/report_status.py
