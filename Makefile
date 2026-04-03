PYTHONPATH=. 

install:
	python -m pip install -r requirements.txt

preflight:
	PYTHONPATH=. python scripts/preflight.py

smoke:
	PYTHONPATH=. python scripts/install_check.py
	PYTHONPATH=. python scripts/smoke_test.py

stage1:
	PYTHONPATH=. python scripts/run_stage1.py

stage2:
	PYTHONPATH=. python scripts/run_stage2.py

plots:
	PYTHONPATH=. python scripts/make_plots.py

tables:
	PYTHONPATH=. python scripts/make_tables.py

paper:
	PYTHONPATH=. python scripts/make_paper.py

clean_artifacts:
	PYTHONPATH=. python scripts/cleanup_artifacts.py

report:
	PYTHONPATH=. python scripts/report_status.py
