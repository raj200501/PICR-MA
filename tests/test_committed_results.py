"""The committed results/ belong to configs/experiments.yaml and to the current code.

* Every run directory under results/ is one the config file defines, and its
  stored config is exactly the config the file now produces.
* For one run of every (suite, environment, method) group, retraining up to the
  first logged update with the current code reproduces that run's committed
  training log. Bitwise equality holds on the machine type that produced the
  results; other CPUs can change float32 summation order, hence the tolerance.
"""

from __future__ import annotations

import csv
import dataclasses
import gzip
import math
from pathlib import Path

import pytest

from picr_ma.experiments import expand_suite, load_suites
from picr_ma.train import train

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPEC = load_suites(ROOT / "configs" / "experiments.yaml")


def _finished_runs():
    for suite in SPEC["suites"]:
        for run in expand_suite(SPEC, suite, RESULTS):
            if (run.out_dir / "summary.json").exists():
                yield run


def _first_logged_row(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", newline="") as f:
        return next(csv.DictReader(f))


def _representatives():
    """The lowest seed of each main-suite method, and of one PICR weight per environment in the lambda suite."""
    chosen = {}
    for run in sorted(_finished_runs(), key=lambda r: r.config.seed):
        key = (run.suite, run.config.env, run.config.method if run.suite == "lambda" else run.out_dir.parent.name)
        chosen.setdefault(key, run)
    return [chosen[k] for k in sorted(chosen)]


def test_every_committed_run_is_defined_by_the_config():
    expected = {run.out_dir for suite in SPEC["suites"] for run in expand_suite(SPEC, suite, RESULTS)}
    on_disk = {p.parent for p in RESULTS.glob("*/*/*/seed*/summary.json")}
    assert on_disk, "no committed runs found"
    assert not on_disk - expected, f"runs not defined by configs/experiments.yaml: {sorted(on_disk - expected)[:5]}"
    for run in _finished_runs():
        assert run.done  # raises if the stored config differs from the one the config file produces


@pytest.mark.slow
@pytest.mark.parametrize("run", _representatives(), ids=lambda r: str(r.out_dir.relative_to(RESULTS)))
def test_current_code_reproduces_committed_training(run, tmp_path):
    committed = _first_logged_row(run.out_dir / "metrics.csv.gz")
    short = dataclasses.replace(run.config, total_steps=int(committed["env_steps"]), eval_episodes=2)
    train(short, tmp_path)
    rerun = _first_logged_row(tmp_path / "metrics.csv.gz")
    assert rerun.keys() == committed.keys()
    mismatched = {
        k: (committed[k], rerun[k])
        for k in committed
        if not math.isclose(float(rerun[k]), float(committed[k]), rel_tol=1e-3, abs_tol=1e-5)
    }
    assert not mismatched, f"committed vs. rerun: {mismatched}"
