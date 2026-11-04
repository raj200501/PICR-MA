"""The committed results/ belong to configs/experiments.yaml and to the current code.

* Every run directory under results/ is one the config file defines, and its
  stored config is exactly the config the file now produces.
* For one run of every (suite, environment, method) group, retraining up to the
  first logged update (5 PPO updates) with the current code reproduces that
  run's committed training log.

The rerun is bitwise identical on the machine type that produced the results.
Other CPUs can sum float32 products in a different order, which perturbs the
weights very slightly. Batch means of large terms (return, value loss, entropy,
permutation sensitivity) still agree to well under 1%. The PPO policy loss,
approximate KL and clip fraction are small averages of terms that nearly
cancel, so in relative terms they move more (0.7% was seen on a GitHub runner).
The tolerances still catch real changes: another seed, 9 instead of 10 epochs,
or a 10% higher learning rate each moved at least two fields of every run
checked beyond them.
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


# Relative tolerance per logged field (see the module docstring).
TOLERANCE = {"env_steps": 0.0, "policy_loss": 0.05, "approx_kl": 0.05, "clip_frac": 0.05}
DEFAULT_TOLERANCE = 5e-3


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
        if not math.isclose(float(rerun[k]), float(committed[k]), rel_tol=TOLERANCE.get(k, DEFAULT_TOLERANCE), abs_tol=1e-6)
    }
    assert not mismatched, f"committed vs. rerun: {mismatched}"
