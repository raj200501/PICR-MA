"""Expand ``configs/experiments.yaml`` into concrete training runs.

A suite lists environments, methods, seeds and (optionally) a list of PICR
weights. Every combination becomes one run directory:

    results/<suite>/<env>/<method>[_lam<lambda>]/seed<k>/

A run is skipped when its ``summary.json`` already exists, so interrupted
suites resume where they stopped.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from picr_ma.ppo import PPOConfig
from picr_ma.train import TrainConfig, train


@dataclass(frozen=True)
class Run:
    suite: str
    config: TrainConfig
    out_dir: Path

    @property
    def done(self) -> bool:
        return (self.out_dir / "summary.json").exists()


def _variant(method: str, lam: float | None) -> str:
    return method if lam is None else f"{method}_lam{lam:g}"


def load_suites(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


TRAIN_KEYS = ("total_steps", "num_envs", "hidden", "picr_lambda", "eval_episodes", "log_every")


def expand_suite(spec: dict, suite: str, results_dir: str | Path = "results") -> list[Run]:
    defaults = spec.get("defaults", {})
    suite_spec = spec["suites"][suite]
    merged = {**defaults, **suite_spec}
    base = {k: merged[k] for k in TRAIN_KEYS if k in merged}
    ppo = PPOConfig(**{**defaults.get("ppo", {}), **suite_spec.get("ppo", {})})
    runs = []
    for env, method, lam, seed in itertools.product(
        suite_spec["envs"], suite_spec["methods"], suite_spec.get("picr_lambdas", [None]), suite_spec["seeds"]
    ):
        fields = base if lam is None else {**base, "picr_lambda": lam}
        config = TrainConfig(env=env, method=method, seed=seed, ppo=ppo, **fields)
        out_dir = Path(results_dir) / suite / env / _variant(method, lam) / f"seed{seed}"
        runs.append(Run(suite=suite, config=config, out_dir=out_dir))
    return runs


def execute(run: Run) -> dict:
    if run.done:
        return json.loads((run.out_dir / "summary.json").read_text())
    return train(run.config, run.out_dir)
