"""Expand ``configs/experiments.yaml`` into concrete training runs.

A suite lists environments, methods, seeds and optionally either a list of
PICR weights to sweep (``picr_lambdas``) or one selected weight per
environment (``picr_lambda_by_env``). Every combination becomes one run
directory:

    results/<suite>/<env>/<method>[_lam<lambda>]/seed<k>/

A run is skipped when it already finished with the same config, so interrupted
suites resume where they stopped; a finished run whose stored config differs
from the current one is an error rather than being silently reused.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
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
        """True if this run finished with the current config; raises if it finished with a different one."""
        if not (self.out_dir / "summary.json").exists():
            return False
        stored = json.loads((self.out_dir / "config.json").read_text())
        if stored != json.loads(json.dumps(asdict(self.config))):
            raise RuntimeError(
                f"{self.out_dir} was trained with a different config than configs/experiments.yaml now specifies; "
                "move the directory away to retrain it"
            )
        return True


def _variant(method: str, lam: float | None) -> str:
    return method if lam is None else f"{method}_lam{lam:g}"


def load_suites(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


TRAIN_KEYS = ("total_steps", "num_envs", "hidden", "picr_lambda", "eval_episodes", "log_every")


def expand_suite(spec: dict, suite: str, results_dir: str | Path = "results", methods: list[str] | None = None) -> list[Run]:
    defaults = spec.get("defaults", {})
    suite_spec = spec["suites"][suite]
    merged = {**defaults, **suite_spec}
    base = {k: merged[k] for k in TRAIN_KEYS if k in merged}
    ppo = PPOConfig(**{**defaults.get("ppo", {}), **suite_spec.get("ppo", {})})
    runs = []
    for env, method, lam, seed in itertools.product(
        suite_spec["envs"],
        [m for m in suite_spec["methods"] if methods is None or m in methods],
        suite_spec.get("picr_lambdas", [None]),
        suite_spec["seeds"],
    ):
        fields = dict(base)
        if lam is not None:
            fields["picr_lambda"] = lam
        elif env in suite_spec.get("picr_lambda_by_env", {}):
            selected = suite_spec["picr_lambda_by_env"][env]
            if selected is None and method == "picr":
                raise ValueError(f"No PICR weight selected for {env} yet; run scripts/select_lambda.py first")
            fields["picr_lambda"] = selected if selected is not None else fields.get("picr_lambda", 0.1)
        if method != "picr":
            fields["picr_lambda"] = 0.0  # unused by other methods; fixed so their configs do not depend on it
        config = TrainConfig(env=env, method=method, seed=seed, ppo=ppo, **fields)
        out_dir = Path(results_dir) / suite / env / _variant(method, lam) / f"seed{seed}"
        runs.append(Run(suite=suite, config=config, out_dir=out_dir))
    return runs


def execute(run: Run) -> dict:
    if run.done:
        return json.loads((run.out_dir / "summary.json").read_text())
    return train(run.config, run.out_dir)
