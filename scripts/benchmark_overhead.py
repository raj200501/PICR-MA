#!/usr/bin/env python
"""Controlled measurement of per-update training cost and inference cost by method.

    python scripts/benchmark_overhead.py

Wall-clock times of runs trained in parallel are affected by machine load, so
the cost of each method is measured here in a single process on one thread:
one PPO update (default config from configs/experiments.yaml) on an identical
synthetic batch, and one batched forward pass at inference time. PICR uses
each environment's selected weight (a run that does not compute the penalty is
an error). Reports medians over repeats and writes results/overhead.json.
Order augmentation and randomization permute inputs while acting, not during
the update, so their update cost equals IPPO's by construction.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from picr_ma.envs.mpe import env_spec
from picr_ma.experiments import expand_suite, load_suites
from picr_ma.networks import ActorCritic
from picr_ma.ppo import PPOConfig, ppo_update
from picr_ma.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def _median_seconds(fn, repeats: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiments.yaml")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    suite = load_suites(args.config)
    picr_runs = {run.config.env: run.config for run in expand_suite(suite, "main", args.results_dir, methods=["picr"])}
    config = next(iter(picr_runs.values()))
    out = {"ppo": asdict(config.ppo), "num_envs": config.num_envs, "torch_threads": torch.get_num_threads(), "envs": {}}
    for env in ("simple_spread_v3", "simple_adversary_v3"):
        picr_lambda = picr_runs[env].picr_lambda
        spec = env_spec(env)
        n = config.num_envs * spec.horizon * spec.n_team
        rng = np.random.default_rng(0)
        batch = {
            "obs": rng.normal(size=(n, spec.obs_dim)).astype(np.float32),
            "actions": rng.integers(0, spec.n_actions, n),
            "log_probs": np.full(n, -np.log(spec.n_actions), dtype=np.float32),
            "advantages": rng.normal(size=n).astype(np.float32),
            "returns": rng.normal(size=n).astype(np.float32),
        }
        results = {}
        for method in ("ippo", "picr", "deepsets", "order_aug", "order_rand"):
            torch.manual_seed(0)
            model = ActorCritic(
                spec.obs_dim, spec.n_actions, encoder="deepsets" if method == "deepsets" else "mlp", slicing=spec.slicing
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr)
            ppo = PPOConfig(**{**asdict(config.ppo), "picr_lambda": picr_lambda if method == "picr" else 0.0})
            stats = ppo_update(model, optimizer, batch, ppo, rng, spec.slicing)
            if (stats["picr_loss"] > 0) != (method == "picr"):
                raise RuntimeError(f"PICR penalty {'missing' if method == 'picr' else 'active'} in the {method} benchmark")
            update = _median_seconds(partial(ppo_update, model, optimizer, batch, ppo, rng, spec.slicing), repeats=args.repeats)
            obs = batch["obs"][: config.num_envs * spec.n_team]
            forward = _median_seconds(partial(model.action_probs, obs), repeats=200)
            results[method] = {
                "update_seconds": update,
                "forward_ms": 1e3 * forward,
                "actor_params": sum(
                    p.numel() for name, p in model.named_parameters() if not name.startswith(("critic", "value"))
                ),
            }
        for method in results:
            results[method]["update_vs_ippo"] = results[method]["update_seconds"] / results["ippo"]["update_seconds"]
            results[method]["forward_vs_ippo"] = results[method]["forward_ms"] / results["ippo"]["forward_ms"]
        out["envs"][env] = {"picr_lambda": picr_lambda, "methods": results}
        print(env)
        for method, r in results.items():
            print(
                f"  {method:>9}: update {r['update_seconds']:.3f}s ({r['update_vs_ippo']:.2f}x)  "
                f"forward {r['forward_ms']:.3f}ms ({r['forward_vs_ippo']:.2f}x)  actor params {r['actor_params']}"
            )
    out["provenance"] = provenance()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "overhead.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
