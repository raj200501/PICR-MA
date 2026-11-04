"""End to end: every script in the pipeline, on a tiny two-suite configuration."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "defaults": {
        "total_steps": 200,
        "num_envs": 4,
        "hidden": 16,
        "eval_episodes": 8,
        "log_every": 1,
        "picr_lambda": 0.1,
        "ppo": {"epochs": 1, "minibatches": 2},
    },
    "suites": {
        "lambda": {
            "envs": ["simple_spread_v3", "simple_adversary_v3"],
            "methods": ["picr"],
            "picr_lambdas": [0.1, 1.0],
            "seeds": [100, 101],
        },
        "main": {
            "envs": ["simple_spread_v3", "simple_adversary_v3"],
            "methods": ["ippo", "picr", "deepsets", "order_aug", "order_rand"],
            "picr_lambda_by_env": {"simple_spread_v3": None, "simple_adversary_v3": None},
            "seeds": [0, 1, 2],
        },
    },
}


def _run(script: str, *args) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True, cwd=ROOT
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    results, paper, config = tmp_path / "results", tmp_path / "paper", tmp_path / "experiments.yaml"
    (paper / "figures").mkdir(parents=True)
    config.write_text(yaml.safe_dump(CONFIG))
    common = ["--results-dir", results]

    _run("run_experiments.py", "--suite", "lambda", "--config", config, *common, "--workers", 2)
    _run("crossplay.py", "--suite", "lambda", *common, "--episodes", 8, "--workers", 2)
    _run("select_lambda.py", *common)
    selection = json.loads((results / "lambda" / "selection.json").read_text())
    CONFIG["suites"]["main"]["picr_lambda_by_env"] = {env: v["selected_lambda"] for env, v in selection.items()}
    config.write_text(yaml.safe_dump(CONFIG))

    _run("run_experiments.py", "--suite", "main", "--config", config, *common, "--workers", 2)
    _run("crossplay.py", "--suite", "main", *common, "--episodes", 8, "--workers", 2)
    _run("order_shift.py", "--suite", "main", *common, "--episodes", 8, "--workers", 2)
    _run("reference_returns.py", *common, "--episodes", 16)
    _run("benchmark_overhead.py", *common, "--config", config, "--repeats", 1)
    _run("analyze.py", *common, "--paper-dir", paper, "--config", config, "--png-dir", tmp_path / "png", "--n-perm", 50)

    numbers = (paper / "tables" / "numbers.tex").read_text()
    for macro in [
        "SpreadIPPOGap",
        "SpreadPICRdGapph",
        "SpreadOrderRandShift",
        "AdversaryDeepSetsSlot",
        "SpreadAddRsq",
        "SpreadLambda",
        "RefAssign",
    ]:
        assert f"\\newcommand{{\\{macro}}}" in numbers, macro
    for table in ["main_results", "comparison", "order_measures", "overhead"]:
        assert (paper / "tables" / f"{table}.tex").exists()
    report = json.loads((results / "analysis.json").read_text())
    assert set(report["estimates"]["simple_spread_v3"]) == set(CONFIG["suites"]["main"]["methods"])
    assert (tmp_path / "png" / "order_shift.png").exists()

    # The analysis refuses PICR runs trained with a weight other than the selected one.
    selection["simple_spread_v3"]["selected_lambda"] = 123.0
    (results / "lambda" / "selection.json").write_text(json.dumps(selection))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze.py"),
            *map(str, common),
            "--paper-dir",
            str(paper),
            "--config",
            str(config),
            "--n-perm",
            "5",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0 and "selects" in proc.stderr
