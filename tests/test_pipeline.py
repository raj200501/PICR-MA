"""End to end: every script in the pipeline, on a tiny two-suite configuration."""

import json
import re
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
            "methods": ["picr", "ippo"],
            "picr_lambdas": [0.1, 1.0],
            "seeds": [100, 101],
        },
        "main": {
            "envs": ["simple_spread_v3", "simple_adversary_v3"],
            "methods": ["ippo", "picr", "deepsets", "order_aug", "order_rand"],
            "picr_lambda_by_env": {"simple_spread_v3": None, "simple_adversary_v3": None},
            "seeds": [0, 1, 2],
        },
        "long": {
            "envs": ["simple_spread_v3"],
            "methods": ["ippo", "picr", "deepsets"],
            "total_steps": 400,
            "picr_lambda_by_env": {"simple_spread_v3": None},
            "seeds": [20, 21],
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
    (results / "tuning").mkdir(parents=True)
    (results / "tuning" / "ippo_sweep.jsonl").write_text((ROOT / "results" / "tuning" / "ippo_sweep.jsonl").read_text())
    (paper / "figures").mkdir(parents=True)
    config.write_text(yaml.safe_dump(CONFIG))
    common = ["--results-dir", results]

    _run("run_experiments.py", "--suite", "lambda", "--config", config, *common, "--workers", 2)
    _run("crossplay.py", "--suite", "lambda", *common, "--episodes", 8, "--workers", 2)
    _run("select_lambda.py", *common)
    selection = json.loads((results / "lambda" / "selection.json").read_text())
    CONFIG["suites"]["main"]["picr_lambda_by_env"] = {env: v["selected_lambda"] for env, v in selection.items()}
    CONFIG["suites"]["long"]["picr_lambda_by_env"] = {"simple_spread_v3": selection["simple_spread_v3"]["selected_lambda"]}
    config.write_text(yaml.safe_dump(CONFIG))

    _run("run_experiments.py", "--suite", "main", "--config", config, *common, "--workers", 2)
    _run("crossplay.py", "--suite", "main", *common, "--episodes", 8, "--workers", 2)
    _run("order_shift.py", "--suite", "main", *common, "--episodes", 8, "--workers", 2)
    _run("run_experiments.py", "--suite", "long", "--config", config, *common, "--workers", 2)
    _run("crossplay.py", "--suite", "long", *common, "--episodes", 8, "--workers", 2)
    _run("order_shift.py", "--suite", "long", *common, "--episodes", 8, "--workers", 2)
    _run("reference_returns.py", *common, "--episodes", 16)
    _run("positive_control.py", *common, "--episodes", 8)
    _run("benchmark_env.py", *common, "--episodes", 20)
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
        "LongSpreadDeepSetsGap",
        "LongSpreadPICRdGapp",
        "SpreadSlopePos",
        "ComputeMainHours",
        "SpreadDeepSetsActorParams",
        "SpreadIPPOSPsd",
        "TuneAdversaryOther",
    ]:
        assert f"\\newcommand{{\\{macro}}}" in numbers, macro
    # Every generated macro the paper uses is defined.
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", numbers))
    used = {
        m
        for tex in (ROOT / "paper").glob("**/*.tex")
        if "tables" not in tex.parts
        for m in re.findall(r"\\((?:Spread|Adversary|Long|Ref|PC|EnvSpeedup|Tune|Compute)\w*)", tex.read_text())
    }
    assert used <= defined, sorted(used - defined)
    for table in ["main_results", "comparison", "order_measures", "overhead", "long_budget", "lambda_sweep", "compute"]:
        assert (paper / "tables" / f"{table}.tex").exists()
    report = json.loads((results / "analysis.json").read_text())
    assert set(report["estimates"]["simple_spread_v3"]) == set(CONFIG["suites"]["main"]["methods"])
    assert report["provenance"]["git_commit"] and set(report["compute"]) == {"main", "lambda", "long"}
    assert json.loads((results / "main" / "simple_spread_v3" / "crossplay.provenance.json").read_text())["split"] == "test"
    assert (tmp_path / "png" / "order_shift.png").exists()

    # An evaluation output missing from results/ is an error, not a silently skipped table.
    (results / "overhead.json").rename(results / "overhead.moved")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze.py"),
            *map(str, common),
            "--paper-dir",
            str(paper),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0 and "overhead.json" in proc.stderr
    (results / "overhead.moved").rename(results / "overhead.json")

    # A selection that disagrees with the configured PICR weights is refused.
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
    assert proc.returncode != 0 and "disagrees" in proc.stderr
