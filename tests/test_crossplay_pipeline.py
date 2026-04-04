import gzip
import os

from scripts.evaluate import evaluate_run
from scripts.train import run_experiment


def test_crossplay_evaluation_populates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ACTIVE_PROFILE.txt").write_text("PROFILE_LEAN_SAFE\n", encoding="utf-8")
    run_experiment(env="matrix_coordination", method="ippo", seed=0, steps=12, tag="pytest")
    run_experiment(env="matrix_coordination", method="ippo", seed=1, steps=12, tag="pytest")
    run_dir = tmp_path / "results" / "run_pytest_matrix_coordination_ippo_s0"
    evaluate_run(run_dir, tmp_path / "results")
    with gzip.open(run_dir / "eval_crossplay.csv.gz", "rt", encoding="utf-8") as handle:
        text = handle.read()
    assert "crossplay_return" in text
