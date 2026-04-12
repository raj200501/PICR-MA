import os

from scripts.train import run_experiment


def test_matrix_training_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ACTIVE_PROFILE.txt").write_text("PROFILE_LEAN_SAFE\n", encoding="utf-8")
    run_id = run_experiment(env="matrix_coordination", method="ippo", seed=0, steps=12, tag="pytest")
    run_dir = tmp_path / "results" / run_id
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "best.ckpt").exists()
