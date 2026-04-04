import os

from src.analysis.aggregate import build_aggregate_outputs
from src.analysis.plots import make_publication_plots
from src.analysis.tables import make_tables
from scripts.evaluate import evaluate_run
from scripts.train import run_experiment


def test_analysis_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ACTIVE_PROFILE.txt").write_text("PROFILE_LEAN_SAFE\n", encoding="utf-8")
    run_experiment(env="matrix_coordination", method="ippo", seed=0, steps=12, tag="pytest")
    run_experiment(env="matrix_coordination", method="ippo", seed=1, steps=12, tag="pytest")
    evaluate_run(tmp_path / "results" / "run_pytest_matrix_coordination_ippo_s0", tmp_path / "results")
    build_aggregate_outputs(tmp_path / "results")
    make_tables(tmp_path / "results", tmp_path / "latex" / "tables")
    make_publication_plots(tmp_path / "results", tmp_path / "latex" / "figures")
    assert (tmp_path / "results" / "aggregate" / "final_metrics_for_paper.json").exists()
    assert (tmp_path / "latex" / "tables" / "main_results.tex").exists()
    assert (tmp_path / "latex" / "figures" / "method_overview.pdf").exists()
