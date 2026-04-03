from src.analysis.aggregate import aggregate_results


def test_aggregate_empty(tmp_path):
    rows = aggregate_results(tmp_path)
    assert rows == []
