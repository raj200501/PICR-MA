from picr_ma.analysis.statistics import bootstrap_mean_ci


def test_bootstrap_mean_ci_constant_values():
    mean, lower, upper = bootstrap_mean_ci([3.0, 3.0, 3.0])
    assert mean == 3.0
    assert lower == 3.0
    assert upper == 3.0


def test_bootstrap_mean_ci_orders_interval():
    mean, lower, upper = bootstrap_mean_ci([1.0, 2.0, 4.0, 8.0], n_boot=2000, seed=7)
    assert lower <= mean <= upper
