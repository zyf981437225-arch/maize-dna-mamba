import numpy as np

from src.m6a_sparse import (
    WindowCandidateSystem,
    average_precision,
    observed_site_metrics,
    sliding_window_starts,
    solve_nonnegative_lasso,
)


def _synthetic_sequence():
    sequence = list("C" * 40)
    for position in [4, 9, 17, 28, 35]:
        sequence[position] = "A"
    return "".join(sequence)


def test_dense_window_starts_cover_tail_once():
    assert sliding_window_starts(15, 8, 4).tolist() == [0, 4, 7]
    assert sliding_window_starts(4, 8, 1).tolist() == [0]
    assert sliding_window_starts(0, 8, 1).tolist() == []


def test_implicit_window_operator_has_correct_adjoint():
    system = WindowCandidateSystem.from_sequence(_synthetic_sequence(), 12, 2)
    site_scores = np.asarray([0.1, 0.7, 0.3, 0.9, 0.2])
    window_values = np.linspace(-0.5, 0.5, system.n_windows)
    assert np.allclose(
        np.dot(system.forward(site_scores), window_values),
        np.dot(site_scores, system.adjoint(window_values)),
    )


def test_oracle_sparse_recovery_finds_observed_adenosines():
    observed = [9, 28]
    system = WindowCandidateSystem.from_sequence(_synthetic_sequence(), 12, 1)
    counts = system.counts_for_positions(observed)
    scores, diagnostics = solve_nonnegative_lasso(
        system,
        counts,
        l1_penalty=0.01,
        max_iterations=1000,
        tolerance=1e-9,
    )
    metrics = observed_site_metrics(
        system.candidate_positions,
        scores,
        observed,
    )
    assert diagnostics["window_fit_mae"] < 0.01
    assert metrics["average_precision"] == 1.0
    assert metrics["recall_at_observed_k"] == 1.0


def test_average_precision_handles_unknown_positive_count():
    assert average_precision([0.2, 0.1], [False, False]) is None
    assert average_precision([0.9, 0.2, 0.8], [True, False, True]) == 1.0
