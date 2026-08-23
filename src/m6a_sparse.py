"""Sparse recovery utilities for window-level m6A count predictions.

The measurement model is ``y = W p``.  Each row of ``W`` is one sequence
window, each column is an adenosine candidate, and ``p`` is a non-negative
site score.  The implementation uses interval sums rather than materializing
the usually large binary matrix ``W``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def sliding_window_starts(
    sequence_length: int,
    window_length: int,
    stride: int,
    include_tail: bool = True,
) -> np.ndarray:
    """Return deterministic window starts and optionally cover the final base."""

    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if sequence_length == 0:
        return np.empty(0, dtype=np.int64)
    if sequence_length <= window_length:
        return np.asarray([0], dtype=np.int64)

    starts = np.arange(
        0,
        sequence_length - window_length + 1,
        stride,
        dtype=np.int64,
    )
    tail_start = sequence_length - window_length
    if include_tail and starts[-1] != tail_start:
        starts = np.concatenate((starts, np.asarray([tail_start], dtype=np.int64)))
    return starts


@dataclass(frozen=True)
class WindowCandidateSystem:
    """Implicit binary window-by-candidate measurement matrix."""

    candidate_positions: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        window_length: int,
        stride: int,
        candidate_base: str = "A",
    ) -> "WindowCandidateSystem":
        sequence = str(sequence).upper()
        candidates = np.fromiter(
            (index for index, base in enumerate(sequence) if base == candidate_base),
            dtype=np.int64,
        )
        starts = sliding_window_starts(len(sequence), window_length, stride)
        ends = np.minimum(starts + int(window_length), len(sequence))
        left = np.searchsorted(candidates, starts, side="left")
        right = np.searchsorted(candidates, ends, side="left")
        return cls(candidates, starts, ends, left, right)

    @property
    def n_candidates(self) -> int:
        return int(self.candidate_positions.size)

    @property
    def n_windows(self) -> int:
        return int(self.starts.size)

    def forward(self, site_scores: np.ndarray) -> np.ndarray:
        """Compute window sums ``W p`` in linear time."""

        values = np.asarray(site_scores, dtype=np.float64)
        if values.shape != (self.n_candidates,):
            raise ValueError(
                f"site_scores must have shape {(self.n_candidates,)}, got {values.shape}"
            )
        prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        return prefix[self.right_indices] - prefix[self.left_indices]

    def adjoint(self, window_values: np.ndarray) -> np.ndarray:
        """Compute ``W.T r`` using range additions in linear time."""

        values = np.asarray(window_values, dtype=np.float64)
        if values.shape != (self.n_windows,):
            raise ValueError(
                f"window_values must have shape {(self.n_windows,)}, got {values.shape}"
            )
        difference = np.zeros(self.n_candidates + 1, dtype=np.float64)
        np.add.at(difference, self.left_indices, values)
        np.add.at(difference, self.right_indices, -values)
        return np.cumsum(difference[:-1])

    def counts_for_positions(self, positions: Sequence[int]) -> np.ndarray:
        """Return exact window counts for a collection of observed sites."""

        positions = np.asarray(sorted(set(int(value) for value in positions)), dtype=np.int64)
        left = np.searchsorted(positions, self.starts, side="left")
        right = np.searchsorted(positions, self.ends, side="left")
        return (right - left).astype(np.float64)


def _largest_gram_eigenvalue(
    system: WindowCandidateSystem,
    iterations: int = 30,
) -> float:
    """Estimate the Lipschitz constant of ``W.T W`` by power iteration."""

    if system.n_candidates == 0 or system.n_windows == 0:
        return 1.0
    vector = np.full(
        system.n_candidates,
        1.0 / np.sqrt(system.n_candidates),
        dtype=np.float64,
    )
    for _ in range(max(1, iterations)):
        transformed = system.adjoint(system.forward(vector))
        norm = float(np.linalg.norm(transformed))
        if norm <= 1e-12:
            return 1.0
        vector = transformed / norm
    eigenvalue = float(np.dot(vector, system.adjoint(system.forward(vector))))
    return max(eigenvalue, 1e-12)


def solve_nonnegative_lasso(
    system: WindowCandidateSystem,
    window_counts: Sequence[float],
    l1_penalty: float = 0.1,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    max_score: Optional[float] = 1.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Recover sparse candidate scores with projected FISTA.

    The optimized objective is ``0.5 * ||W p - y||^2 + lambda * ||p||_1``
    with ``p >= 0`` and, by default, ``p <= 1``.
    """

    if l1_penalty < 0:
        raise ValueError("l1_penalty must be non-negative")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    if max_score is not None and max_score <= 0:
        raise ValueError("max_score must be positive or None")

    counts = np.clip(np.asarray(window_counts, dtype=np.float64), 0.0, None)
    if counts.shape != (system.n_windows,):
        raise ValueError(
            f"window_counts must have shape {(system.n_windows,)}, got {counts.shape}"
        )
    if system.n_candidates == 0:
        residual = -counts
        return np.empty(0, dtype=np.float64), {
            "iterations": 0.0,
            "objective": float(0.5 * np.dot(residual, residual)),
            "window_fit_mae": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        }

    lipschitz = _largest_gram_eigenvalue(system)
    scores = np.zeros(system.n_candidates, dtype=np.float64)
    accelerated = scores.copy()
    momentum = 1.0
    iterations_run = 0

    for iteration in range(1, max_iterations + 1):
        gradient = system.adjoint(system.forward(accelerated) - counts)
        updated = accelerated - gradient / lipschitz - l1_penalty / lipschitz
        updated = np.maximum(updated, 0.0)
        if max_score is not None:
            updated = np.minimum(updated, max_score)

        relative_change = float(
            np.linalg.norm(updated - scores) / max(1.0, np.linalg.norm(scores))
        )
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - scores)
        scores = updated
        momentum = next_momentum
        iterations_run = iteration
        if relative_change <= tolerance:
            break

    residual = system.forward(scores) - counts
    objective = 0.5 * float(np.dot(residual, residual)) + l1_penalty * float(scores.sum())
    diagnostics = {
        "iterations": float(iterations_run),
        "objective": objective,
        "window_fit_mae": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "lipschitz": float(lipschitz),
    }
    return scores, diagnostics


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Compute observed-site average precision without external dependencies."""

    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=bool)
    if scores_array.shape != labels_array.shape:
        raise ValueError("scores and labels must have the same shape")
    positives = int(labels_array.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores_array, kind="stable")
    ranked = labels_array[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked].sum() / positives)


def observed_site_metrics(
    candidate_positions: Sequence[int],
    scores: Sequence[float],
    observed_positions: Iterable[int],
    tolerance_nt: int = 0,
) -> Dict[str, Optional[float]]:
    """Evaluate ranking recovery of known sites among adenosine candidates.

    Unlabelled adenosines remain biologically unknown.  These metrics therefore
    measure recovery of *observed calls*, not verified methylation specificity.
    """

    candidates = np.asarray(candidate_positions, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if candidates.shape != scores_array.shape:
        raise ValueError("candidate_positions and scores must have the same shape")
    if tolerance_nt < 0:
        raise ValueError("tolerance_nt must be non-negative")

    observed = np.asarray(sorted(set(int(value) for value in observed_positions)), dtype=np.int64)
    observed_set = set(observed.tolist())
    labels = np.asarray([position in observed_set for position in candidates], dtype=bool)
    n_observed = int(observed.size)

    if n_observed == 0 or candidates.size == 0:
        return {
            "n_candidates": float(candidates.size),
            "n_observed": float(n_observed),
            "average_precision": average_precision(scores_array, labels),
            "recall_at_observed_k": None,
            "recall_at_observed_k_with_tolerance": None,
        }

    k = min(n_observed, int(candidates.size))
    # Break score ties deterministically by coordinate.
    order = np.lexsort((candidates, -scores_array))
    selected = candidates[order[:k]]
    exact_recovered = sum(int(position) in observed_set for position in selected)
    tolerant_recovered = sum(
        bool(np.any(np.abs(selected - int(position)) <= tolerance_nt))
        for position in observed
    )
    return {
        "n_candidates": float(candidates.size),
        "n_observed": float(n_observed),
        "average_precision": average_precision(scores_array, labels),
        "recall_at_observed_k": exact_recovered / n_observed,
        "recall_at_observed_k_with_tolerance": tolerant_recovered / n_observed,
    }
