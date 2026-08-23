# Inspired by https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/metrics/perplexity.py
# But we compute the perplexity correctly: exp(average(nll)), not average(exp(nll))
# Also adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/perplexity.py
# But we pass in the loss to avoid recomputation

from functools import partial
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torchmetrics import Metric

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

try:
    from apex.transformer import parallel_state
except ImportError:
    parallel_state = None


class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.
    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self, ignore_index: int = -100, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.
        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        count = (target != self.ignore_index).sum()
        if count == 0:
            return
        if loss is None:
            loss = self.loss_fn(preds, target)
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.
        Returns:
           Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)

class NumTokens(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    count: Tensor

    def __init__(self, ignore_index: int = -100, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.add_state(
            "count",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
            persistent=True,
        )
        if parallel_state is not None and not parallel_state.is_unitialized():
            self.tensor_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            self.tensor_parallel_world_size = 1

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        # ✅ 只统计有效 label（参与 loss 的 token）
        valid = (target != self.ignore_index)
        self.count += valid.sum().to(torch.int64) // self.tensor_parallel_world_size


    def compute(self) -> Tensor:
        return self.count

    # Adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/metric.py
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.
        This can be done when the global metric state is a sinple reduction of batch states.
        """
        self.update(*args, **kwargs)
        return self.compute()


class M6AHistogramMetric(Metric):
    """Distributed, bounded-memory ranking metric for millions of A sites."""

    is_differentiable = False
    full_state_update = False

    def __init__(self, metric: str, bins: int = 2048, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        if metric not in {"average_precision", "auroc"}:
            raise ValueError(f"Unsupported histogram metric: {metric}")
        self.metric = metric
        self.bins = int(bins)
        self.add_state(
            "positive_hist",
            default=torch.zeros(self.bins, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "negative_hist",
            default=torch.zeros(self.bins, dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:
        probabilities = torch.sigmoid(preds.reshape(-1).detach().float())
        labels = target.reshape(-1).to(torch.bool)
        indices = torch.clamp((probabilities * self.bins).to(torch.long), max=self.bins - 1)
        if labels.any():
            self.positive_hist += torch.bincount(
                indices[labels], minlength=self.bins
            ).to(self.positive_hist.dtype)
        negatives = ~labels
        if negatives.any():
            self.negative_hist += torch.bincount(
                indices[negatives], minlength=self.bins
            ).to(self.negative_hist.dtype)

    def compute(self) -> Tensor:
        positives = torch.flip(self.positive_hist, dims=[0])
        negatives = torch.flip(self.negative_hist, dims=[0])
        total_positive = positives.sum()
        total_negative = negatives.sum()
        if total_positive == 0 or total_negative == 0:
            return torch.tensor(float("nan"), device=positives.device, dtype=torch.float64)

        true_positive = torch.cumsum(positives, dim=0)
        false_positive = torch.cumsum(negatives, dim=0)
        if self.metric == "average_precision":
            precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
            return (precision * positives).sum() / total_positive

        true_positive_rate = true_positive / total_positive
        false_positive_rate = false_positive / total_negative
        zero = torch.zeros(1, device=positives.device, dtype=torch.float64)
        return torch.trapezoid(
            torch.cat([zero, true_positive_rate]),
            torch.cat([zero, false_positive_rate]),
        )


class M6AConfusionMetric(Metric):
    """Thresholded binary metric accumulated across all candidate A sites."""

    is_differentiable = False
    full_state_update = False

    def __init__(self, metric: str, threshold: float = 0.5, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        if metric not in {"precision", "recall", "f1", "accuracy", "positive_rate"}:
            raise ValueError(f"Unsupported confusion metric: {metric}")
        self.metric = metric
        self.threshold = float(threshold)
        for name in ("tp", "fp", "tn", "fn"):
            self.add_state(name, default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:
        predicted = torch.sigmoid(preds.reshape(-1).detach()) >= self.threshold
        labels = target.reshape(-1).to(torch.bool)
        self.tp += (predicted & labels).sum()
        self.fp += (predicted & ~labels).sum()
        self.tn += (~predicted & ~labels).sum()
        self.fn += (~predicted & labels).sum()

    def compute(self) -> Tensor:
        tp, fp, tn, fn = (value.to(torch.float64) for value in (self.tp, self.fp, self.tn, self.fn))
        if self.metric == "precision":
            return tp / (tp + fp).clamp_min(1.0)
        if self.metric == "recall":
            return tp / (tp + fn).clamp_min(1.0)
        if self.metric == "f1":
            return 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
        if self.metric == "positive_rate":
            return (tp + fn) / (tp + fp + tn + fn).clamp_min(1.0)
        return (tp + tn) / (tp + fp + tn + fn).clamp_min(1.0)

torchmetric_fns = {
    "perplexity": Perplexity,
    "num_tokens": NumTokens,
    "m6a_average_precision": partial(M6AHistogramMetric, metric="average_precision"),
    "m6a_auroc": partial(M6AHistogramMetric, metric="auroc"),
    "m6a_precision": partial(M6AConfusionMetric, metric="precision"),
    "m6a_recall": partial(M6AConfusionMetric, metric="recall"),
    "m6a_f1": partial(M6AConfusionMetric, metric="f1"),
    "m6a_accuracy": partial(M6AConfusionMetric, metric="accuracy"),
    "m6a_positive_rate": partial(M6AConfusionMetric, metric="positive_rate"),
}
