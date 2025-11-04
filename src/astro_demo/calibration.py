"""Calibration utilities for classification and regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt


@dataclass
class CalibrationReport:
    """Summary container for calibration metrics."""

    ece: float
    mce: float
    brier: float
    nll: float


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
) -> Tuple[float, float]:
    """Return ECE and MCE in percentage points."""

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    mce = torch.zeros(1, device=probs.device)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = confidences.gt(lower) & confidences.le(upper)
        if mask.sum() == 0:
            continue
        accuracy = accuracies[mask].float().mean()
        confidence = confidences[mask].mean()
        gap = (confidence - accuracy).abs()
        ece += mask.float().mean() * gap
        mce = torch.maximum(mce, gap)

    return ece.item() * 100.0, mce.item() * 100.0


def nll_and_brier(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[float, float]:
    """Compute negative log-likelihood and Brier score."""

    eps = 1e-8
    label_one_hot = torch.zeros_like(probs)
    label_one_hot[torch.arange(probs.shape[0]), labels] = 1.0
    brier = torch.mean(torch.sum((probs - label_one_hot) ** 2, dim=1))
    nll = -torch.mean(torch.log(torch.gather(probs, 1, labels[:, None]) + eps))
    return nll.item(), brier.item()


def make_reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    *,
    n_bins: int = 15,
) -> plt.Axes:
    """Plot a reliability diagram with confidence histogram."""

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Per-sample max confidence and correctness
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()

    # Bins and centers
    bins = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    acc = torch.empty(n_bins, device=probs.device)
    hist = torch.empty(n_bins, device=probs.device)

    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # [lower, upper) for all bins except the last, which is [lower, upper]
        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = in_bin.sum()
        hist[i] = count

        if count > 0:
            acc[i] = correct[in_bin].mean()
        else:
            # No data -> no bar instead of "0 accuracy"
            acc[i] = float("nan")

    # Main reliability bars
    ax.bar(
        bin_centers.cpu().numpy(),
        acc.cpu().numpy(),
        width=1.0 / n_bins,
        alpha=0.7,
        edgecolor="black",
        label="Empirical acc",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    # Inset histogram of confidences
    ax_hist = ax.inset_axes([0.55, 0.05, 0.4, 0.4])
    ax_hist.bar(
        bin_centers.cpu().numpy(),
        hist.cpu().numpy(),
        width=1.0 / n_bins,
        alpha=0.6,
        color="gray",
    )
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("# samples")
    ax_hist.set_xlim(0, 1)

    return ax



def summarize_calibration(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 15,
) -> CalibrationReport:
    """Compute the key calibration metrics."""

    ece, mce = expected_calibration_error(probs, labels, n_bins=n_bins)
    nll, brier = nll_and_brier(probs, labels)
    return CalibrationReport(ece=ece, mce=mce, brier=brier, nll=nll)


def predictive_interval_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Check nominal vs. empirical coverage for regression intervals."""

    contains = (targets >= lower) & (targets <= upper)
    return float(np.mean(contains))
