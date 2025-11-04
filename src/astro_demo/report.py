"""Helpers to collate experiment outputs into a concise report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .calibration import CalibrationReport


@dataclass
class ExperimentRow:
    """Row entries for the final summary table."""

    variant: str
    accuracy: float
    ece: float
    brier: float
    nll: float


def build_summary_table(rows: List[ExperimentRow]) -> pd.DataFrame:
    """Create a tidy dataframe for easy export."""

    payload = [asdict(row) for row in rows]
    df = pd.DataFrame(payload)
    df.sort_values("ece", inplace=True)
    return df


def export_report(
    table: pd.DataFrame,
    calibration_fig_path: Path,
    output_markdown: Path,
) -> None:
    """Write a lightweight markdown report referencing the generated artifacts."""

    md_lines = [
        "# Calibration Snapshot",
        "",
        "## Summary Table",
        "",
        table.to_markdown(index=False),
        "",
        "## Reliability Diagram",
        "",
        f"![reliability diagram]({calibration_fig_path.as_posix()})",
        "",
    ]
    output_markdown.write_text("\n".join(md_lines))


def to_experiment_row(
    variant: str,
    accuracy: float,
    calibration: CalibrationReport,
) -> ExperimentRow:
    """Create a row from raw metrics."""

    return ExperimentRow(
        variant=variant,
        accuracy=accuracy,
        ece=calibration.ece,
        brier=calibration.brier,
        nll=calibration.nll,
    )
