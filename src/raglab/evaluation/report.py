"""Evaluation reporting and visualization module.

Generates comparative Markdown tables, ablation analysis, and Plotly charts
from benchmark run results stored in eval/results/ to visualize retrieval recall,
faithfulness gains, and latency tradeoffs across pipeline configurations.
"""

from pathlib import Path
from typing import List
import pandas as pd


class EvaluationReporter:
    """Aggregates and formats benchmark results into reports and visual figures."""

    def __init__(self, results_dir: Path) -> None:
        """Initialize reporter with results directory path.

        Args:
            results_dir: Path where run JSON files reside.
        """
        pass

    def generate_summary_table(self, run_names: List[str]) -> pd.DataFrame:
        """Create comparative summary DataFrame across specified evaluation runs.

        Args:
            run_names: List of run names or filenames to compare.

        Returns:
            pd.DataFrame: Comparative metrics table.
        """
        raise NotImplementedError

    def export_markdown_report(self, output_path: Path) -> None:
        """Generate and save Markdown comparison report.

        Args:
            output_path: Path to write the report markdown.
        """
        raise NotImplementedError
