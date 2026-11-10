"""Evaluation runner module for executing benchmark suites.

Runs gold evaluation datasets across configured pipeline variants, calculates
per-question retrieval and generation metrics, aggregates results, and saves
detailed telemetry to eval/results/<run_name>.json.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List
from pydantic import BaseModel
from raglab.evaluation.dataset import GoldQuestion

if TYPE_CHECKING:
    from raglab.config import AppConfig


class BenchmarkRunResult(BaseModel):
    """Encapsulates full results and metadata of an evaluation run.

    Attributes:
        run_name: Name of the experiment variant or run identifier.
        config: Pipeline configuration used for the run.
        metrics_summary: Aggregated metrics (mean recall, MRR, faithfulness, etc.).
        per_question_results: Granular per-question outputs, scores, and timings.
    """
    run_name: str
    config: Dict[str, Any]
    metrics_summary: Dict[str, float]
    per_question_results: List[Dict[str, Any]]


class EvaluationRunner:
    """Executes evaluation benchmarks over gold datasets."""

    def __init__(self, config_path: Path, output_dir: Path) -> None:
        """Initialize runner with configuration file path and destination results directory.

        Args:
            config_path: Path to experiment YAML config.
            output_dir: Path to directory where results JSON files will be written.
        """
        pass

    def run(self, dataset: List[GoldQuestion]) -> BenchmarkRunResult:
        """Execute benchmark run over questions and compute all metrics.

        Args:
            dataset: List of GoldQuestion items to evaluate.

        Returns:
            BenchmarkRunResult: Complete run results with aggregated and itemized metrics.
        """
        raise NotImplementedError
