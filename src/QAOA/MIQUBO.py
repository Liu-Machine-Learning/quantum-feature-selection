"""Run mutual-information QUBO feature selection on IBM Runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score

from qaoa_runtime import (
    PreparedDataset,
    QuboModel,
    add_runtime_arguments,
    assemble_symmetric_qubo,
    configure_logging,
    execute_runtime_experiment,
    load_s1_dataset,
    runtime_config_from_arguments,
)


LOGGER = logging.getLogger("mutual_information_qubo")
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / ""


def quantile_bin(values: NDArray[np.float64], bin_count: int) -> NDArray[np.int_]:
    """Discretize one feature with the notebook's unique-quantile rule."""

    numeric_values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(
        numeric_values,
        np.linspace(0.0, 1.0, bin_count + 1),
    )
    unique_edges = np.unique(edges)
    if unique_edges.size <= 2:
        return np.zeros(numeric_values.size, dtype=np.int_)
    return np.digitize(
        numeric_values,
        unique_edges[1:-1],
        right=True,
    ).astype(np.int_)


def build_mutual_information_qubo(
    dataset: PreparedDataset,
    alpha: float,
    bin_count: int,
) -> QuboModel:
    """Build label-importance and feature-redundancy MI terms."""

    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")

    binned_predictors = np.column_stack(
        [
            quantile_bin(dataset.predictors[:, index], bin_count)
            for index in range(dataset.feature_count)
        ]
    )
    importance = np.asarray(
        [
            mutual_info_score(binned_predictors[:, index], dataset.labels)
            for index in range(dataset.feature_count)
        ],
        dtype=np.float64,
    )

    redundancy = np.zeros(
        (dataset.feature_count, dataset.feature_count),
        dtype=np.float64,
    )
    for index in range(dataset.feature_count):
        for other_index in range(index + 1, dataset.feature_count):
            score = mutual_info_score(
                binned_predictors[:, index],
                binned_predictors[:, other_index],
            )
            redundancy[index, other_index] = score
            redundancy[other_index, index] = score

    return assemble_symmetric_qubo(importance, redundancy, alpha)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the mutual-information experiment arguments."""

    parser = argparse.ArgumentParser(
        description="Run mutual-information QUBO feature selection on IBM Runtime."
    )
    add_runtime_arguments(
        parser,
        default_data_path=DEFAULT_DATA_PATH,
        default_alpha=0.974,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of requested quantile bins per feature.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the mutual-information IBM Runtime workflow."""

    parsed = parse_arguments(arguments)
    configure_logging(parsed.log_level)
    try:
        config = runtime_config_from_arguments(parsed)
        dataset = load_s1_dataset(config.data_path)
        qubo_model = build_mutual_information_qubo(
            dataset,
            config.alpha,
            parsed.bins,
        )
        execute_runtime_experiment(
            dataset,
            qubo_model,
            config,
            importance_label="MI(label)",
        )
    except KeyboardInterrupt:
        LOGGER.error("Execution interrupted by the user.")
        return 130
    except Exception:
        LOGGER.exception("Mutual-information QUBO execution failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
