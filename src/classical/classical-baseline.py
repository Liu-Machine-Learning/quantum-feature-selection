"""Run classical feature-selection benchmarks on the S1 dataset.

The benchmark implements three complementary methods:

* Pearson correlation ranks every feature by absolute correlation with the
  binary ASD label.
* SVM-RFE ranks every feature through recursive elimination with a linear SVM.
* Exhaustive search evaluates every five-feature subset with a linear SVM and
  reports the ten subsets with the highest cross-validated accuracy.

The exhaustive benchmark fits median imputation and standardization inside
each cross-validation fold to prevent information leakage. Results are printed
and saved as CSV files for reproducible downstream analysis.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LOGGER = logging.getLogger("classical_feature_selection")

SCRIPT_DIRECTORY: Final = Path(__file__).resolve().parent
DEFAULT_DATA_PATH: Final = ""
DEFAULT_OUTPUT_DIRECTORY: ""
SUBSET_SIZE: Final = 5
TOP_SUBSET_COUNT: Final = 10

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
ScoreFunction = Callable[[IntArray, IntArray], float]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Defines one deterministic classical benchmark run."""

    data_path: Path
    output_directory: Path
    cross_validation_folds: int = 5
    random_seed: int = 42
    svm_c: float = 1.0
    scoring: str = "accuracy"
    progress_interval: int = 1_000

    def validate(self) -> None:
        """Raise ``ValueError`` when a configuration value is invalid."""

        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least 2.")
        if self.svm_c <= 0.0:
            raise ValueError("svm_c must be greater than zero.")
        if self.scoring not in {"accuracy", "balanced_accuracy"}:
            raise ValueError("Unsupported scoring method: " + self.scoring)
        if self.progress_interval < 0:
            raise ValueError("progress_interval cannot be negative.")


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Stores cleaned predictors, labels, and ordered feature names."""

    predictors: FloatArray
    labels: IntArray
    feature_names: tuple[str, ...]

    @property
    def subject_count(self) -> int:
        """Return the number of subjects."""

        return int(self.predictors.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of candidate features."""

        return int(self.predictors.shape[1])


@dataclass(frozen=True, slots=True)
class SubsetPerformance:
    """Stores the cross-validation result for one feature subset."""

    indices: tuple[int, ...]
    mean_score: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class BenchmarkResults:
    """Contains the output tables from all benchmark methods."""

    pearson_ranking: pd.DataFrame
    svm_rfe_ranking: pd.DataFrame
    exhaustive_top_subsets: pd.DataFrame
    exhaustive_runtime_seconds: float


def configure_logging(level: str) -> None:
    """Configure timestamped console logging."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Pearson ranking, SVM-RFE, and exhaustive five-feature "
            "linear-SVM selection on the S1 dataset."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to S1Dataset.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for the three result CSV files.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used to shuffle cross-validation folds.",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="Regularization parameter C for all linear SVM models.",
    )
    parser.add_argument(
        "--scoring",
        choices=("accuracy", "balanced_accuracy"),
        default="accuracy",
        help="Cross-validation metric used to rank exhaustive-search subsets.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1_000,
        help="Log progress after this many subsets; use 0 to disable.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level.",
    )
    return parser.parse_args(arguments)


def config_from_arguments(arguments: argparse.Namespace) -> BenchmarkConfig:
    """Create and validate benchmark configuration from parsed arguments."""

    config = BenchmarkConfig(
        data_path=arguments.data_path.expanduser().resolve(),
        output_directory=arguments.output_directory.expanduser().resolve(),
        cross_validation_folds=arguments.cv_folds,
        random_seed=arguments.random_seed,
        svm_c=arguments.svm_c,
        scoring=arguments.scoring,
        progress_interval=arguments.progress_interval,
    )
    config.validate()
    return config


def load_s1_dataset(data_path: Path) -> PreparedDataset:
    """Load S1 and reproduce the QAOA notebooks' preprocessing.

    The first CSV data row contains measurement units rather than a subject.
    SIB observations are excluded to retain the ASD-versus-NEU task. Numeric
    conversion occurs here; each benchmark handles imputation in the scope
    appropriate to its statistical purpose.

    Args:
        data_path: Location of ``S1Dataset.csv``.

    Returns:
        Clean predictors, binary labels, and ordered feature names.

    Raises:
        FileNotFoundError: The input file does not exist.
        ValueError: The dataset schema or values cannot support the benchmark.
    """

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path, header=0)
    required_columns = {"Group", "Vineland ABC"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if data.empty:
        raise ValueError("Dataset contains no rows.")

    data = data.iloc[1:].copy()
    data = data[data["Group"] != "SIB"].copy()
    if data["Group"].isna().any():
        raise ValueError("At least one subject has no diagnostic group.")

    observed_groups = set(data["Group"].astype(str).unique())
    if observed_groups != {"ASD", "NEU"}:
        raise ValueError(
            "Expected only ASD and NEU groups; found: "
            + ", ".join(sorted(observed_groups))
        )

    labels = (data["Group"] == "ASD").to_numpy(dtype=np.int_)
    predictor_frame = data.drop(columns=["Group", "Vineland ABC"])
    predictor_frame = predictor_frame.apply(pd.to_numeric, errors="coerce")
    entirely_missing_columns = predictor_frame.columns[
        predictor_frame.isna().all()
    ].tolist()
    if entirely_missing_columns:
        raise ValueError(
            "Predictors contain no numeric values: "
            + ", ".join(entirely_missing_columns)
        )

    predictors = predictor_frame.to_numpy(dtype=np.float64)
    if np.isinf(predictors).any():
        raise ValueError("Predictors contain infinite values.")
    if predictors.shape[1] < SUBSET_SIZE:
        raise ValueError(
            f"At least {SUBSET_SIZE} features are required for exhaustive search."
        )

    return PreparedDataset(
        predictors=predictors,
        labels=labels,
        feature_names=tuple(str(name) for name in predictor_frame.columns),
    )


def rank_features_by_pearson(dataset: PreparedDataset) -> pd.DataFrame:
    """Rank every feature by absolute Pearson correlation with the label.

    Constant features receive correlation zero because their Pearson
    correlation is mathematically undefined and they contain no ranking signal.
    """

    imputed_predictors = SimpleImputer(strategy="median").fit_transform(
        dataset.predictors
    )
    centered_predictors = imputed_predictors - np.mean(
        imputed_predictors,
        axis=0,
    )
    centered_labels = dataset.labels - np.mean(dataset.labels)
    numerator = centered_predictors.T @ centered_labels
    denominator = np.sqrt(
        np.sum(np.square(centered_predictors), axis=0)
        * np.sum(np.square(centered_labels))
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros(dataset.feature_count, dtype=np.float64),
        where=denominator > 0.0,
    )

    ranking = pd.DataFrame(
        {
            "Original_Index": np.arange(dataset.feature_count, dtype=int),
            "Feature": dataset.feature_names,
            "Pearson_Correlation": correlations,
            "Absolute_Correlation": np.abs(correlations),
        }
    )
    ranking = ranking.sort_values(
        by=["Absolute_Correlation", "Original_Index"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, dataset.feature_count + 1))
    return ranking


def rank_features_by_svm_rfe(
    dataset: PreparedDataset,
    svm_c: float,
) -> pd.DataFrame:
    """Rank every standardized feature with linear-SVM recursive elimination.

    SVM-RFE is a descriptive ranking fitted on the complete dataset. It does
    not report generalization performance; exhaustive-search performance is
    evaluated separately through leakage-safe cross-validation.
    """

    imputed_predictors = SimpleImputer(strategy="median").fit_transform(
        dataset.predictors
    )
    standardized_predictors = StandardScaler().fit_transform(imputed_predictors)
    selector = RFE(
        estimator=SVC(kernel="linear", C=svm_c),
        n_features_to_select=1,
        step=1,
    )
    selector.fit(standardized_predictors, dataset.labels)

    ranking = pd.DataFrame(
        {
            "Original_Index": np.arange(dataset.feature_count, dtype=int),
            "Feature": dataset.feature_names,
            "SVM_RFE_Rank": selector.ranking_.astype(int),
        }
    )
    return ranking.sort_values(
        by=["SVM_RFE_Rank", "Original_Index"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)


def _create_linear_svm_pipeline(svm_c: float) -> Pipeline:
    """Create a fold-local standardization and linear-SVM pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", C=svm_c)),
        ]
    )


def _resolve_score_function(scoring: str) -> ScoreFunction:
    """Return the configured classification score function."""

    if scoring == "accuracy":
        return accuracy_score
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score
    raise ValueError("Unsupported scoring method: " + scoring)


def evaluate_subset(
    dataset: PreparedDataset,
    indices: tuple[int, ...],
    splits: Sequence[tuple[IntArray, IntArray]],
    svm_c: float,
    score_function: ScoreFunction,
) -> SubsetPerformance:
    """Evaluate one feature subset across fixed cross-validation folds."""

    fold_scores: list[float] = []
    for training_indices, validation_indices in splits:
        model = _create_linear_svm_pipeline(svm_c)
        model.fit(
            dataset.predictors[training_indices][:, indices],
            dataset.labels[training_indices],
        )
        predictions = model.predict(
            dataset.predictors[validation_indices][:, indices]
        )
        fold_scores.append(
            score_function(dataset.labels[validation_indices], predictions)
        )

    return SubsetPerformance(
        indices=indices,
        mean_score=float(np.mean(fold_scores)),
        standard_deviation=float(np.std(fold_scores, ddof=0)),
    )


def exhaustive_five_feature_search(
    dataset: PreparedDataset,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, float]:
    """Evaluate all five-feature subsets and return the ten best performers."""

    class_counts = np.bincount(dataset.labels)
    smallest_class_count = int(np.min(class_counts[class_counts > 0]))
    if config.cross_validation_folds > smallest_class_count:
        raise ValueError(
            "cv-folds cannot exceed the number of subjects in the smallest "
            f"class ({smallest_class_count})."
        )

    cross_validator = StratifiedKFold(
        n_splits=config.cross_validation_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    splits = [
        (
            np.asarray(training, dtype=np.int_),
            np.asarray(validation, dtype=np.int_),
        )
        for training, validation in cross_validator.split(
            dataset.predictors,
            dataset.labels,
        )
    ]
    score_function = _resolve_score_function(config.scoring)
    total_subsets = math.comb(dataset.feature_count, SUBSET_SIZE)
    LOGGER.info(
        "Exhaustive search | subsets=%s | subset size=%d | folds=%d | metric=%s",
        f"{total_subsets:,}",
        SUBSET_SIZE,
        config.cross_validation_folds,
        config.scoring,
    )

    start_time = time.perf_counter()
    performances: list[SubsetPerformance] = []
    for completed_count, indices in enumerate(
        combinations(range(dataset.feature_count), SUBSET_SIZE),
        start=1,
    ):
        performances.append(
            evaluate_subset(
                dataset,
                indices,
                splits,
                config.svm_c,
                score_function,
            )
        )
        if (
            config.progress_interval > 0
            and completed_count % config.progress_interval == 0
        ):
            LOGGER.info(
                "Exhaustive-search progress: %s/%s subsets (%.1f%%)",
                f"{completed_count:,}",
                f"{total_subsets:,}",
                100.0 * completed_count / total_subsets,
            )

    runtime_seconds = time.perf_counter() - start_time
    performances.sort(
        key=lambda result: (
            -result.mean_score,
            result.standard_deviation,
            result.indices,
        )
    )

    rows: list[dict[str, object]] = []
    for rank, performance in enumerate(
        performances[:TOP_SUBSET_COUNT],
        start=1,
    ):
        rows.append(
            {
                "Rank": rank,
                "Feature_Indices": ", ".join(
                    str(index) for index in performance.indices
                ),
                "Feature_Names": " | ".join(
                    dataset.feature_names[index] for index in performance.indices
                ),
                "Mean_CV_Score": performance.mean_score,
                "CV_Score_Standard_Deviation": performance.standard_deviation,
                "Scoring": config.scoring,
                "CV_Folds": config.cross_validation_folds,
            }
        )

    return pd.DataFrame(rows), runtime_seconds


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run all three feature-selection benchmarks."""

    dataset = load_s1_dataset(config.data_path)
    LOGGER.info(
        "Prepared S1 dataset | subjects=%d | features=%d | ASD=%d | NEU=%d",
        dataset.subject_count,
        dataset.feature_count,
        int(np.sum(dataset.labels == 1)),
        int(np.sum(dataset.labels == 0)),
    )

    pearson_ranking = rank_features_by_pearson(dataset)
    svm_rfe_ranking = rank_features_by_svm_rfe(dataset, config.svm_c)
    exhaustive_top_subsets, runtime_seconds = exhaustive_five_feature_search(
        dataset,
        config,
    )
    return BenchmarkResults(
        pearson_ranking=pearson_ranking,
        svm_rfe_ranking=svm_rfe_ranking,
        exhaustive_top_subsets=exhaustive_top_subsets,
        exhaustive_runtime_seconds=runtime_seconds,
    )


def write_results(results: BenchmarkResults, output_directory: Path) -> None:
    """Write all benchmark tables as CSV files."""

    output_directory.mkdir(parents=True, exist_ok=True)
    results.pearson_ranking.to_csv(
        output_directory / "pearson_feature_ranking.csv",
        index=False,
    )
    results.svm_rfe_ranking.to_csv(
        output_directory / "svm_rfe_feature_ranking.csv",
        index=False,
    )
    results.exhaustive_top_subsets.to_csv(
        output_directory / "exhaustive_top_10_five_feature_subsets.csv",
        index=False,
    )
    LOGGER.info("Saved benchmark results to %s", output_directory)


def print_results(results: BenchmarkResults) -> None:
    """Print complete rankings and exhaustive-search winners."""

    print("\nPearson correlation ranking — all features\n")
    print(results.pearson_ranking.to_string(index=False))

    print("\nSVM-RFE ranking — all features\n")
    print(results.svm_rfe_ranking.to_string(index=False))

    print("\nExhaustive linear-SVM search — top 10 five-feature subsets\n")
    print(results.exhaustive_top_subsets.to_string(index=False))
    print(
        "\nExhaustive-search runtime: "
        f"{results.exhaustive_runtime_seconds:.3f} seconds"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line benchmark and return its process exit code."""

    parsed = parse_arguments(arguments)
    configure_logging(parsed.log_level)
    try:
        config = config_from_arguments(parsed)
        results = run_benchmark(config)
        print_results(results)
        write_results(results, config.output_directory)
    except KeyboardInterrupt:
        LOGGER.error("Benchmark interrupted by the user.")
        return 130
    except Exception:
        LOGGER.exception("Classical feature-selection benchmark failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
