"""Evaluate a five-feature subset with five classifiers and LOOCV.

The evaluator supports the S1, plasma, and shotgun datasets used throughout
this study. A feature subset can be supplied as five zero-based indices
or five feature names. Each subset is evaluated with these classifiers:

* Linear discriminant analysis (LDA)
* Linear support vector machine (SVM)
* Logistic regression
* Random forest
* Partial least squares discriminant analysis (PLS-DA)

Imputation and standardization are fitted independently inside every
leave-one-out cross-validation fold to prevent test-subject leakage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LOGGER = logging.getLogger("feature_subset_evaluation")

SCRIPT_DIRECTORY: Final = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY: Final = SCRIPT_DIRECTORY / "evaluation_results"
FEATURE_SUBSET_SIZE: Final = 5
SUPPORTED_DATASETS: Final = ("s1", "plasma", "shotgun")
DEFAULT_DATA_PATHS: Final[Mapping[str, Path]] = {
    "s1": SCRIPT_DIRECTORY / "dataset" / "S1Dataset.csv",
    "plasma": SCRIPT_DIRECTORY / "dataset" / "plasma_prescreen_56.csv",
    "shotgun": SCRIPT_DIRECTORY / "dataset" / "shotgun_prescreen_54.csv",
}

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Defines one five-feature LOOCV evaluation."""

    dataset_name: str
    data_path: Path
    output_directory: Path
    feature_indices: tuple[int, ...] | None
    feature_names: tuple[str, ...] | None
    random_seed: int = 42
    svm_c: float = 1.0
    logistic_c: float = 1.0
    random_forest_estimators: int = 500
    pls_components: int = 5

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is invalid."""

        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        if (self.feature_indices is None) == (self.feature_names is None):
            raise ValueError(
                "Provide either feature_indices or feature_names, but not both."
            )

        selected_features = self.feature_indices or self.feature_names or ()
        if len(selected_features) != FEATURE_SUBSET_SIZE:
            raise ValueError(
                f"Exactly {FEATURE_SUBSET_SIZE} features must be selected."
            )
        if len(set(selected_features)) != FEATURE_SUBSET_SIZE:
            raise ValueError("Selected features must be unique.")
        if self.svm_c <= 0.0 or self.logistic_c <= 0.0:
            raise ValueError("SVM and logistic-regression C must be positive.")
        if self.random_forest_estimators < 1:
            raise ValueError("random_forest_estimators must be at least 1.")
        if not 1 <= self.pls_components <= FEATURE_SUBSET_SIZE:
            raise ValueError(
                f"pls_components must be between 1 and {FEATURE_SUBSET_SIZE}."
            )


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Stores one cleaned binary-classification dataset."""

    predictors: FloatArray
    labels: IntArray
    feature_names: tuple[str, ...]

    @property
    def subject_count(self) -> int:
        """Return the number of subjects."""

        return int(self.predictors.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of available features."""

        return int(self.predictors.shape[1])


@dataclass(frozen=True, slots=True)
class SelectedSubset:
    """Stores the resolved feature indices, names, and predictor matrix."""

    indices: tuple[int, ...]
    names: tuple[str, ...]
    predictors: FloatArray


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """Stores aggregate metrics and subject-level LOOCV predictions."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Adapt PLS regression to binary classification with a 0.5 threshold."""

    def __init__(self, n_components: int = 2) -> None:
        """Initialize the classifier with the requested component count."""

        self.n_components = n_components

    def fit(self, predictors: FloatArray, labels: IntArray) -> PLSDAClassifier:
        """Fit a PLS regression model to binary labels."""

        self.classes_ = np.unique(labels)
        if self.classes_.size != 2:
            raise ValueError("PLS-DA requires exactly two classes.")
        maximum_components = min(predictors.shape[0] - 1, predictors.shape[1])
        if self.n_components > maximum_components:
            raise ValueError(
                "PLS component count exceeds the training-fold limit of "
                f"{maximum_components}."
            )

        self.model_ = PLSRegression(
            n_components=self.n_components,
            scale=False,
        )
        self.model_.fit(predictors, labels)
        return self

    def decision_function(self, predictors: FloatArray) -> FloatArray:
        """Return continuous PLS scores for the positive class."""

        return np.asarray(self.model_.predict(predictors), dtype=np.float64).ravel()

    def predict(self, predictors: FloatArray) -> IntArray:
        """Return binary predictions using the standard PLS-DA threshold."""

        return (self.decision_function(predictors) >= 0.5).astype(np.int_)


def configure_logging(level: str) -> None:
    """Configure timestamped console logging."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one five-feature subset with five classifiers using "
            "leave-one-out cross-validation."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset containing the selected features.",
    )
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--feature-indices",
        nargs=FEATURE_SUBSET_SIZE,
        type=int,
        metavar=("I1", "I2", "I3", "I4", "I5"),
        help="Five zero-based feature indices in dataset order.",
    )
    selection_group.add_argument(
        "--feature-names",
        nargs=FEATURE_SUBSET_SIZE,
        metavar=("F1", "F2", "F3", "F4", "F5"),
        help="Five feature names. Quote names containing spaces.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional dataset path overriding the repository default.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for metrics and subject-level prediction CSV files.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--random-forest-estimators", type=int, default=500)
    parser.add_argument("--pls-components", type=int, default=5)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(arguments)


def config_from_arguments(arguments: argparse.Namespace) -> EvaluationConfig:
    """Create and validate an evaluation configuration."""

    data_path = arguments.data_path or DEFAULT_DATA_PATHS[arguments.dataset]
    config = EvaluationConfig(
        dataset_name=arguments.dataset,
        data_path=data_path.expanduser().resolve(),
        output_directory=arguments.output_directory.expanduser().resolve(),
        feature_indices=(
            tuple(arguments.feature_indices) if arguments.feature_indices else None
        ),
        feature_names=(
            tuple(arguments.feature_names) if arguments.feature_names else None
        ),
        random_seed=arguments.random_seed,
        svm_c=arguments.svm_c,
        logistic_c=arguments.logistic_c,
        random_forest_estimators=arguments.random_forest_estimators,
        pls_components=arguments.pls_components,
    )
    config.validate()
    return config


def load_dataset(dataset_name: str, data_path: Path) -> PreparedDataset:
    """Load one of the repository's supported datasets."""

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if dataset_name == "s1":
        return _load_s1_dataset(data_path)
    if dataset_name in {"plasma", "shotgun"}:
        return _load_feature_by_subject_dataset(data_path)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_s1_dataset(data_path: Path) -> PreparedDataset:
    """Load the row-per-subject S1 dataset."""

    data = pd.read_csv(data_path, header=0)
    missing_columns = {"Group", "Vineland ABC"}.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "S1 is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # The first data row stores measurement units, not a subject.
    data = data.iloc[1:].copy()
    data = data[data["Group"] != "SIB"].copy()
    if data["Group"].isna().any():
        raise ValueError("At least one S1 subject has no diagnostic group.")
    observed_groups = set(data["Group"].astype(str).unique())
    if observed_groups != {"ASD", "NEU"}:
        raise ValueError(
            "Expected only ASD and NEU groups; found: "
            + ", ".join(sorted(observed_groups))
        )

    labels = (data["Group"] == "ASD").to_numpy(dtype=np.int_)
    predictor_frame = data.drop(columns=["Group", "Vineland ABC"])
    predictor_frame = predictor_frame.apply(pd.to_numeric, errors="coerce")
    _validate_numeric_feature_frame(predictor_frame)
    return PreparedDataset(
        predictors=predictor_frame.to_numpy(dtype=np.float64),
        labels=labels,
        feature_names=tuple(str(name) for name in predictor_frame.columns),
    )


def _load_feature_by_subject_dataset(data_path: Path) -> PreparedDataset:
    """Load a plasma or shotgun feature-by-subject dataset."""

    raw = pd.read_csv(data_path, header=None)
    if raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError("Feature-by-subject dataset is empty or malformed.")

    group_labels = raw.iloc[0, 1:].astype(str).to_numpy()
    feature_names = raw.iloc[1:, 0].astype(str).to_numpy()
    feature_by_subject = raw.iloc[1:, 1:].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Preserve the repository's plasma/shotgun indexing convention by removing
    # incomplete feature rows before filtering subjects.
    valid_feature_mask = ~feature_by_subject.isna().any(axis=1)
    feature_by_subject = feature_by_subject.loc[valid_feature_mask].copy()
    feature_names = feature_names[valid_feature_mask.to_numpy()]

    subject_mask = np.isin(group_labels, ["ASD", "TD", "NEU"])
    filtered_groups = group_labels[subject_mask]
    feature_by_subject = feature_by_subject.loc[:, subject_mask]
    labels = np.asarray(filtered_groups == "ASD", dtype=np.int_)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Dataset must contain ASD and TD/NEU subjects.")

    predictors = feature_by_subject.T.to_numpy(dtype=np.float64)
    if not np.isfinite(predictors).all():
        raise ValueError("Cleaned predictors contain non-finite values.")
    return PreparedDataset(
        predictors=predictors,
        labels=labels,
        feature_names=tuple(str(name) for name in feature_names),
    )


def _validate_numeric_feature_frame(feature_frame: pd.DataFrame) -> None:
    """Validate a numeric feature frame before fold-local imputation."""

    if feature_frame.shape[1] < FEATURE_SUBSET_SIZE:
        raise ValueError(
            f"Dataset must contain at least {FEATURE_SUBSET_SIZE} features."
        )
    entirely_missing = feature_frame.columns[feature_frame.isna().all()].tolist()
    if entirely_missing:
        raise ValueError(
            "Features contain no numeric values: " + ", ".join(entirely_missing)
        )
    if np.isinf(feature_frame.to_numpy(dtype=np.float64)).any():
        raise ValueError("Predictors contain infinite values.")


def resolve_feature_subset(
    dataset: PreparedDataset,
    config: EvaluationConfig,
) -> SelectedSubset:
    """Resolve requested feature indices or names against a dataset."""

    if config.feature_indices is not None:
        invalid_indices = [
            index
            for index in config.feature_indices
            if index < 0 or index >= dataset.feature_count
        ]
        if invalid_indices:
            raise ValueError(
                "Feature indices are outside the valid range "
                f"0..{dataset.feature_count - 1}: {invalid_indices}"
            )
        indices = config.feature_indices
    else:
        indices = _resolve_feature_names(
            config.feature_names or (),
            dataset.feature_names,
        )

    names = tuple(dataset.feature_names[index] for index in indices)
    return SelectedSubset(
        indices=indices,
        names=names,
        predictors=np.asarray(dataset.predictors[:, indices], dtype=np.float64),
    )


def _resolve_feature_names(
    requested_names: Sequence[str],
    available_names: Sequence[str],
) -> tuple[int, ...]:
    """Resolve exact or unambiguous case-insensitive feature names."""

    indices: list[int] = []
    for requested_name in requested_names:
        exact_matches = [
            index
            for index, available_name in enumerate(available_names)
            if available_name == requested_name
        ]
        matches = exact_matches or [
            index
            for index, available_name in enumerate(available_names)
            if available_name.casefold() == requested_name.casefold()
        ]
        if not matches:
            raise ValueError(f"Unknown feature name: {requested_name!r}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous feature name: {requested_name!r}")
        indices.append(matches[0])

    if len(set(indices)) != FEATURE_SUBSET_SIZE:
        raise ValueError("Resolved feature names must identify five unique columns.")
    return tuple(indices)


def build_classifiers(config: EvaluationConfig) -> Mapping[str, BaseEstimator]:
    """Create the five leakage-safe classification pipelines."""

    scaled_prefix = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "LDA": Pipeline(
            steps=scaled_prefix
            + [
                (
                    "classifier",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                )
            ]
        ),
        "Linear SVM": Pipeline(
            steps=scaled_prefix
            + [("classifier", SVC(kernel="linear", C=config.svm_c))]
        ),
        "Logistic Regression": Pipeline(
            steps=scaled_prefix
            + [
                (
                    "classifier",
                    LogisticRegression(
                        C=config.logistic_c,
                        max_iter=5_000,
                        random_state=config.random_seed,
                    ),
                )
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=config.random_forest_estimators,
                        random_state=config.random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "PLS-DA": Pipeline(
            steps=scaled_prefix
            + [
                (
                    "classifier",
                    PLSDAClassifier(n_components=config.pls_components),
                )
            ]
        ),
    }


def evaluate_with_loocv(
    subset: SelectedSubset,
    labels: IntArray,
    classifiers: Mapping[str, BaseEstimator],
) -> EvaluationResults:
    """Evaluate every classifier with leave-one-out cross-validation."""

    predictions = pd.DataFrame(
        {
            "Subject_Index": np.arange(labels.size, dtype=int),
            "Actual_Label": labels,
        }
    )
    metric_rows: list[dict[str, object]] = []
    leave_one_out = LeaveOneOut()

    for classifier_name, classifier_template in classifiers.items():
        predicted_labels = np.empty(labels.size, dtype=np.int_)
        decision_scores = np.empty(labels.size, dtype=np.float64)

        for training_indices, test_indices in leave_one_out.split(
            subset.predictors
        ):
            classifier = clone(classifier_template)
            classifier.fit(
                subset.predictors[training_indices],
                labels[training_indices],
            )
            predicted_labels[test_indices[0]] = int(
                classifier.predict(subset.predictors[test_indices])[0]
            )
            decision_scores[test_indices[0]] = _positive_class_score(
                classifier,
                subset.predictors[test_indices],
            )

        predictions[f"{classifier_name}_Prediction"] = predicted_labels
        predictions[f"{classifier_name}_Score"] = decision_scores
        metric_rows.append(
            _calculate_metrics(
                classifier_name,
                labels,
                predicted_labels,
                decision_scores,
            )
        )

    return EvaluationResults(
        metrics=pd.DataFrame(metric_rows),
        predictions=predictions,
    )


def _positive_class_score(classifier: BaseEstimator, predictors: FloatArray) -> float:
    """Extract a continuous positive-class score from a fitted classifier."""

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(predictors)
        return float(probabilities[0, 1])
    if hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(predictors)).ravel()
        return float(scores[0])
    return float(classifier.predict(predictors)[0])


def _calculate_metrics(
    classifier_name: str,
    actual: IntArray,
    predicted: IntArray,
    scores: FloatArray,
) -> dict[str, object]:
    """Calculate binary-classification metrics from all LOOCV predictions."""

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    )
    specificity_denominator = true_negative + false_positive
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator > 0
        else float("nan")
    )
    return {
        "Classifier": classifier_name,
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(actual, predicted),
        "Sensitivity_Recall": recall_score(actual, predicted, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "Matthews_Correlation": matthews_corrcoef(actual, predicted),
        "ROC_AUC": roc_auc_score(actual, scores),
        "True_Negative": int(true_negative),
        "False_Positive": int(false_positive),
        "False_Negative": int(false_negative),
        "True_Positive": int(true_positive),
    }


def write_results(
    results: EvaluationResults,
    subset: SelectedSubset,
    config: EvaluationConfig,
) -> tuple[Path, Path]:
    """Write aggregate metrics and subject-level predictions to CSV files."""

    config.output_directory.mkdir(parents=True, exist_ok=True)
    subset_key = "-".join(str(index) for index in subset.indices)
    file_prefix = f"{config.dataset_name}_features_{subset_key}"
    metrics_path = config.output_directory / f"{file_prefix}_loocv_metrics.csv"
    predictions_path = (
        config.output_directory / f"{file_prefix}_loocv_predictions.csv"
    )
    results.metrics.to_csv(metrics_path, index=False)
    results.predictions.to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


def print_results(
    results: EvaluationResults,
    subset: SelectedSubset,
    config: EvaluationConfig,
    subject_count: int,
) -> None:
    """Print the selected subset and aggregate LOOCV metrics."""

    print(f"\nDataset: {config.dataset_name}")
    print(f"Subjects: {subject_count}")
    print("Selected five-feature subset:")
    for index, name in zip(subset.indices, subset.names, strict=True):
        print(f"  {index}: {name}")
    print("\nLOOCV classifier metrics:\n")
    print(results.metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line evaluator and return its process exit code."""

    parsed = parse_arguments(arguments)
    configure_logging(parsed.log_level)
    try:
        config = config_from_arguments(parsed)
        dataset = load_dataset(config.dataset_name, config.data_path)
        subset = resolve_feature_subset(dataset, config)
        classifiers = build_classifiers(config)
        LOGGER.info(
            "Starting LOOCV | dataset=%s | subjects=%d | features=%s",
            config.dataset_name,
            dataset.subject_count,
            subset.indices,
        )
        results = evaluate_with_loocv(subset, dataset.labels, classifiers)
        print_results(results, subset, config, dataset.subject_count)
        metrics_path, predictions_path = write_results(results, subset, config)
        LOGGER.info("Saved metrics to %s", metrics_path)
        LOGGER.info("Saved predictions to %s", predictions_path)
    except KeyboardInterrupt:
        LOGGER.error("Evaluation interrupted by the user.")
        return 130
    except Exception:
        LOGGER.exception("Five-feature LOOCV evaluation failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
