"""Run Fisher-covariance feature selection with QAOA on IBM Runtime.

The program submits IBM Runtime jobs unless ``--validate-only`` is supplied.
Use ``python Fisher-covariance-QUBO.py --help`` to list all configuration
options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
from scipy.optimize import OptimizeResult, minimize


LOGGER = logging.getLogger("fisher_covariance_qubo")

SCRIPT_DIRECTORY: Final = Path(__file__).resolve().parent
DEFAULT_DATA_PATH: Final = ""
DEFAULT_SERVICE_NAME: Final = ""
DEFAULT_BACKEND_NAME: Final = ""
NUMERICAL_TOLERANCE: Final = 1e-12
MAPPING_ERROR_TOLERANCE: Final = 1e-10
MAPPING_RANDOM_STATE_COUNT: Final = 100

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Contains all values that control one QAOA experiment."""

    data_path: Path = DEFAULT_DATA_PATH
    service_name: str = DEFAULT_SERVICE_NAME
    backend_name: str = DEFAULT_BACKEND_NAME
    alpha: float = 0.77
    reps: int = 2
    shots: int = 10_000
    max_iterations: int = 6
    random_seed: int = 42
    transpiler_seed: int = 42
    optimization_level: int = 3

    def validate(self) -> None:
        """Validate the experiment configuration before performing any work.

        Raises:
            ValueError: A configuration value is outside its supported range.
        """

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0, inclusive.")
        if self.reps < 1:
            raise ValueError("reps must be at least 1.")
        if self.shots < 1:
            raise ValueError("shots must be at least 1.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if self.optimization_level not in range(4):
            raise ValueError("optimization_level must be 0, 1, 2, or 3.")
        if not self.service_name.strip():
            raise ValueError("service_name cannot be empty.")
        if not self.backend_name.strip():
            raise ValueError("backend_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class CommandLineOptions:
    """Contains parsed command-line behavior and experiment settings."""

    config: ExperimentConfig
    validate_only: bool
    log_level: str


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Contains the standardized predictors and binary outcome."""

    predictors: FloatArray
    outcome: IntArray
    feature_names: tuple[str, ...]

    @property
    def subject_count(self) -> int:
        """Return the number of subjects in the prepared dataset."""

        return int(self.predictors.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of candidate features."""

        return int(self.predictors.shape[1])


@dataclass(frozen=True, slots=True)
class QuboModel:
    """Contains the Fisher importance, redundancy, and QUBO matrices."""

    importance: FloatArray
    redundancy: FloatArray
    matrix: FloatArray


@dataclass(frozen=True, slots=True)
class IsingModel:
    """Contains the Ising representation of a QUBO model."""

    linear: FloatArray
    quadratic: FloatArray
    offset: float
    operator: SparsePauliOp
    maximum_mapping_error: float


@dataclass(frozen=True, slots=True)
class TranspiledCircuit:
    """Contains a transpiled QAOA circuit and its resource summary."""

    circuit: QuantumCircuit
    operation_counts: Mapping[str, int]
    gate_count: int
    depth: int
    qubit_count: int
    classical_bit_count: int
    width: int


@dataclass(frozen=True, slots=True)
class RuntimeOptimizationResult:
    """Contains the scientific outputs from the IBM Runtime optimization."""

    optimized_parameters: FloatArray
    optimizer_succeeded: bool
    optimizer_message: str
    evaluation_count: int
    sampled_expectation: float
    best_qiskit_bitstring: str
    best_feature_bitstring: str
    best_state: IntArray
    best_energy: float
    best_count: int
    selected_indices: tuple[int, ...]
    selected_feature_names: tuple[str, ...]


def configure_logging(level: str) -> None:
    """Configure process-wide console logging.

    Args:
        level: Standard Python logging level name.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> CommandLineOptions:
    """Parse command-line arguments.

    Args:
        arguments: Optional argument sequence. Uses ``sys.argv`` when omitted.

    Returns:
        The validated command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Fisher-covariance QUBO feature selection with QAOA on IBM "
            "Runtime."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the S1 CSV dataset.",
    )
    parser.add_argument(
        "--service-name",
        default=DEFAULT_SERVICE_NAME,
        help="Saved Qiskit Runtime account name.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND_NAME,
        help="IBM Quantum backend name.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.77,
        help="Importance-versus-redundancy weight in the interval [0, 1].",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="QAOA circuit depth p.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10_000,
        help="Sampler shots per objective evaluation and final sample.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum number of IBM Runtime objective evaluations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for initial parameters and mapping-validation states.",
    )
    parser.add_argument(
        "--transpiler-seed",
        type=int,
        default=42,
        help="Seed used by the Qiskit transpiler.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(4),
        default=3,
        help="Qiskit transpiler optimization level.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Prepare and validate the local QUBO and Ising models without "
            "connecting to IBM Runtime or submitting jobs."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level.",
    )

    parsed = parser.parse_args(arguments)
    config = ExperimentConfig(
        data_path=parsed.data_path.expanduser().resolve(),
        service_name=parsed.service_name,
        backend_name=parsed.backend,
        alpha=parsed.alpha,
        reps=parsed.reps,
        shots=parsed.shots,
        max_iterations=parsed.max_iterations,
        random_seed=parsed.random_seed,
        transpiler_seed=parsed.transpiler_seed,
        optimization_level=parsed.optimization_level,
    )
    config.validate()

    return CommandLineOptions(
        config=config,
        validate_only=parsed.validate_only,
        log_level=parsed.log_level,
    )


def load_and_prepare_dataset(data_path: Path) -> PreparedDataset:
    """Load and standardize the S1 ASD-versus-NEU dataset.

    The first CSV data row contains measurement units rather than a subject.
    SIB subjects are removed to retain the binary ASD-versus-NEU task. Missing
    predictor values are imputed with their feature medians before z-scoring.

    Args:
        data_path: Location of ``S1Dataset.csv``.

    Returns:
        Standardized predictors, binary outcomes, and ordered feature names.

    Raises:
        FileNotFoundError: The dataset does not exist.
        ValueError: Required columns, classes, or usable predictor values are
            missing.
    """

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path, header=0)
    required_columns = {"Group", "Vineland ABC"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")
    if data.empty:
        raise ValueError("Dataset contains no rows.")

    # The first row stores measurement units and is not an observation.
    data = data.iloc[1:].copy()
    data = data[data["Group"] != "SIB"].copy()

    observed_groups = set(data["Group"].dropna().astype(str).unique())
    expected_groups = {"ASD", "NEU"}
    unexpected_groups = observed_groups.difference(expected_groups)
    if unexpected_groups:
        group_text = ", ".join(sorted(unexpected_groups))
        raise ValueError(f"Unexpected diagnostic groups: {group_text}")
    if not expected_groups.issubset(observed_groups):
        raise ValueError("Both ASD and NEU subjects are required.")

    outcome = (data["Group"] == "ASD").to_numpy(dtype=np.int_)
    predictor_frame = data.drop(columns=["Group", "Vineland ABC"]).copy()
    if predictor_frame.shape[1] == 0:
        raise ValueError("Dataset contains no predictor columns.")

    predictor_frame = predictor_frame.apply(pd.to_numeric, errors="coerce")
    predictor_frame = predictor_frame.fillna(
        predictor_frame.median(numeric_only=True)
    )
    columns_with_missing_values = predictor_frame.columns[
        predictor_frame.isna().any()
    ].tolist()
    if columns_with_missing_values:
        column_text = ", ".join(columns_with_missing_values)
        raise ValueError(
            "Predictors remain missing after median imputation: " + column_text
        )

    predictors = predictor_frame.to_numpy(dtype=np.float64)
    means = np.mean(predictors, axis=0)
    standard_deviations = np.std(predictors, axis=0, ddof=0)

    # Preserve constant features without introducing division-by-zero values.
    safe_standard_deviations = standard_deviations.copy()
    safe_standard_deviations[safe_standard_deviations == 0.0] = 1.0
    standardized_predictors = (predictors - means) / safe_standard_deviations

    if not np.isfinite(standardized_predictors).all():
        raise ValueError("Standardized predictors contain non-finite values.")

    return PreparedDataset(
        predictors=np.asarray(standardized_predictors, dtype=np.float64),
        outcome=outcome,
        feature_names=tuple(str(column) for column in predictor_frame.columns),
    )


def build_fisher_covariance_qubo(
    dataset: PreparedDataset,
    alpha: float,
) -> QuboModel:
    """Construct the Fisher-importance and covariance-redundancy QUBO.

    The symmetric matrix follows the notebook's ``x.T @ Q @ x`` convention.
    Consequently, each stored off-diagonal coefficient contributes twice to
    the binary objective.

    Args:
        dataset: Prepared binary-classification dataset.
        alpha: Weight assigned to Fisher importance. ``1 - alpha`` is assigned
            to covariance redundancy.

    Returns:
        The importance vector, redundancy matrix, and symmetric QUBO matrix.
    """

    asd_predictors = dataset.predictors[dataset.outcome == 1]
    neu_predictors = dataset.predictors[dataset.outcome == 0]

    mean_difference = np.mean(asd_predictors, axis=0) - np.mean(
        neu_predictors, axis=0
    )
    variance_sum = np.var(asd_predictors, axis=0, ddof=0) + np.var(
        neu_predictors, axis=0, ddof=0
    )
    importance = np.divide(
        np.square(mean_difference),
        variance_sum,
        out=np.zeros(dataset.feature_count, dtype=np.float64),
        where=variance_sum > 0.0,
    )

    covariance = np.cov(dataset.predictors, rowvar=False, bias=True)
    redundancy = np.abs(np.asarray(covariance, dtype=np.float64))
    np.fill_diagonal(redundancy, 0.0)

    qubo_matrix = (1.0 - alpha) * redundancy
    np.fill_diagonal(qubo_matrix, -alpha * importance)

    if not np.allclose(redundancy, redundancy.T):
        raise ValueError("Covariance redundancy matrix is not symmetric.")
    if not np.allclose(np.diag(redundancy), 0.0):
        raise ValueError("Covariance redundancy matrix has a nonzero diagonal.")
    if not np.allclose(qubo_matrix, qubo_matrix.T):
        raise ValueError("QUBO matrix is not symmetric.")

    return QuboModel(
        importance=np.asarray(importance, dtype=np.float64),
        redundancy=redundancy,
        matrix=qubo_matrix,
    )


def calculate_qubo_energy(state: IntArray, qubo_matrix: FloatArray) -> float:
    """Calculate ``x.T @ Q @ x`` for one binary state.

    Args:
        state: Binary feature-selection state in feature-index order.
        qubo_matrix: Symmetric QUBO matrix.

    Returns:
        The scalar QUBO energy.
    """

    numeric_state = np.asarray(state, dtype=np.float64)
    return float(numeric_state @ qubo_matrix @ numeric_state)


def _calculate_ising_energy_without_offset(
    state: IntArray,
    linear: FloatArray,
    quadratic: FloatArray,
) -> float:
    """Calculate the state-dependent portion of an Ising energy."""

    spins = 1.0 - (2.0 * np.asarray(state, dtype=np.float64))
    linear_energy = float(linear @ spins)
    quadratic_energy = float(
        sum(
            quadratic[index, other_index] * spins[index] * spins[other_index]
            for index in range(state.size)
            for other_index in range(index + 1, state.size)
        )
    )
    return linear_energy + quadratic_energy


def build_ising_model(
    qubo_matrix: FloatArray,
    random_seed: int,
) -> IsingModel:
    """Convert a symmetric QUBO matrix to a validated Ising Hamiltonian.

    Args:
        qubo_matrix: Matrix interpreted with the ``x.T @ Q @ x`` convention.
        random_seed: Seed for randomized mapping-validation states.

    Returns:
        Ising coefficients, constant offset, Qiskit operator, and validation
        error.

    Raises:
        ValueError: The matrix is invalid or the mapping exceeds tolerance.
    """

    if qubo_matrix.ndim != 2 or qubo_matrix.shape[0] != qubo_matrix.shape[1]:
        raise ValueError("QUBO matrix must be square.")
    if not np.allclose(qubo_matrix, qubo_matrix.T):
        raise ValueError("QUBO matrix must be symmetric.")

    feature_count = qubo_matrix.shape[0]
    diagonal = np.diag(qubo_matrix)
    off_diagonal_row_sums = np.sum(qubo_matrix, axis=1) - diagonal
    linear = -0.5 * (diagonal + off_diagonal_row_sums)

    quadratic = (qubo_matrix + qubo_matrix.T) / 4.0
    np.fill_diagonal(quadratic, 0.0)

    offset = 0.5 * float(np.trace(qubo_matrix))
    offset += float(np.sum(np.triu(quadratic, k=1)))

    pauli_labels: list[str] = []
    coefficients: list[float] = []

    for index, coefficient in enumerate(linear):
        if abs(coefficient) <= NUMERICAL_TOLERANCE:
            continue
        label = ["I"] * feature_count
        label[index] = "Z"
        # Qiskit labels are big-endian; qubit zero is the rightmost character.
        pauli_labels.append("".join(reversed(label)))
        coefficients.append(float(coefficient))

    for index in range(feature_count):
        for other_index in range(index + 1, feature_count):
            coefficient = quadratic[index, other_index]
            if abs(coefficient) <= NUMERICAL_TOLERANCE:
                continue
            label = ["I"] * feature_count
            label[index] = "Z"
            label[other_index] = "Z"
            pauli_labels.append("".join(reversed(label)))
            coefficients.append(float(coefficient))

    if not pauli_labels:
        raise ValueError("Ising Hamiltonian contains no nonzero Pauli terms.")

    operator = SparsePauliOp(
        pauli_labels,
        coeffs=np.asarray(coefficients, dtype=np.float64),
    )
    maximum_mapping_error = validate_qubo_to_ising_mapping(
        qubo_matrix=qubo_matrix,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        random_seed=random_seed,
    )

    return IsingModel(
        linear=np.asarray(linear, dtype=np.float64),
        quadratic=np.asarray(quadratic, dtype=np.float64),
        offset=offset,
        operator=operator,
        maximum_mapping_error=maximum_mapping_error,
    )


def validate_qubo_to_ising_mapping(
    qubo_matrix: FloatArray,
    linear: FloatArray,
    quadratic: FloatArray,
    offset: float,
    random_seed: int,
) -> float:
    """Validate the QUBO-to-Ising mapping on deterministic state classes.

    Zero, one, singleton, and pair states determine all linear and quadratic
    coefficients. Random states provide an additional end-to-end check without
    enumerating all ``2**n`` feature subsets.

    Args:
        qubo_matrix: Symmetric QUBO matrix.
        linear: Linear Ising coefficients.
        quadratic: Symmetric pairwise Ising coefficients.
        offset: Constant QUBO-to-Ising energy offset.
        random_seed: Seed for the supplemental random states.

    Returns:
        Maximum absolute mapping error across all validation states.

    Raises:
        ValueError: The maximum error exceeds ``MAPPING_ERROR_TOLERANCE``.
    """

    feature_count = qubo_matrix.shape[0]
    validation_states: list[IntArray] = [
        np.zeros(feature_count, dtype=np.int_),
        np.ones(feature_count, dtype=np.int_),
    ]
    validation_states.extend(np.eye(feature_count, dtype=np.int_))

    for index in range(feature_count):
        for other_index in range(index + 1, feature_count):
            pair_state = np.zeros(feature_count, dtype=np.int_)
            pair_state[[index, other_index]] = 1
            validation_states.append(pair_state)

    random_number_generator = np.random.default_rng(random_seed)
    validation_states.extend(
        random_number_generator.integers(
            0,
            2,
            size=(MAPPING_RANDOM_STATE_COUNT, feature_count),
            dtype=np.int_,
        )
    )

    mapping_errors = [
        abs(
            calculate_qubo_energy(state, qubo_matrix)
            - (
                offset
                + _calculate_ising_energy_without_offset(
                    state,
                    linear,
                    quadratic,
                )
            )
        )
        for state in validation_states
    ]
    maximum_mapping_error = max(mapping_errors)
    if maximum_mapping_error >= MAPPING_ERROR_TOLERANCE:
        raise ValueError(
            "QUBO-to-Ising mapping error exceeds tolerance: "
            f"{maximum_mapping_error:.3e}"
        )

    return maximum_mapping_error


def build_qaoa_ansatz(ising_model: IsingModel, reps: int) -> QuantumCircuit:
    """Build the measured, parameterized QAOA ansatz.

    Args:
        ising_model: Validated cost Hamiltonian.
        reps: Number of QAOA cost-and-mixer layers.

    Returns:
        A measured QAOA circuit with ``2 * reps`` free parameters.
    """

    ansatz = QAOAAnsatz(cost_operator=ising_model.operator, reps=reps)
    ansatz.measure_all()

    expected_parameter_count = 2 * reps
    if ansatz.num_parameters != expected_parameter_count:
        raise ValueError(
            f"Expected {expected_parameter_count} QAOA parameters, found "
            f"{ansatz.num_parameters}."
        )

    return ansatz


def transpile_for_backend(
    ansatz: QuantumCircuit,
    backend: Any,
    config: ExperimentConfig,
) -> TranspiledCircuit:
    """Transpile a parameterized QAOA circuit for an IBM backend.

    Args:
        ansatz: Measured QAOA circuit.
        backend: IBM Runtime backend instance.
        config: Experiment and transpiler settings.

    Returns:
        The ISA circuit and a stable summary of its resources.
    """

    pass_manager = generate_preset_pass_manager(
        optimization_level=config.optimization_level,
        backend=backend,
        seed_transpiler=config.transpiler_seed,
    )
    isa_circuit = pass_manager.run(ansatz)
    operation_counts = {
        str(name): int(count) for name, count in isa_circuit.count_ops().items()
    }

    return TranspiledCircuit(
        circuit=isa_circuit,
        operation_counts=operation_counts,
        gate_count=int(isa_circuit.size()),
        depth=int(isa_circuit.depth()),
        qubit_count=int(isa_circuit.num_qubits),
        classical_bit_count=int(isa_circuit.num_clbits),
        width=int(isa_circuit.width()),
    )


def qiskit_bitstring_to_state(bitstring: str, feature_count: int) -> IntArray:
    """Convert a Qiskit count key to feature-index order.

    Qiskit displays ``c[n-1]...c[0]``. Reversing the displayed key produces
    ``x[0], x[1], ..., x[n-1]``, which matches the QUBO feature ordering.

    Args:
        bitstring: Measurement key returned by Qiskit.
        feature_count: Expected number of measured feature qubits.

    Returns:
        Binary state in ascending feature-index order.

    Raises:
        ValueError: The measurement key has an unexpected format or length.
    """

    displayed_bits = bitstring.replace(" ", "")
    if (
        len(displayed_bits) != feature_count
        or set(displayed_bits).difference({"0", "1"})
    ):
        raise ValueError(f"Unexpected measurement key: {bitstring!r}")

    return np.fromiter(
        (int(bit) for bit in reversed(displayed_bits)),
        dtype=np.int_,
        count=feature_count,
    )


def calculate_sampled_qubo_expectation(
    counts: Mapping[str, int],
    qubo_matrix: FloatArray,
) -> float:
    """Calculate the empirical QUBO expectation from sampler counts.

    Args:
        counts: Qiskit measurement counts.
        qubo_matrix: Symmetric QUBO matrix.

    Returns:
        Shot-weighted QUBO expectation value.

    Raises:
        ValueError: The sampler returned no positive counts.
    """

    total_count = sum(counts.values())
    if total_count <= 0:
        raise ValueError("Sampler returned no counts.")

    feature_count = qubo_matrix.shape[0]
    return float(
        sum(
            (count / total_count)
            * calculate_qubo_energy(
                qiskit_bitstring_to_state(bitstring, feature_count),
                qubo_matrix,
            )
            for bitstring, count in counts.items()
        )
    )


def find_lowest_energy_observation(
    counts: Mapping[str, int],
    qubo_matrix: FloatArray,
) -> tuple[str, IntArray, float, int]:
    """Find the lowest-energy state observed in sampler counts.

    Energy is the scientific selection criterion. Count and bitstring only
    provide deterministic ordering when multiple observations have equal
    energy.

    Args:
        counts: Qiskit measurement counts.
        qubo_matrix: Symmetric QUBO matrix.

    Returns:
        Qiskit key, feature-ordered state, energy, and observation count.

    Raises:
        ValueError: The sampler returned no observations.
    """

    if not counts:
        raise ValueError("Sampler returned no observations.")

    feature_count = qubo_matrix.shape[0]
    candidates: list[tuple[float, int, str, IntArray]] = []
    for bitstring, count in counts.items():
        state = qiskit_bitstring_to_state(bitstring, feature_count)
        candidates.append(
            (
                calculate_qubo_energy(state, qubo_matrix),
                -count,
                bitstring,
                state,
            )
        )

    energy, negative_count, bitstring, state = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )
    return bitstring, state, energy, -negative_count


def run_runtime_optimization(
    backend: Any,
    isa_circuit: QuantumCircuit,
    dataset: PreparedDataset,
    qubo_model: QuboModel,
    config: ExperimentConfig,
) -> RuntimeOptimizationResult:
    """Optimize and sample the transpiled QAOA circuit through IBM Runtime.

    Args:
        backend: IBM Runtime backend instance.
        isa_circuit: Transpiled parameterized QAOA circuit.
        dataset: Prepared dataset used to map selected feature names.
        qubo_model: QUBO used to score sampled states.
        config: Runtime, optimizer, and sampling settings.

    Returns:
        Optimizer diagnostics and the lowest-energy final observation.
    """

    random_number_generator = np.random.default_rng(config.random_seed)
    initial_parameters = random_number_generator.uniform(
        0.0,
        2.0 * np.pi,
        size=isa_circuit.num_parameters,
    )
    evaluation_count = 0

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)

        def objective(parameters: FloatArray) -> float:
            """Submit one parameterized circuit and estimate QUBO energy."""

            nonlocal evaluation_count

            # SciPy can request more calls than maxiter for small COBYLA runs.
            # Do not convert those bookkeeping calls into paid runtime jobs.
            if evaluation_count >= config.max_iterations:
                return float("inf")

            publication = (
                isa_circuit,
                np.asarray(parameters, dtype=np.float64),
                config.shots,
            )
            sampler_result = sampler.run([publication]).result()
            counts = sampler_result[0].data.meas.get_counts()
            expectation = calculate_sampled_qubo_expectation(
                counts,
                qubo_model.matrix,
            )

            evaluation_count += 1
            LOGGER.info(
                "Evaluation %d/%d | sampled QUBO expectation = %.12f",
                evaluation_count,
                config.max_iterations,
                expectation,
            )
            return expectation

        LOGGER.info(
            "Starting COBYLA | p=%d | max evaluations=%d | shots=%d",
            config.reps,
            config.max_iterations,
            config.shots,
        )
        optimizer_result: OptimizeResult = minimize(
            objective,
            initial_parameters,
            method="COBYLA",
            options={
                "maxiter": config.max_iterations,
                "rhobeg": 1.0,
                "catol": 1e-6,
            },
        )

        # Use an independent final sample so optimizer noise does not determine
        # the reported feature subset.
        final_publication = (
            isa_circuit,
            np.asarray(optimizer_result.x, dtype=np.float64),
            config.shots,
        )
        final_sampler_result = sampler.run([final_publication]).result()
        final_counts = final_sampler_result[0].data.meas.get_counts()

    sampled_expectation = calculate_sampled_qubo_expectation(
        final_counts,
        qubo_model.matrix,
    )
    qiskit_bitstring, best_state, best_energy, best_count = (
        find_lowest_energy_observation(final_counts, qubo_model.matrix)
    )
    selected_indices = tuple(
        int(index) for index in np.flatnonzero(best_state).tolist()
    )
    selected_feature_names = tuple(
        dataset.feature_names[index] for index in selected_indices
    )
    feature_bitstring = "".join(str(int(bit)) for bit in best_state)

    return RuntimeOptimizationResult(
        optimized_parameters=np.asarray(optimizer_result.x, dtype=np.float64),
        optimizer_succeeded=bool(optimizer_result.success),
        optimizer_message=str(optimizer_result.message),
        evaluation_count=evaluation_count,
        sampled_expectation=sampled_expectation,
        best_qiskit_bitstring=qiskit_bitstring,
        best_feature_bitstring=feature_bitstring,
        best_state=best_state,
        best_energy=best_energy,
        best_count=best_count,
        selected_indices=selected_indices,
        selected_feature_names=selected_feature_names,
    )


def log_local_model_summary(
    dataset: PreparedDataset,
    qubo_model: QuboModel,
    ising_model: IsingModel,
    ansatz: QuantumCircuit,
    config: ExperimentConfig,
) -> None:
    """Log the local preprocessing and model-construction results."""

    LOGGER.info(
        "Prepared dataset | subjects=%d | features=%d",
        dataset.subject_count,
        dataset.feature_count,
    )
    LOGGER.info("Feature index and Fisher scores:")
    for index, (name, score) in enumerate(
        zip(dataset.feature_names, qubo_model.importance, strict=True)
    ):
        LOGGER.info("%2d -> %s | Fisher score = %.6f", index, name, score)

    LOGGER.info(
        "Ising Hamiltonian | Pauli terms=%d | offset=%.12f | "
        "maximum mapping error=%.3e",
        len(ising_model.operator),
        ising_model.offset,
        ising_model.maximum_mapping_error,
    )
    LOGGER.info(
        "QAOA ansatz | logical qubits=%d | p=%d | parameters=%d | alpha=%.4f",
        ansatz.num_qubits,
        config.reps,
        ansatz.num_parameters,
        config.alpha,
    )


def log_transpiled_circuit_summary(
    transpiled_circuit: TranspiledCircuit,
    backend_name: str,
) -> None:
    """Log stable resource metrics for a transpiled QAOA circuit."""

    LOGGER.info(
        "Transpiled circuit | backend=%s | gates=%d | depth=%d | "
        "qubits=%d | classical bits=%d | width=%d",
        backend_name,
        transpiled_circuit.gate_count,
        transpiled_circuit.depth,
        transpiled_circuit.qubit_count,
        transpiled_circuit.classical_bit_count,
        transpiled_circuit.width,
    )
    LOGGER.info("Transpiled operations: %s", transpiled_circuit.operation_counts)


def log_runtime_result(
    result: RuntimeOptimizationResult,
    config: ExperimentConfig,
) -> None:
    """Log the final IBM Runtime optimization result."""

    LOGGER.info("Final Fisher-covariance QUBO + QAOA result")
    LOGGER.info("Alpha: %.4f", config.alpha)
    LOGGER.info("QAOA depth p: %d", config.reps)
    LOGGER.info("COBYLA evaluations: %d", result.evaluation_count)
    LOGGER.info("Shots per evaluation: %d", config.shots)
    LOGGER.info("Optimizer succeeded: %s", result.optimizer_succeeded)
    LOGGER.info("Optimizer message: %s", result.optimizer_message)
    LOGGER.info(
        "Optimized sampled expectation: %.12f",
        result.sampled_expectation,
    )
    LOGGER.info(
        "Best bitstring (x[0]...x[%d]): %s",
        result.best_state.size - 1,
        result.best_feature_bitstring,
    )
    LOGGER.info("Best Qiskit count key: %s", result.best_qiskit_bitstring)
    LOGGER.info("Best QUBO energy: %.12f", result.best_energy)
    LOGGER.info(
        "Best-state frequency in %d shots: %d",
        config.shots,
        result.best_count,
    )
    LOGGER.info("Selected feature count: %d", len(result.selected_indices))
    LOGGER.info("Selected feature indices: %s", list(result.selected_indices))
    LOGGER.info("Selected feature names: %s", list(result.selected_feature_names))


def run(options: CommandLineOptions) -> RuntimeOptimizationResult | None:
    """Execute the configured local and remote workflow.

    Args:
        options: Parsed command-line settings.

    Returns:
        Runtime results, or ``None`` when local validation was requested.
    """

    config = options.config
    dataset = load_and_prepare_dataset(config.data_path)
    qubo_model = build_fisher_covariance_qubo(dataset, config.alpha)
    ising_model = build_ising_model(qubo_model.matrix, config.random_seed)
    ansatz = build_qaoa_ansatz(ising_model, config.reps)
    log_local_model_summary(dataset, qubo_model, ising_model, ansatz, config)

    if options.validate_only:
        LOGGER.info(
            "Local validation completed. No IBM Runtime jobs were submitted."
        )
        return None

    LOGGER.info(
        "Loading IBM Runtime service profile '%s' and backend '%s'.",
        config.service_name,
        config.backend_name,
    )
    service = QiskitRuntimeService(name=config.service_name)
    backend = service.backend(config.backend_name)
    transpiled_circuit = transpile_for_backend(ansatz, backend, config)
    log_transpiled_circuit_summary(transpiled_circuit, config.backend_name)

    result = run_runtime_optimization(
        backend=backend,
        isa_circuit=transpiled_circuit.circuit,
        dataset=dataset,
        qubo_model=qubo_model,
        config=config,
    )
    log_runtime_result(result, config)
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line program and return its process exit code."""

    options = parse_arguments(arguments)
    configure_logging(options.log_level)

    try:
        run(options)
    except KeyboardInterrupt:
        LOGGER.error("Execution interrupted by the user.")
        return 130
    except Exception:
        LOGGER.exception("Fisher-covariance QUBO execution failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
