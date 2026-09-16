%% Mutual-information QUBO feature selection for Shotgun dataset
%
% Dataset format:
% - First row: subject labels in columns 2:end
% - First column: feature names in rows 2:end
% - Rows: features
% - Columns: subjects
%
% Preprocessing matches the Python implementation:
% - Convert measurements to numeric
% - Remove feature rows containing any missing values
% - Keep ASD, TD, and NEU subjects
% - Encode ASD = 1 and TD/NEU = 0
% - Retain the first n_keep valid features
% - Transpose to subjects-by-features format
% - Z-score using population standard deviation
% - Discretize each feature into 20 quantile bins
%
% MI-QUBO:
%
%   I_i  = MI(feature_i, class label)
%   R_ij = MI(feature_i, feature_j)
%
%   Q(i,i) = -alpha * I_i
%   Q(i,j) = (1-alpha) * R_ij
%
% MATLAB minimizes:
%
%   x' * Q * x
%
% Since Q is symmetric:
%
%   x'Qx = sum_i Q(i,i)x_i
%          + 2*sum_{i<j}Q(i,j)x_i*x_j

clear;
clc;

%% ------------------------------------------------------------
% User settings
%% ------------------------------------------------------------

alpha = 0.98;

% Number of valid Shotgun features to retain
n_keep = 54;

% Number of quantile bins
B = 20;

dataset_file = "datasets/shotgun_prescreen_54.csv";

% MATLAB tabu-search settings
max_time_seconds       = 300;
max_iterations         = 1e9;
max_stall_time_seconds = 60;
max_stall_iterations   = 1e6;

% Numerical threshold for counting and printing QUBO terms
term_tolerance = 1e-12;

% Reproducibility for stochastic tabu search
rng(42, "twister");

%% ------------------------------------------------------------
% Load Shotgun dataset
%% ------------------------------------------------------------

raw = readcell(dataset_file);

if size(raw, 1) < 2 || size(raw, 2) < 2
    error("The Shotgun dataset does not contain sufficient data.");
end

fprintf("Original dataset dimensions: %d rows x %d columns\n", ...
    size(raw, 1), size(raw, 2));

%% ------------------------------------------------------------
% Extract subject labels
%
% Python:
% labels = raw.iloc[0, 1:].astype(str).values
%% ------------------------------------------------------------

labels = strip(string(raw(1, 2:end)));
labels(ismissing(labels)) = "";
labels = labels(:);

%% ------------------------------------------------------------
% Extract feature names
%
% Python:
% feature_names_all = raw.iloc[1:, 0].astype(str).values
%% ------------------------------------------------------------

feature_names_all = strip(string(raw(2:end, 1)));
feature_names_all(ismissing(feature_names_all)) = "";
feature_names_all = feature_names_all(:);

%% ------------------------------------------------------------
% Extract feature-by-subject measurement matrix
%% ------------------------------------------------------------

Xraw = raw(2:end, 2:end);

n_features_original = size(Xraw, 1);
n_subjects_original = size(Xraw, 2);

if numel(feature_names_all) ~= n_features_original
    error("The number of feature names does not match the data rows.");
end

if numel(labels) ~= n_subjects_original
    error("The number of subject labels does not match the data columns.");
end

%% ------------------------------------------------------------
% Convert all measurements to numeric
%
% Equivalent to:
% pd.to_numeric(..., errors="coerce")
%
% Nonconvertible values become NaN.
%% ------------------------------------------------------------

Xnum = nan(n_features_original, n_subjects_original);

for i = 1:n_features_original

    for j = 1:n_subjects_original

        value = Xraw{i, j};

        if isnumeric(value) && isscalar(value)

            Xnum(i, j) = double(value);

        elseif islogical(value) && isscalar(value)

            Xnum(i, j) = double(value);

        elseif isempty(value)

            Xnum(i, j) = NaN;

        else

            value_string = strip(string(value));

            if ismissing(value_string) || strlength(value_string) == 0
                Xnum(i, j) = NaN;
            else
                Xnum(i, j) = str2double(value_string);
            end

        end
    end
end

%% ------------------------------------------------------------
% Remove feature rows containing any missing value
%
% This occurs before subject filtering, matching the Python code.
%% ------------------------------------------------------------

valid_feature_mask = ~any(isnan(Xnum), 2);

Xnum = Xnum(valid_feature_mask, :);
feature_names_all = feature_names_all(valid_feature_mask);

n_removed_missing = sum(~valid_feature_mask);

fprintf("Features removed because of missing values: %d\n", ...
    n_removed_missing);

fprintf("Valid features after missing-value filtering: %d\n", ...
    size(Xnum, 1));

if isempty(Xnum)
    error("No valid features remain after missing-value filtering.");
end

%% ------------------------------------------------------------
% Keep ASD, TD, and NEU subjects
%
% Python:
%
% keep_subject_mask = ...
%     (labels == "ASD") | ...
%     (labels == "TD")  | ...
%     (labels == "NEU")
%% ------------------------------------------------------------

keep_subject_mask = ...
    (labels == "ASD") | ...
    (labels == "TD")  | ...
    (labels == "NEU");

labels = labels(keep_subject_mask);
Xnum = Xnum(:, keep_subject_mask);

if isempty(labels)
    error("No ASD, TD, or NEU subjects were found.");
end

%% ------------------------------------------------------------
% Binary labels
%
% ASD    = 1
% TD/NEU = 0
%% ------------------------------------------------------------

y = double(labels == "ASD");

n_asd = sum(labels == "ASD");
n_td = sum(labels == "TD");
n_neu = sum(labels == "NEU");
n_control = sum(y == 0);

fprintf("Subjects retained: %d\n", numel(y));
fprintf("ASD subjects: %d\n", n_asd);
fprintf("TD subjects: %d\n", n_td);
fprintf("NEU subjects: %d\n", n_neu);
fprintf("Combined TD/NEU controls: %d\n", n_control);

if n_asd == 0 || n_control == 0
    error("Both ASD and TD/NEU control subjects are required.");
end

%% ------------------------------------------------------------
% Keep the first n_keep valid features
%
% Python:
% X_feature_by_subject =
%     X_feature_by_subject.iloc[:n_keep, :]
%% ------------------------------------------------------------

n_valid_features = size(Xnum, 1);
n_features_to_keep = min(n_keep, n_valid_features);

if n_valid_features < n_keep

    warning( ...
        "Only %d valid features are available; requested n_keep was %d.", ...
        n_valid_features, n_keep);

end

Xnum = Xnum(1:n_features_to_keep, :);
feature_names = feature_names_all(1:n_features_to_keep);

%% ------------------------------------------------------------
% Transpose into standard machine-learning format
%
% Rows    = subjects
% Columns = features
%% ------------------------------------------------------------

X = Xnum';

n_subjects = size(X, 1);
n_features = size(X, 2);

fprintf("Features retained for QUBO: %d\n", n_features);

%% ------------------------------------------------------------
% Z-score normalization
%
% Python StandardScaler uses population standard deviation:
%
%   sigma = sqrt(sum((x-mu).^2)/N)
%
% MATLAB std(X,1,1) also normalizes by N.
%% ------------------------------------------------------------

feature_mean = mean(X, 1);
feature_std = std(X, 1, 1);

% Match StandardScaler behavior for constant features.
constant_feature_mask = ...
    (feature_std == 0) | ...
    isnan(feature_std) | ...
    isinf(feature_std);

if any(constant_feature_mask)

    fprintf("Constant features assigned unit scale: %d\n", ...
        sum(constant_feature_mask));

    feature_std(constant_feature_mask) = 1;

end

Xz = (X - feature_mean) ./ feature_std;

if any(~isfinite(Xz), "all")
    error("The standardized feature matrix contains nonfinite values.");
end

%% ------------------------------------------------------------
% Quantile discretization into B bins
%
% Python:
%
% edges = np.quantile(
%     x,
%     np.linspace(0,1,B+1)
% )
%
% edges = np.unique(edges)
%
% if len(edges) <= 2:
%     return zeros(...)
%
% np.digitize(
%     x,
%     edges[1:-1],
%     right=True
% )
%% ------------------------------------------------------------

X_binned = zeros(n_subjects, n_features);
effective_bin_counts = zeros(1, n_features);

for j = 1:n_features

    [X_binned(:, j), effective_bin_counts(j)] = ...
        quantileBinLikeNumPy(Xz(:, j), B);

end

fprintf("Requested quantile bins: %d\n", B);

%% ------------------------------------------------------------
% Mutual-information importance
%
% I_i = MI(feature_i, class label)
%
% Natural logarithms are used, matching sklearn's
% mutual_info_score. MI is therefore measured in nats.
%% ------------------------------------------------------------

I = zeros(n_features, 1);

for i = 1:n_features

    I(i) = mutualInformationDiscrete( ...
        X_binned(:, i), ...
        y);

end

%% ------------------------------------------------------------
% Mutual-information redundancy
%
% R_ij = MI(feature_i, feature_j)
% R_ii = 0
%% ------------------------------------------------------------

R = zeros(n_features, n_features);

for i = 1:n_features

    for j = i+1:n_features

        rij = mutualInformationDiscrete( ...
            X_binned(:, i), ...
            X_binned(:, j));

        R(i, j) = rij;
        R(j, i) = rij;

    end
end

R(1:n_features+1:end) = 0;

%% ------------------------------------------------------------
% Build symmetric MI-QUBO matrix
%
% Diagonal:
%   Q_ii = -alpha * I_i
%
% Off-diagonal:
%   Q_ij = (1-alpha) * R_ij
%
% The supplied Python implementation does not scale I or R.
%% ------------------------------------------------------------

Q = zeros(n_features, n_features);

for i = 1:n_features
    Q(i, i) = -alpha * I(i);
end

for i = 1:n_features

    for j = i+1:n_features

        qij = (1-alpha) * R(i, j);

        Q(i, j) = qij;
        Q(j, i) = qij;

    end
end

%% ------------------------------------------------------------
% Confirm matrix symmetry
%% ------------------------------------------------------------

symmetry_error = max(abs(Q - Q'), [], "all");

if symmetry_error > term_tolerance
    error("The QUBO matrix is not symmetric.");
end

%% ------------------------------------------------------------
% Count equivalent Iskay terms
%
% The Python hubo dictionary contains:
% - One constant term
% - Nonzero diagonal terms
% - Nonzero upper-triangular quadratic terms
%% ------------------------------------------------------------

linear_term_count = sum( ...
    abs(diag(Q)) > term_tolerance);

upper_Q = triu(Q, 1);

quadratic_term_count = sum( ...
    abs(2 * upper_Q) > term_tolerance, ...
    "all");

total_term_count = ...
    1 + linear_term_count + quadratic_term_count;

%% ------------------------------------------------------------
% Print dataset and QUBO information
%% ------------------------------------------------------------

fprintf("\nSubjects: %d\n", n_subjects);

fprintf("Variables: %d | Terms: %d\n", ...
    n_features, total_term_count);

%% ------------------------------------------------------------
% Print MI importance scores
%
% Indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nMI importance scores:\n");

for i = 1:n_features

    fprintf( ...
        "%2d. %s | MI(label) = %.6f\n", ...
        i-1, ...
        char(feature_names(i)), ...
        I(i));

end

%% ------------------------------------------------------------
% Print equivalent Iskay QUBO terms
%
% MATLAB does not require this dictionary. It is printed to
% permit direct comparison with the Python output.
%% ------------------------------------------------------------

fprintf("\nQUBO terms equivalent to Iskay dictionary:\n");

% Constant term
fprintf("(): %.6f\n", 0.0);

% Linear terms
for i = 1:n_features

    coefficient = Q(i, i);

    if abs(coefficient) > term_tolerance

        fprintf( ...
            "(%d,): %.6f\n", ...
            i-1, ...
            coefficient);

    end
end

% Quadratic terms
for i = 1:n_features

    for j = i+1:n_features

        % Iskay stores each pair once, so its coefficient is
        % 2*Q(i,j).
        coefficient = 2 * Q(i, j);

        if abs(coefficient) > term_tolerance

            fprintf( ...
                "(%d, %d): %.6f\n", ...
                i-1, ...
                j-1, ...
                coefficient);

        end
    end
end

%% ------------------------------------------------------------
% Feature index mapping
%
% Indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nIndex to feature mapping:\n");

for i = 1:n_features

    fprintf( ...
        "%d -> %s\n", ...
        i-1, ...
        char(feature_names(i)));

end

fprintf("\nAlpha: %.6f\n", alpha);
fprintf("Quantile bins: %d\n", B);

%% ------------------------------------------------------------
% Create MATLAB QUBO problem
%
% MATLAB minimizes:
%
%   x'Qx
%
% Supply the complete symmetric matrix directly. Do not multiply
% the off-diagonal matrix entries by two again.
%% ------------------------------------------------------------

qprob = qubo(Q);

%% ------------------------------------------------------------
% Configure MATLAB tabu-search optimizer
%% ------------------------------------------------------------

ts = tabuSearch( ...
    MaxTime            = max_time_seconds, ...
    MaxIterations      = max_iterations, ...
    MaxStallTime       = max_stall_time_seconds, ...
    MaxStallIterations = max_stall_iterations, ...
    Display            = "final");

%% ------------------------------------------------------------
% Solve QUBO and measure solver runtime
%
% Only the solve call is timed. Dataset loading, preprocessing,
% discretization, MI calculations, QUBO construction, and
% printing are excluded.
%% ------------------------------------------------------------

fprintf("\nStarting MATLAB tabu-search MI-QUBO solver...\n");

solve_timer = tic;

result = solve( ...
    qprob, ...
    Algorithm = ts);

qubo_runtime_seconds = toc(solve_timer);

%% ------------------------------------------------------------
% Retrieve solution
%% ------------------------------------------------------------

x_best = double(result.BestX(:));

selected_matlab_indices = find(x_best > 0.5);

best_value = result.BestFunctionValue;

% Independent objective verification
manual_objective_value = x_best' * Q * x_best;

%% ------------------------------------------------------------
% Display raw solver output
%% ------------------------------------------------------------

fprintf("\nRaw QUBO result:\n");
disp(result);

fprintf("\nRaw tabu-search result:\n");
disp(result.AlgorithmResult);

%% ------------------------------------------------------------
% Final summary
%% ------------------------------------------------------------

fprintf("\nSubjects: %d\n", n_subjects);
fprintf("Variables: %d\n", n_features);
fprintf("Terms: %d\n", total_term_count);

fprintf("Selected feature count: %d\n", ...
    numel(selected_matlab_indices));

fprintf("Best QUBO objective value: %.10f\n", ...
    best_value);

fprintf("Manual x''Qx objective check: %.10f\n", ...
    manual_objective_value);

fprintf("Objective difference: %.3e\n", ...
    abs(best_value - manual_objective_value));

fprintf("MATLAB MI-QUBO solve runtime: %.6f seconds\n", ...
    qubo_runtime_seconds);

%% ------------------------------------------------------------
% Print selected features
%
% Indices are zero-based to match Python.
% The complete binary feature mask is not printed.
%% ------------------------------------------------------------

fprintf("\nSelected features from mapped solution:\n");

if isempty(selected_matlab_indices)

    fprintf("No features selected.\n");

else

    for k = 1:numel(selected_matlab_indices)

        matlab_index = selected_matlab_indices(k);
        python_index = matlab_index - 1;

        fprintf( ...
            "%d -> %s | MI(label) = %.6f\n", ...
            python_index, ...
            char(feature_names(matlab_index)), ...
            I(matlab_index));

    end
end

%% ============================================================
% Local helper functions
%% ============================================================

function [bins, effective_bin_count] = ...
    quantileBinLikeNumPy(x, n_bins)
%QUANTILEBINLIKENUMPY Match the supplied Python quantile binning.
%
% Reproduces:
%
%   edges = np.quantile(
%       x,
%       np.linspace(0,1,n_bins+1)
%   )
%
%   edges = np.unique(edges)
%
%   if len(edges) <= 2:
%       return np.zeros(len(x), dtype=int)
%
%   np.digitize(
%       x,
%       edges(2:end-1),
%       right=True
%   )

    x = double(x(:));

    if isempty(x)
        error("Cannot discretize an empty feature.");
    end

    if any(~isfinite(x))
        error("Quantile discretization received nonfinite values.");
    end

    probabilities = linspace(0, 1, n_bins + 1);

    % Match NumPy's default linear quantile interpolation.
    edges = numpyLinearQuantiles(x, probabilities);

    % Match np.unique.
    edges = unique(edges);

    % Match the Python degenerate-feature condition.
    if numel(edges) <= 2

        bins = zeros(numel(x), 1);
        effective_bin_count = 1;
        return;

    end

    interior_edges = edges(2:end-1);
    interior_edges = interior_edges(:)';

    % For np.digitize(..., right=True), values equal to an edge
    % remain in the lower bin. Thus, the bin index is the number
    % of interior edges strictly smaller than the observation.
    bins = sum(x > interior_edges, 2);
    bins = double(bins);

    effective_bin_count = numel(unique(bins));

end


function q = numpyLinearQuantiles(x, probabilities)
%NUMPYLINEARQUANTILES Match np.quantile method="linear".
%
% NumPy's linear interpolation uses:
%
%   h = (N-1)*p
%
% and interpolates between floor(h) and ceil(h).

    x = sort(double(x(:)));
    probabilities = double(probabilities(:)');

    n = numel(x);

    if n == 0
        error("Cannot calculate quantiles of an empty vector.");
    end

    if any(probabilities < 0 | probabilities > 1)
        error("Quantile probabilities must be between 0 and 1.");
    end

    h = (n-1) .* probabilities;

    lower_zero_based = floor(h);
    upper_zero_based = ceil(h);

    interpolation_weight = h - lower_zero_based;

    % Convert zero-based positions to MATLAB one-based indices.
    lower_indices = lower_zero_based + 1;
    upper_indices = upper_zero_based + 1;

    lower_values = reshape( ...
        x(lower_indices), ...
        size(probabilities));

    upper_values = reshape( ...
        x(upper_indices), ...
        size(probabilities));

    q = ...
        (1-interpolation_weight) .* lower_values + ...
        interpolation_weight .* upper_values;

end


function mi = mutualInformationDiscrete(x, y)
%MUTUALINFORMATIONDISCRETE Calculate MI between discrete vectors.
%
% Computes:
%
%   MI(X,Y) =
%       sum_x sum_y p(x,y) *
%       log(p(x,y)/(p(x)p(y)))
%
% Natural logarithms are used, matching sklearn.metrics'
% mutual_info_score output in nats.

    x = x(:);
    y = y(:);

    if numel(x) ~= numel(y)
        error("The two MI inputs must have the same number of values.");
    end

    if isempty(x)
        error("The MI inputs cannot be empty.");
    end

    if any(ismissing(x)) || any(ismissing(y))
        error("The MI inputs cannot contain missing values.");
    end

    % Convert arbitrary discrete values to consecutive indices.
    [~, ~, x_index] = unique(x, "sorted");
    [~, ~, y_index] = unique(y, "sorted");

    n_x = max(x_index);
    n_y = max(y_index);

    % Construct contingency table.
    contingency = accumarray( ...
        [x_index, y_index], ...
        1, ...
        [n_x, n_y], ...
        @sum, ...
        0);

    total_count = sum(contingency, "all");

    joint_probability = contingency / total_count;

    x_probability = sum(joint_probability, 2);
    y_probability = sum(joint_probability, 1);

    independent_probability = ...
        x_probability * y_probability;

    nonzero_mask = joint_probability > 0;

    mi_terms = joint_probability(nonzero_mask) .* ...
        log( ...
            joint_probability(nonzero_mask) ./ ...
            independent_probability(nonzero_mask));

    mi = sum(mi_terms);

    % Correct negligible negative values caused by floating-point
    % roundoff.
    if mi < 0 && abs(mi) < 1e-14
        mi = 0;
    end

end