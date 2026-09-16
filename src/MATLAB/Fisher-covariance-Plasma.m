%% Fisher-score QUBO feature selection for plasma dataset
%
% Dataset format:
% - First row: subject labels in columns 2:end
% - First column: metabolite names in rows 2:end
% - Rows: metabolite features
% - Columns: subjects
%
% Preprocessing matches the supplied Python implementation:
% - Convert measurements to numeric
% - Remove feature rows containing any missing values
% - Keep only ASD and TD subjects
% - Keep the first n_keep valid features
% - Transpose to subjects-by-features format
% - Z-score using population standard deviation
%
% QUBO objective:
%
%   minimize x' * Q * x
%
% where:
%
%   Q(i,i) = -alpha * scaled Fisher score
%   Q(i,j) = (1-alpha) * scaled absolute covariance
%
% Because Q is symmetric:
%
%   x'Qx = sum_i Q(i,i)x_i
%          + 2*sum_{i<j}Q(i,j)x_i*x_j

clear;
clc;

%% ------------------------------------------------------------
% User settings
%% ------------------------------------------------------------

alpha = 0.84;

% Number of valid plasma features to retain
n_keep = 56;

dataset_file = "datasets/plasma_prescreen_56.csv";

% Tabu-search settings
max_time_seconds       = 1000;
max_iterations         = 1e9;
max_stall_time_seconds = 60;
max_stall_iterations   = 1e3;

% Numerical threshold for counting and printing QUBO terms
term_tolerance = 1e-12;

% Reproducibility for stochastic tabu search
rng(42, "twister");

%% ------------------------------------------------------------
% Load plasma dataset
%% ------------------------------------------------------------

raw = readcell(dataset_file);

if size(raw, 1) < 2 || size(raw, 2) < 2
    error("The plasma dataset does not contain sufficient data.");
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

% Replace missing strings with empty strings
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
% Extract raw feature-by-subject measurement matrix
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
% Python:
% pd.to_numeric(..., errors="coerce")
%
% Values that cannot be converted are assigned NaN.
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
% Python:
% valid_feature_mask =
%     ~X_feature_by_subject.isna().any(axis=1)
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
% Keep only ASD and TD subjects
%
% Python:
% keep_subject_mask = (labels == "ASD") | (labels == "TD")
%% ------------------------------------------------------------

keep_subject_mask = ...
    (labels == "ASD") | ...
    (labels == "TD");

labels = labels(keep_subject_mask);
Xnum = Xnum(:, keep_subject_mask);

if isempty(labels)
    error("No ASD or TD subjects were found.");
end

%% ------------------------------------------------------------
% Binary labels
%
% ASD = 1
% TD  = 0
%% ------------------------------------------------------------

y = double(labels == "ASD");

n_asd = sum(y == 1);
n_td = sum(y == 0);

fprintf("Subjects retained: %d\n", numel(y));
fprintf("ASD subjects: %d\n", n_asd);
fprintf("TD subjects: %d\n", n_td);

if n_asd < 2 || n_td < 2
    error( ...
        "At least two ASD and two TD subjects are required for Fisher scores.");
end

%% ------------------------------------------------------------
% Keep first n_keep features
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
%   sqrt(sum((x-mu).^2)/N)
%
% MATLAB std(X,1,1) also uses normalization by N.
%% ------------------------------------------------------------

feature_mean = mean(X, 1);
feature_std = std(X, 1, 1);

% StandardScaler assigns an effective scale of 1 to constant columns.
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

%% ------------------------------------------------------------
% Fisher score
%
% F_i = (mu_ASD - mu_TD)^2 / (var_ASD + var_TD)
%
% Python uses np.var(..., ddof=1).
% MATLAB var(...,0,1) uses the N-1 denominator.
%% ------------------------------------------------------------

X_asd = Xz(y == 1, :);
X_td = Xz(y == 0, :);

mu_asd = mean(X_asd, 1);
mu_td = mean(X_td, 1);

var_asd = var(X_asd, 0, 1);
var_td = var(X_td, 0, 1);

eps_value = 1e-12;

F = (mu_asd - mu_td).^2 ./ ...
    (var_asd + var_td + eps_value);

%% ------------------------------------------------------------
% Pairwise covariance on z-scored data
%
% Python:
% C = np.cov(Xz, rowvar=False)
%% ------------------------------------------------------------

C = cov(Xz);

% Absolute covariance represents redundancy strength
C = abs(C);

% Set diagonal to zero
C(1:n_features+1:end) = 0;

%% ------------------------------------------------------------
% Scale Fisher score and covariance
%% ------------------------------------------------------------

F_max = max(F);

if F_max > 0
    F_scaled = F / F_max;
else
    F_scaled = F;
end

C_max = max(C(:));

if C_max > 0
    C_scaled = C / C_max;
else
    C_scaled = C;
end

%% ------------------------------------------------------------
% Build symmetric QUBO matrix
%
% Diagonal:
%   -alpha * scaled Fisher score
%
% Off-diagonal:
%   (1-alpha) * scaled absolute covariance
%% ------------------------------------------------------------

Q = zeros(n_features, n_features);

for i = 1:n_features
    Q(i, i) = -alpha * F_scaled(i);
end

for i = 1:n_features

    for j = i+1:n_features

        qij = (1-alpha) * C_scaled(i, j);

        Q(i, j) = qij;
        Q(j, i) = qij;

    end
end

%% ------------------------------------------------------------
% Confirm Q is symmetric
%% ------------------------------------------------------------

symmetry_error = max(abs(Q - Q'), [], "all");

if symmetry_error > 1e-12
    error("The QUBO matrix is not symmetric.");
end

%% ------------------------------------------------------------
% Count equivalent Iskay QUBO terms
%
% The Python hubo dictionary contains:
% - one constant term
% - nonzero diagonal terms
% - nonzero upper-triangular quadratic terms
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
% Print scaled Fisher scores
%
% Indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nScaled Fisher scores:\n");

for i = 1:n_features

    fprintf( ...
        "%2d. %s | Fisher = %.6f\n", ...
        i-1, ...
        char(feature_names(i)), ...
        F_scaled(i));

end

%% ------------------------------------------------------------
% Print equivalent Iskay QUBO terms
%
% MATLAB does not need this dictionary. It is printed to allow
% direct comparison with the Python coefficients.
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
        % twice the corresponding symmetric matrix entry.
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
% Feature mapping
%
% Indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nIndex to metabolite mapping:\n");

for i = 1:n_features

    fprintf( ...
        "%d -> %s\n", ...
        i-1, ...
        char(feature_names(i)));

end

fprintf("\nAlpha: %.6f\n", alpha);

%% ------------------------------------------------------------
% Create MATLAB QUBO problem
%
% MATLAB minimizes:
%
%   x'Qx
%
% The full symmetric matrix is supplied directly.
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
% Solve and record QUBO runtime
%
% The timer measures only the solve call. Dataset loading,
% preprocessing, QUBO construction, and printing are excluded.
%% ------------------------------------------------------------

fprintf("\nStarting MATLAB tabu-search QUBO solver...\n");

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

fprintf("MATLAB QUBO solve runtime: %.6f seconds\n", ...
    qubo_runtime_seconds);

%% ------------------------------------------------------------
% Print selected metabolites
%
% Output indices are zero-based to match Python.
% The complete binary feature mask is intentionally not printed.
%% ------------------------------------------------------------

fprintf("\nSelected features from mapped solution:\n");

if isempty(selected_matlab_indices)

    fprintf("No features selected.\n");

else

    for k = 1:numel(selected_matlab_indices)

        matlab_index = selected_matlab_indices(k);
        python_index = matlab_index - 1;

        fprintf( ...
            "%d -> %s | Scaled Fisher = %.6f\n", ...
            python_index, ...
            char(feature_names(matlab_index)), ...
            F_scaled(matlab_index));

    end
end