%% Fisher-score QUBO feature selection for S1Dataset
%
% MATLAB translation of the Python/Iskay pipeline.
%
% Objective:
%
%   minimize x' * Q * x
%
% where:
%
%   Q(i,i) = -alpha * scaled Fisher score
%   Q(i,j) = (1-alpha) * scaled absolute covariance, i ~= j
%
% Because Q is symmetric:
%
%   x'Qx = sum_i Q(i,i)x_i ...
%          + 2*sum_{i<j}Q(i,j)x_i*x_j
%
% This is equivalent to the Iskay dictionary construction in which each
% quadratic pair was stored once with coefficient 2*Q(i,j).

clear;
clc;

%% ------------------------------------------------------------
% User settings
%% ------------------------------------------------------------

alpha = 0.86;

dataset_file = "datasets/S1Dataset.csv";

% Tabu-search settings
max_time_seconds       = 1000;
max_iterations         = 1e9;
max_stall_time_seconds = 60;
max_stall_iterations   = 1e6;

% Numerical threshold used when printing/counting QUBO terms
term_tolerance = 1e-12;

% Optional reproducibility setting.
% MATLAB tabu search is stochastic.
rng(42, "twister");

%% ------------------------------------------------------------
% Load S1Dataset
%% ------------------------------------------------------------

data = readtable( ...
    dataset_file, ...
    "VariableNamingRule", "preserve");

fprintf("Original dataset dimensions: %d rows x %d columns\n", ...
    height(data), width(data));

%% ------------------------------------------------------------
% Drop the first data row
%
% Python:
% data = data.drop(labels=0, axis=0)
%% ------------------------------------------------------------

if height(data) < 2
    error("The dataset does not contain enough rows.");
end

data(1, :) = [];

%% ------------------------------------------------------------
% Locate required columns
%% ------------------------------------------------------------

variable_names = data.Properties.VariableNames;

group_col = find(strcmp(variable_names, "Group"), 1);
vineland_col = find(strcmp(variable_names, "Vineland ABC"), 1);

if isempty(group_col)
    error('The dataset does not contain a column named "Group".');
end

if isempty(vineland_col)
    error('The dataset does not contain a column named "Vineland ABC".');
end

%% ------------------------------------------------------------
% Keep ASD and NEU data by removing SIB
%
% This follows the Python statement:
%
% data = data[data["Group"] != "SIB"]
%% ------------------------------------------------------------

group_values = strip(string(data.(variable_names{group_col})));

keep_subject = group_values ~= "SIB";

data = data(keep_subject, :);
group_values = group_values(keep_subject);

% Check whether an unexpected group remains
unexpected_groups = setdiff(unique(group_values), ["ASD", "NEU"]);

if ~isempty(unexpected_groups)
    warning( ...
        "Groups other than ASD and NEU remain after removing SIB: %s", ...
        strjoin(unexpected_groups, ", "));
end

%% ------------------------------------------------------------
% Binary labels
%
% ASD = 1
% Every non-ASD remaining observation = 0
%
% This matches:
% y = np.array([int(g == "ASD") ...])
%% ------------------------------------------------------------

y = double(group_values == "ASD");

n_asd = sum(y == 1);
n_neu = sum(y == 0);

fprintf("Subjects after removing SIB: %d\n", numel(y));
fprintf("ASD subjects: %d\n", n_asd);
fprintf("NEU subjects: %d\n", n_neu);

if n_asd < 2 || n_neu < 2
    error( ...
        "At least two subjects per group are required for sample variance.");
end

%% ------------------------------------------------------------
% Predictor columns
%
% Remove:
% - Group
% - Vineland ABC
%% ------------------------------------------------------------

predictor_columns = setdiff( ...
    1:width(data), ...
    [group_col, vineland_col], ...
    "stable");

feature_names = string(variable_names(predictor_columns));
n_features = numel(feature_names);
n_subjects = height(data);

fprintf("Predictor variables: %d\n", n_features);

%% ------------------------------------------------------------
% Convert all predictor columns to numeric
%
% This matches:
% pd.to_numeric(..., errors="coerce")
%
% Values that cannot be converted become NaN.
%% ------------------------------------------------------------

X = nan(n_subjects, n_features);

for j = 1:n_features

    table_column_index = predictor_columns(j);
    current_column = data.(variable_names{table_column_index});

    if isnumeric(current_column) || islogical(current_column)

        X(:, j) = double(current_column);

    elseif iscell(current_column)

        for r = 1:n_subjects

            value = current_column{r};

            if isnumeric(value) && isscalar(value)

                X(r, j) = double(value);

            elseif islogical(value) && isscalar(value)

                X(r, j) = double(value);

            elseif isempty(value)

                X(r, j) = NaN;

            else

                X(r, j) = str2double(strip(string(value)));

            end
        end

    else

        % Handles string, categorical, character, and similar columns
        X(:, j) = str2double(strip(string(current_column)));

    end
end

%% ------------------------------------------------------------
% Median imputation
%
% Python:
% X_df = X_df.fillna(X_df.median(numeric_only=True))
%% ------------------------------------------------------------

missing_before = sum(isnan(X), 1);

for j = 1:n_features

    column_median = median(X(:, j), "omitnan");

    if isnan(column_median)
        error( ...
            'Feature "%s" contains no valid numeric values.', ...
            char(feature_names(j)));
    end

    missing_mask = isnan(X(:, j));
    X(missing_mask, j) = column_median;
end

fprintf("Total missing values imputed: %d\n", sum(missing_before));

if any(isnan(X), "all")
    error("Missing values remain after median imputation.");
end

%% ------------------------------------------------------------
% Z-score normalization
%
% Python StandardScaler uses population standard deviation:
%
% sigma = sqrt(sum((x-mu).^2) / N)
%
% MATLAB std(X,1,1) also uses normalization by N.
%
% Constant features are assigned a scale of 1, matching the practical
% behavior of StandardScaler, so their standardized values become zero.
%% ------------------------------------------------------------

global_mean = mean(X, 1);
global_std = std(X, 1, 1);

constant_feature_mask = ...
    (global_std == 0) | ...
    isnan(global_std) | ...
    isinf(global_std);

if any(constant_feature_mask)
    fprintf( ...
        "Constant predictor columns assigned unit scale: %d\n", ...
        sum(constant_feature_mask));

    global_std(constant_feature_mask) = 1;
end

Xz = (X - global_mean) ./ global_std;

%% ------------------------------------------------------------
% Fisher score
%
% F_i = (mu_ASD - mu_NEU)^2 / (var_ASD + var_NEU)
%
% Python uses:
% np.var(..., ddof=1)
%
% MATLAB var(...,0,1) uses sample variance with N-1 denominator.
%% ------------------------------------------------------------

X_asd = Xz(y == 1, :);
X_neu = Xz(y == 0, :);

mu_asd = mean(X_asd, 1);
mu_neu = mean(X_neu, 1);

var_asd = var(X_asd, 0, 1);
var_neu = var(X_neu, 0, 1);

eps_value = 1e-12;

F = (mu_asd - mu_neu).^2 ./ ...
    (var_asd + var_neu + eps_value);

%% ------------------------------------------------------------
% Pairwise covariance
%
% np.cov(Xz, rowvar=False) uses sample covariance.
% MATLAB cov(Xz) also uses sample covariance.
%% ------------------------------------------------------------

C = cov(Xz);

% Absolute covariance as redundancy strength
C = abs(C);

% Remove diagonal
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
% Build the symmetric QUBO matrix
%
% diagonal:
%   -alpha * Fisher score
%
% off-diagonal:
%   (1-alpha) * covariance redundancy
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

% Explicit symmetry check
symmetry_error = max(abs(Q - Q'), [], "all");

if symmetry_error > 1e-12
    error("Q is not symmetric.");
end

%% ------------------------------------------------------------
% Count equivalent Iskay terms
%
% hubo contained:
% - one constant term: ()
% - nonzero diagonal/linear terms
% - nonzero upper-triangle quadratic terms
%% ------------------------------------------------------------

linear_term_count = sum(abs(diag(Q)) > term_tolerance);

upper_Q = triu(Q, 1);

quadratic_term_count = sum( ...
    abs(2 * upper_Q) > term_tolerance, ...
    "all");

total_term_count = ...
    1 + linear_term_count + quadratic_term_count;

fprintf("\nVariables: %d | Terms: %d\n", ...
    n_features, total_term_count);

%% ------------------------------------------------------------
% Print scaled Fisher scores
%
% Printed indices are zero-based to match the Python output.
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
% Print the equivalent Iskay QUBO dictionary
%
% MATLAB itself does not require this dictionary.
% This output is included so it can be compared directly with Python.
%% ------------------------------------------------------------

fprintf("\nEquivalent QUBO terms:\n");

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

        % Iskay stores a pair once, so its coefficient is 2*Q(i,j)
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
% Printed indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nIndex to feature mapping:\n");

for i = 1:n_features
    fprintf( ...
        "%d -> %s\n", ...
        i-1, ...
        char(feature_names(i)));
end

%% ------------------------------------------------------------
% Display alpha
%% ------------------------------------------------------------

fprintf("\nAlpha: %.6f\n", alpha);

%% ------------------------------------------------------------
% Create MATLAB QUBO problem
%
% MATLAB minimizes:
%
%   x'Qx
%
% The diagonal entries act as linear terms because x_i^2 = x_i
% for binary variables.
%% ------------------------------------------------------------

qprob = qubo(Q);

%% ------------------------------------------------------------
% Configure MATLAB tabu search
%% ------------------------------------------------------------

ts = tabuSearch( ...
    MaxTime             = max_time_seconds, ...
    MaxIterations       = max_iterations, ...
    MaxStallTime        = max_stall_time_seconds, ...
    MaxStallIterations  = max_stall_iterations, ...
    Display             = "final");

%% ------------------------------------------------------------
% Solve and record QUBO runtime
%
% The timer starts immediately before solve and ends immediately after.
% Therefore, it excludes dataset loading, preprocessing, and printing.
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

% Independent objective check
manual_objective_value = x_best' * Q * x_best;

%% ------------------------------------------------------------
% Raw result
%% ------------------------------------------------------------

fprintf("\nRaw QUBO result:\n");
disp(result);

fprintf("\nRaw tabu-search result:\n");
disp(result.AlgorithmResult);

%% ------------------------------------------------------------
% Print final results
%% ------------------------------------------------------------

fprintf("\nVariables: %d\n", n_features);
fprintf("Terms: %d\n", total_term_count);
fprintf("Selected feature count: %d\n", ...
    numel(selected_matlab_indices));

fprintf("Best QUBO objective value: %.10f\n", best_value);
fprintf("Manual x''Qx objective check: %.10f\n", ...
    manual_objective_value);

fprintf( ...
    "Objective difference: %.3e\n", ...
    abs(best_value - manual_objective_value));

fprintf( ...
    "MATLAB QUBO solve runtime: %.6f seconds\n", ...
    qubo_runtime_seconds);

%% ------------------------------------------------------------
% Selected features
%
% Printed indices are zero-based to match the Python result.
%% ------------------------------------------------------------

fprintf("\nSelected features from mapped solution:\n");

if isempty(selected_matlab_indices)

    fprintf("No features selected.\n");

else

    for k = 1:numel(selected_matlab_indices)

        matlab_index = selected_matlab_indices(k);
        python_index = matlab_index - 1;

        fprintf( ...
            "%d -> %s\n", ...
            python_index, ...
            char(feature_names(matlab_index)));
    end
end