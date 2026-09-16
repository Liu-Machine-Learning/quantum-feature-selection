%% Mutual-information QUBO feature selection for S1Dataset
%
% MATLAB translation of the Python MI-QUBO pipeline.
%
% Relevance:
%   I_i = MI(feature_i, class label)
%
% Redundancy:
%   R_ij = MI(feature_i, feature_j)
%
% QUBO:
%   diagonal:    Q(i,i) = -alpha * I_i
%   offdiagonal: Q(i,j) = (1-alpha) * R_ij
%
% Objective:
%   minimize x' * Q * x
%
% Since Q is symmetric:
%
%   x'Qx = sum_i Q(i,i)x_i ...
%          + 2*sum_{i<j}Q(i,j)x_i*x_j
%
% The full symmetric Q matrix is passed directly to MATLAB's qubo object.

clear;
clc;

%% ------------------------------------------------------------
% User settings
%% ------------------------------------------------------------

alpha = 0.975;

% Number of quantile bins
B = 20;

dataset_file = "datasets/S1Dataset.csv";

% Tabu-search settings
max_time_seconds       = 300;
max_iterations         = 1e9;
max_stall_time_seconds = 60;
max_stall_iterations   = 1e6;

% Numerical threshold for printing/counting QUBO terms
term_tolerance = 1e-12;

% Reproducibility for stochastic tabu search
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
% Remove SIB subjects
%
% Python:
% data = data[data["Group"] != "SIB"].copy()
%% ------------------------------------------------------------

group_values = strip(string(data.(variable_names{group_col})));

% Prevent missing strings from causing logical-indexing problems
group_values(ismissing(group_values)) = "";

keep_subject = group_values ~= "SIB";

data = data(keep_subject, :);
group_values = group_values(keep_subject);

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
% NEU = 0
%
% This follows:
% y = int(group == "ASD")
%% ------------------------------------------------------------

y = double(group_values == "ASD");

n_asd = sum(y == 1);
n_neu = sum(y == 0);

fprintf("Subjects after removing SIB: %d\n", numel(y));
fprintf("ASD subjects: %d\n", n_asd);
fprintf("NEU subjects: %d\n", n_neu);

if n_asd == 0 || n_neu == 0
    error("Both ASD and NEU subjects are required.");
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
% Convert predictors to numeric
%
% Python:
% pd.to_numeric(column, errors="coerce")
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

        % Handles string, categorical, and character columns
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

fprintf("Total missing values imputed: %d\n", ...
    sum(missing_before));

if any(isnan(X), "all")
    error("Missing values remain after median imputation.");
end

%% ------------------------------------------------------------
% Quantile discretization
%
% Python:
%
% edges = np.quantile(
%     x,
%     np.linspace(0.0, 1.0, B + 1)
% )
%
% edges = np.unique(edges)
%
% return np.digitize(
%     x,
%     edges(2:end-1),
%     right=True
% )
%
% The helper function quantileBinLikeNumPy is defined at the
% end of this script.
%% ------------------------------------------------------------

X_binned = zeros(n_subjects, n_features);
effective_bin_counts = zeros(1, n_features);

for j = 1:n_features

    [X_binned(:, j), effective_bin_counts(j)] = ...
        quantileBinLikeNumPy(X(:, j), B);

end

fprintf("\nRequested quantile bins: %d\n", B);

%% ------------------------------------------------------------
% Importance vector
%
% I_i = MI(feature_i, y)
%
% Equivalent to:
% mutual_info_score(X_binned(:,i), y)
%
% Natural logarithms are used, so MI is measured in nats.
%% ------------------------------------------------------------

I = zeros(n_features, 1);

for i = 1:n_features

    I(i) = mutualInformationDiscrete( ...
        X_binned(:, i), ...
        y);

end

%% ------------------------------------------------------------
% Redundancy matrix
%
% R_ij = MI(feature_i, feature_j)
%
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
% Build QUBO matrix
%
% Q_ij(alpha) =
%     R_ij - alpha*(R_ij + delta_ij*I_i)
%
% Since R_ii = 0:
%
% diagonal:
%     Q_ii = -alpha*I_i
%
% off-diagonal:
%     Q_ij = (1-alpha)*R_ij
%
% No separate maximum scaling is performed because the supplied
% Python MI implementation does not scale I or R.
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

if symmetry_error > 1e-12
    error("The QUBO matrix is not symmetric.");
end

%% ------------------------------------------------------------
% Count equivalent Iskay terms
%
% The Python dictionary contains:
% - one constant term: ()
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

fprintf("\nVariables: %d | Terms: %d\n", ...
    n_features, total_term_count);

%% ------------------------------------------------------------
% Print MI relevance scores
%
% Indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nMutual-information relevance scores:\n");

for i = 1:n_features

    fprintf( ...
        "%2d. %s | MI = %.10f | Effective bins = %d\n", ...
        i-1, ...
        char(feature_names(i)), ...
        I(i), ...
        effective_bin_counts(i));

end

%% ------------------------------------------------------------
% Optional summary of redundancy values
%% ------------------------------------------------------------

nonzero_redundancy = R(triu(true(n_features), 1));

fprintf("\nRedundancy MI summary:\n");
fprintf("Minimum pairwise MI: %.10f\n", ...
    min(nonzero_redundancy));

fprintf("Mean pairwise MI: %.10f\n", ...
    mean(nonzero_redundancy));

fprintf("Maximum pairwise MI: %.10f\n", ...
    max(nonzero_redundancy));

%% ------------------------------------------------------------
% Print equivalent Iskay QUBO terms
%
% MATLAB does not need this dictionary. It is printed so that the
% coefficients can be compared with the Python implementation.
%% ------------------------------------------------------------

fprintf("\nEquivalent QUBO terms:\n");

% Constant
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

        % Python/Iskay stores each pair once, so its dictionary
        % coefficient is 2*Q(i,j).
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
% Printed indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nIndex to feature mapping:\n");

for i = 1:n_features

    fprintf( ...
        "%d -> %s\n", ...
        i-1, ...
        char(feature_names(i)));

end

fprintf("\nAlpha: %.6f\n", alpha);
fprintf("Quantile bins B: %d\n", B);

%% ------------------------------------------------------------
% Create MATLAB QUBO problem
%% ------------------------------------------------------------

qprob = qubo(Q);

%% ------------------------------------------------------------
% Configure MATLAB tabu search
%% ------------------------------------------------------------

ts = tabuSearch( ...
    MaxTime            = max_time_seconds, ...
    MaxIterations      = max_iterations, ...
    MaxStallTime       = max_stall_time_seconds, ...
    MaxStallIterations = max_stall_iterations, ...
    Display            = "final");

%% ------------------------------------------------------------
% Solve QUBO and record solver runtime
%
% Only the solve call is timed. Loading, preprocessing,
% discretization, MI calculation, and printing are excluded.
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
% Raw solver output
%% ------------------------------------------------------------

fprintf("\nRaw QUBO result:\n");
disp(result);

fprintf("\nRaw tabu-search result:\n");
disp(result.AlgorithmResult);

%% ------------------------------------------------------------
% Final summary
%% ------------------------------------------------------------

fprintf("\nVariables: %d\n", n_features);
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
% Selected features
%
% Output indices are zero-based to match Python.
%% ------------------------------------------------------------

fprintf("\nSelected features from mapped solution:\n");

if isempty(selected_matlab_indices)

    fprintf("No features selected.\n");

else

    for k = 1:numel(selected_matlab_indices)

        matlab_index = selected_matlab_indices(k);
        python_index = matlab_index - 1;

        fprintf( ...
            "%d -> %s | Relevance MI = %.10f\n", ...
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
%QUANTILEBINLIKENUMPY Quantile discretization matching Python.
%
% Reproduces:
%
% edges = np.quantile(
%     x,
%     np.linspace(0,1,n_bins+1)
% )
%
% edges = np.unique(edges)
%
% if len(edges) <= 2:
%     return zeros(...)
%
% bins = np.digitize(
%     x,
%     edges(2:end-1),
%     right=True
% )
%
% NumPy's default quantile method is linear interpolation using:
%
% h = (N-1)*p

    x = double(x(:));

    if any(isnan(x))
        error("quantileBinLikeNumPy received missing values.");
    end

    probabilities = linspace(0, 1, n_bins + 1);

    edges = numpyLinearQuantiles( ...
        x, probabilities);

    % np.unique sorts the edges. Quantile edges are already ordered.
    edges = unique(edges);

    if numel(edges) <= 2

        bins = zeros(numel(x), 1);
        effective_bin_count = 1;
        return;

    end

    interior_edges = edges(2:end-1);
    interior_edges = interior_edges(:)';

    % np.digitize(x, interior_edges, right=True)
    %
    % With right=True, a value equal to an edge remains in the
    % lower bin. Therefore, the bin number equals the number of
    % interior edges that are strictly less than x.
    bins = sum(x > interior_edges, 2);

    bins = double(bins);

    effective_bin_count = numel(unique(bins));

end


function q = numpyLinearQuantiles(x, probabilities)
%NUMPYLINEARQUANTILES Reproduce np.quantile method="linear".
%
% For sorted observations x:
%
% h = (N-1)*p
%
% The quantile is linearly interpolated between:
% floor(h) and ceil(h).

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

    weights = h - lower_zero_based;

    % Convert zero-based positions to MATLAB one-based indices
    lower_indices = lower_zero_based + 1;
    upper_indices = upper_zero_based + 1;

    lower_values = reshape( ...
        x(lower_indices), ...
        size(probabilities));

    upper_values = reshape( ...
        x(upper_indices), ...
        size(probabilities));

    q = ...
        (1-weights) .* lower_values + ...
        weights .* upper_values;

end


function mi = mutualInformationDiscrete(x, y)
%MUTUALINFORMATIONDISCRETE Mutual information of discrete vectors.
%
% Computes:
%
% MI(X,Y) = sum_x sum_y p(x,y) *
%           log[p(x,y)/(p(x)p(y))]
%
% Natural logarithms are used, matching sklearn's
% mutual_info_score output in nats.

    x = x(:);
    y = y(:);

    if numel(x) ~= numel(y)
        error("The two MI inputs must have equal lengths.");
    end

    if isempty(x)
        error("The MI inputs cannot be empty.");
    end

    if any(ismissing(x)) || any(ismissing(y))
        error("The MI inputs cannot contain missing values.");
    end

    % Convert arbitrary discrete values to consecutive indices
    [~, ~, x_index] = unique(x, "sorted");
    [~, ~, y_index] = unique(y, "sorted");

    n_x = max(x_index);
    n_y = max(y_index);

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

    % Remove very small negative values caused only by floating-point
    % roundoff.
    if mi < 0 && abs(mi) < 1e-14
        mi = 0;
    end

end