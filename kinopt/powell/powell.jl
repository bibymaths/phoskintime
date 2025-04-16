#List of required packages
required_packages = [
    "LinearAlgebra",
    "DataFrames",
    "CSV",
    "StatsBase",
    "Random",
    "Optimization",
    "OptimizationPRIMA",
    "Plots",
    "Distributions",
    "Statistics",
    "XLSX",
    "DataStructures",
    "SparseArrays",
    "ColorSchemes",
    "ArgParse"
]
#= import Pkg
# Install missing packages
for pkg in required_packages
    if !haskey(Pkg.dependencies(), pkg)
        println("Installing missing package: $pkg")
        Pkg.add(pkg)
    end
end =#

for pkg in required_packages
    @eval using $(Symbol(pkg))
end


function parse_command_line_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--beta_lower"
            help = "Lower bound for beta values"
            arg_type = Float64
            default = -4
        "--beta_upper"
            help = "Upper bound for beta values"
            arg_type = Float64
            default = 4
        "--estimate_missing_kinases"
            help = "Enable estimation of missing kinases (true/false)"
            action = :store_true
        "--constraint_type"
            help = "Constraint type: linear or nonlinear"
            arg_type = String
            default = "nonlinear"
        "--verbosity"
            help = "Verbosity level for optimizer output (0: NONE, 1: EXIT, 2: RHO, 3: FEVL)"
            arg_type = Int
            default = 2
    end
    return parse_args(s)
end

# Parse command-line arguments
args = parse_command_line_args()

# Import the packages
using LinearAlgebra
using DataFrames
using CSV
using StatsBase
using Random
using Optimization
using OptimizationPRIMA
using Distributions
using Statistics
using XLSX
using DataStructures
using SparseArrays
using Base.Threads
using ColorSchemes
using Plots

# Configure GR for headless, high-quality operation
gr()
ENV["GKSwstype"] = "100"  # Disable interactive windows
ENV["GR_DPI"] = 1200       # Set high resolution

# Extract arguments
beta_lower = args["beta_lower"]
beta_upper = args["beta_upper"]
estimate_missing_kinases = args["estimate_missing_kinases"]
constraint_type = args["constraint_type"]
verbosity_map = Dict(0 => PRIMA.MSG_NONE, 1 => PRIMA.MSG_EXIT, 2 => PRIMA.MSG_RHO, 3 => PRIMA.MSG_FEVL)
verbosity_level = get(verbosity_map, args["verbosity"], PRIMA.MSG_NONE)

# ------------------------------------------------------------------------------
# 1) Directory
# ------------------------------------------------------------------------------
# Ensure Directory Exists
function ensure_directory_exists(dir::String)
    """
    Ensures that the specified directory exists. If the directory does not exist,
    it creates the directory.

    # Arguments
    - `dir::String`: The path of the directory to check or create.
    """

    if !isdir(dir)
        mkdir(dir)
    end
end

# ------------------------------------------------------------------------------
# 2) Data loading
# ------------------------------------------------------------------------------
function load_data(input1_path::String, input2_path::String, estimate_missing_kinases::Bool)
    """
    Loads and preprocesses data from two CSV files.

    # Arguments
    - `input1_path::String`: Path to the full HGNC data CSV file.
    - `input2_path::String`: Path to the interaction data CSV file.
    - `estimate_missing_kinases::Bool`: Whether to estimate missing kinases.

    # Returns
    - `(full_hgnc_df::DataFrame, interaction_df::DataFrame)`: A tuple of the processed DataFrames.
    """
    # Load the CSV files into DataFrames
    full_hgnc_df = CSV.read(input1_path, DataFrame)
    filter!(row -> !(ismissing(row.Psite) || row.Psite == ""), full_hgnc_df)

    interaction_df = CSV.read(input2_path, DataFrame)
    if estimate_missing_kinases
        # Keep the `Kinase` column as-is
        # Additional logic can be added here if processing is needed without splitting/stripping
    else
        # Create a set of valid GeneIDs from the full_hgnc_df
        valid_kinases = Set(full_hgnc_df.GeneID[2:end])  # Skip the first entry if it's a header or placeholder
        # Filter rows in interaction_df where all kinases in the 'Kinase' column are valid
        interaction_df = filter(row -> begin
            # Strip curly brackets and split the 'Kinase' column by commas
            kinases = split(strip(row.Kinase, ['{', '}']), ",")
            # Check if all kinases are in the set of valid GeneIDs
            all(kinase -> strip(kinase) in valid_kinases, kinases)
        end, interaction_df)
        # Process the 'Kinase' column to remove curly brackets, split by commas, and clean up spacing
        interaction_df.Kinase .= [
            "{" * join([strip(kinase) for kinase in split(strip(k, ['{', '}']), ",")], ",") * "}"
            for k in interaction_df.Kinase
        ]
    end
    return full_hgnc_df, interaction_df
end

# ------------------------------------------------------------------------------
# 3) Prepare initial arrays for P
# ------------------------------------------------------------------------------
function prepare_initial_arrays(full_hgnc_df::DataFrame, interaction_df::DataFrame, time_cols::Vector{String})
    """
    Prepares the initial arrays for P by processing the DataFrames based
    on specified time columns.

    # Arguments
    - `full_hgnc_df::DataFrame`: DataFrame containing time series.
    - `interaction_df::DataFrame`: DataFrame containing interaction.
    - `time_cols::Vector{String}`: Vector of column names representing time points.

    # Returns
    - `P_initial::Dict{Tuple{String, String}, Dict{String, Any}}`: Dictionary mapping
      `(gene, psite)` tuples to their corresponding kinases and time series data.
    - `P_initial_array::Matrix{Float64}`: Matrix where each row represents the time
      series data for a `(gene, psite)` pair.
    """
    # Number of time points
    t_max = length(time_cols)
    # Number of (gene, psite) pairs
    n_pairs = nrow(interaction_df)
    # Pre-allocate P_initial_array
    P_initial_array = zeros(n_pairs, t_max)
    # Initialize P_initial as an empty dictionary
    P_initial = Dict()
    for (idx, row) in enumerate(eachrow(interaction_df))
        gene, psite, kinases = row.GeneID, row.Psite, split(strip(row.Kinase, ['{', '}']), ",")
        observed_data = filter(r -> r.GeneID == gene && r.Psite == psite, full_hgnc_df)
        # Extract the time series or assign default ones
        time_series = isempty(observed_data) ? ones(t_max) : Array(observed_data[:, time_cols][1, :])
        # Populate P_initial_array
        P_initial_array[idx, :] .= time_series
        # Populate P_initial dictionary
        P_initial[(gene, psite)] = Dict("Kinases" => kinases, "TimeSeries" => time_series)
    end
    return P_initial, P_initial_array
end

# ------------------------------------------------------------------------------
# 4) Prepare kinase data
# ------------------------------------------------------------------------------
function prepare_kinase_data(
    full_hgnc_df::DataFrame,
    interaction_df::DataFrame,
    time_cols::Vector{String},
    estimate_missing_kinases::Bool
)
    """
    Prepares kinase-related data, including indexing, array preparation, and beta
    counts. Handles missing kinases based on user settings.

    # Arguments
    - `full_hgnc_df::DataFrame`: DataFrame containing time series.
    - `interaction_df::DataFrame`: DataFrame containing interaction.
    - `time_cols::Vector{String}`: Vector of column names representing time points.
    - `estimate_missing_kinases::Bool`: Flag indicating whether to estimate missing
      kinase-psite combinations.

    # Returns
    - `K_index::Dict{String, Vector{Float64}}`: Dictionary mapping "Kinase|Psite"
      keys to their corresponding time series vectors.
    - `K_array::Matrix{Float64}`: Matrix where each column represents a kinase-psite
      time series.
    - `beta_counts::Dict{String, Int}`: Dictionary tracking the count of beta values
      for each kinase-psite pair.
    """
    K_index = Dict{String,Vector{Float64}}()
    K_array = []
    beta_counts = Dict{String,Int64}()

    # Build a set of all (kinase|psite) combos we expect from interaction_df
    expected_pairs = Set{String}()
    for row in eachrow(interaction_df)
        kinases = split(strip(row.Kinase, ['{', '}']), ",")
        for k in kinases
            combined_key = "$(strip(k))|$(row.Psite)"
            push!(expected_pairs, combined_key)
        end
    end

    # Process each unique kinase
    unique_kinases = unique(vcat([split(strip(row.Kinase, ['{', '}']), ",") for row in eachrow(interaction_df)]...))

    # Define the dictionary mapping kinases to their respective additional beta counts
    kinase_to_psites = Dict(
        "CDK5" => 1,
        "TTK" => 7,
        "GSK3B" => 4,
        "MAP2K4" => 4,
        "MAP2K1" => 2,
        "MAP2K3" => 1,
        "CDK4" => 2
    )

    # Initialize synthetic label counters for each kinase
    synthetic_psite_counters = Dict{String, Int}()
    for kinase in keys(kinase_to_psites)
        synthetic_psite_counters[kinase] = 1  # Start counter for specified kinases
    end

    # Loop through existing kinases and their data
    for kinase in unique_kinases
        kinase_psite_data = filter(row -> row.GeneID == kinase, full_hgnc_df)

        if !isempty(kinase_psite_data)
            for row in eachrow(kinase_psite_data)
                psite = row.Psite
                # Extract the time series as an array
                time_series = Array(row[time_cols])
                idx = length(K_array) + 1
                push!(K_array, time_series)
                combined_key = "$(kinase)|$(psite)"
                K_index[combined_key] = time_series
                # Add default beta count (1 beta per psite)
                beta_counts[combined_key] = 1
            end
        end
    end

    # Handle missing kinases if required
    if estimate_missing_kinases
        for pair in expected_pairs
            if !haskey(K_index, pair)
                println("No data avialable; inserting ones vector for $pair.")
                # Extract the kinase from the pair
                kinase, psite = split(pair, "|")
                # Only add synthetic psites for kinases in the dictionary
                if haskey(kinase_to_psites, kinase)
                    # Number of psites expected for this kinase
                    count = kinase_to_psites[kinase]
                    # Generate synthetic psites only up to the specified count
                    existing_psites = [
                        split(key, "|")[2] for key in keys(K_index) if startswith(key, "$(kinase)|")
                    ]
                    num_existing_psites = length(existing_psites)
                    for i in 1:(count - num_existing_psites)
                        # Generate synthetic psite label
                        synthetic_psite = "P$(num_existing_psites + i)"
                        synthetic_key = "$(kinase)|$(synthetic_psite)"
                        if !haskey(K_index, synthetic_key)
                            K_index[synthetic_key] = ones(length(time_cols))  # Placeholder zero vector
                            push!(K_array, K_index[synthetic_key])
                            # Assign a beta count for the synthetic psite
                            beta_counts[synthetic_key] = 1  # One beta per synthetic psite
                        end
                    end
                end
            end
        end
    end
    # Ensure `beta_counts` is of the correct type
    beta_counts = Dict{String, Int64}(beta_counts)
    return K_index, hcat(K_array...), beta_counts
end

# ------------------------------------------------------------------------------
# 5) Objective-related functions
# ------------------------------------------------------------------------------
function calculate_estimated_series(
    params::Vector{Float64},
    P_initial::Dict,
    K_index::Dict,
    gene_psite_counts::Vector{Int},
    beta_counts::Dict{String,Int},
    time_cols::Vector{String}
)
    """
    Calculates the estimated phosphorylation series based on the provided parameters,
    initial P data, and kinase index.

    # Arguments
    - `params::Vector{Float64}`: Vector of parameters containing alpha and beta values.
    - `P_initial::Dict{Tuple{String, String}, Dict{String, Any}}`: Initial P data mapping
      genes and psites to kinases and time series.
    - `K_index::Dict{String, Vector{Float64}}`: Index mapping kinase-psite pairs to
      their time series vectors.
    - `gene_psite_counts::Vector{Int}`: Vector indicating the number of kinases per
      gene-psite pair.
    - `beta_counts::Dict{String, Int}`: Dictionary tracking beta counts for each
      kinase-psite pair.
    - `time_cols::Vector{String}`: Vector of column names representing time points.

    # Returns
    - `P_estimated::Matrix{Float64}`: Matrix of estimated phosphorylation levels for
      each gene-psite pair across time points.
    """
    # Reconstruct alpha and beta
    alpha = Dict{Tuple{Any, Any}, Vector{Float64}}()
    beta = Dict{String, Float64}()

    alpha_start = 1
    for ((gene, psite), count) in zip(keys(P_initial), gene_psite_counts)
        alpha[(gene, psite)] = params[alpha_start:alpha_start+count-1]
        alpha_start += count
    end

    beta_start = sum(gene_psite_counts) + 1
    for (kpair, count) in beta_counts
        beta[kpair] = params[beta_start]
        beta_start += 1
    end

    t_max = length(time_cols)
    n_pairs = length(P_initial)
    P_estimated = zeros(n_pairs, t_max)

    @threads for i in 1:n_pairs
        gene, psite = collect(keys(P_initial))[i]
        kinases = P_initial[(gene, psite)]["Kinases"]
        prediction = zeros(t_max)

        for (j, kinase) in enumerate(kinases)
            alpha_j = alpha[(gene, psite)][j]

            # Construct combined keys and check
            for combined_key in keys(K_index)
                if startswith(combined_key, "$kinase|")
                    _, psiteK = split(combined_key, "|", limit=2)
                    if haskey(beta, combined_key)
                        prediction .+= alpha_j * beta[combined_key] .* K_index[combined_key]
                    end
                end
            end
        end

        P_estimated[i, :] .= prediction
    end

    return P_estimated
end

function objective_function(params::Vector{Float64}, problem_data::Dict)
    """
    Computes the objective function value, typically the mean squared error between
    observed and estimated phosphorylation levels.

    # Arguments
    - `params::Vector{Float64}`: Vector of parameters containing alpha and beta values.
    - `problem_data::Dict`: Dictionary containing all necessary data for the objective
      function, including P_initial_array, P_initial, K_index, etc.

    # Returns
    - `Float64`: The computed objective function value.
    """
    P_initial_array = problem_data[:P_initial_array]
    P_initial = problem_data[:P_initial]
    K_index = problem_data[:K_index]
    K_array = problem_data[:K_array]
    gene_psite_counts = problem_data[:gene_psite_counts]
    beta_counts = problem_data[:beta_counts]
    time_cols = problem_data[:time_cols]
    n_pairs = size(P_initial_array, 1)

    P_estimated = calculate_estimated_series(
        params,
        P_initial,
        K_index,
        gene_psite_counts,
        beta_counts,
        time_cols
    )
    ## Simple
    residuals = P_initial_array - P_estimated
    return sum(residuals .^ 2) / n_pairs
end

# Function to calculate residuals
function calculate_residuals(observed, estimated)
    """
    Calculates the residuals between observed and estimated phosphorylation levels.

    # Arguments
    - `observed::Matrix{Float64}`: Matrix of observed phosphorylation levels.
    - `estimated::Matrix{Float64}`: Matrix of estimated phosphorylation levels.

    # Returns
    - `Matrix{Float64}`: Matrix of residuals computed as `observed - estimated`.
    """
    return estimated .- observed
end
# ------------------------------------------------------------------------------
# 6) Constrained optimization
# ------------------------------------------------------------------------------
function constrained_optimization(
    P_initial_array,
    P_initial,
    K_index,
    K_array,
    gene_psite_counts,
    beta_counts,
    bounds,
    time_cols
)
    """
    Performs constrained optimization using the PRIMA COBYLA algorithm to minimize
    the objective function while adhering to specified constraints.

    # Arguments
    - `P_initial_array::Matrix{Float64}`: Matrix of initial phosphorylation levels.
    - `P_initial::Dict{Tuple{String, String}, Dict{String, Any}}`: Initial P data
      mapping genes and psites to kinases and time series.
    - `K_index::Dict{String, Vector{Float64}}`: Index mapping kinase-psite pairs to
      their time series vectors.
    - `K_array::Matrix{Float64}`: Matrix where each column represents a kinase-psite
      time series.
    - `gene_psite_counts::Vector{Int}`: Vector indicating the number of kinases per
      gene-psite pair.
    - `beta_counts::Dict{String, Int}`: Dictionary tracking beta counts for each
      kinase-psite pair.
    - `bounds::Matrix{Float64}`: Matrix specifying lower and upper bounds for each
      parameter.
    - `time_cols::Vector{String}`: Vector of column names representing time points.

    # Returns
    - `Tuple{Float64, Vector{Float64}}`: A tuple containing the minimum objective
      function value and the optimized parameter vector.
    """
    # Number of alpha and beta parameters
    num_alpha = sum(gene_psite_counts)
    num_beta = length(beta_counts)

    # Extract beta_bounds from the concatenated bounds
    beta_params = [rand() for _ in 1:num_beta]

    # Combine alpha and beta parameters into a single vector
    alpha_params = [rand() for _ in 1:num_alpha]  # Random values between 0 and 1 for alpha

    x = vcat(alpha_params, beta_params)

    # Define bounds for optimization
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    # Comment out
    # For using nonlinear_constraints for non_lineareq type (20x slower)
    if constraint_type == "nonlinear"
        # Nonlinear constraints for PRIMA
        function nonlinear_constraints(x::Vector{Float64})::Vector{Float64}
            constraints = Float64[]

            # Sum of alphas for each gene-psite group must equal 1
            alpha_start = 1
            for count in gene_psite_counts
                push!(constraints, sum(x[alpha_start:alpha_start+count-1]) - 1)
                alpha_start += count
            end

            # Sum of betas for each kinase's psites must equal 1
            beta_start = num_alpha + 1
            for kinase in unique([split(k, "|")[1] for k in keys(beta_counts)])
                # Collect all indices of betas associated with the current kinase
                beta_indices = [
                    idx for (idx, kpair) in enumerate(keys(beta_counts))
                    if startswith(kpair, "$(kinase)|")
                ]
                push!(constraints, sum(x[beta_start.+beta_indices.-1]) - 1)
            end

            return constraints
        end
        nonlinear_eq_constraint = nonlinear_constraints
        linear_eq_constraint = nothing
    else
        # Linear Constraints for PRIMA
        function linear_constraints(num_alpha, gene_psite_counts, beta_counts)
            num_constraints = length(gene_psite_counts) + length(beta_counts)
            num_vars = num_alpha + length(beta_counts)

            Aₑ = spzeros(num_constraints, num_vars)
            bₑ = ones(num_constraints)

            # Construct alpha constraints
            alpha_start = 1
            for (i, count) in enumerate(gene_psite_counts)
                Aₑ[i, alpha_start:(alpha_start + count - 1)] .= 1.0
                alpha_start += count
            end

            # Construct beta constraints
            beta_start = num_alpha + 1
            for (j, kinase) in enumerate(unique([split(k, "|")[1] for k in keys(beta_counts)]))
                # Collect indices of betas associated with this kinase
                beta_indices = [
                    idx for (idx, kpair) in enumerate(keys(beta_counts))
                    if startswith(kpair, "$(kinase)|")
                ]
                Aₑ[length(gene_psite_counts) + j, beta_start .+ beta_indices .- 1] .= 1.0
            end

            return Aₑ, bₑ
        end
        Aₑ, bₑ = linear_constraints(num_alpha, gene_psite_counts, beta_counts)
        nonlinear_eq_constraint = nothing
        linear_eq_constraint = (Aₑ, bₑ)
    end

    # Wrap objective function to include problem data
    problem_data = Dict(
        :P_initial_array => P_initial_array,
        :P_initial => P_initial,
        :K_index => K_index,
        :K_array => K_array,
        :gene_psite_counts => gene_psite_counts,
        :beta_counts => beta_counts,
        :time_cols => time_cols
    )

    function wrapped_objective(x::Vector{Cdouble})::Real
        return objective_function(x, problem_data)
    end

    # Solve the optimization problem using most suitable Powell's algorithm
    x, info = prima(
        wrapped_objective,
        x;
        xl=lb,
        xu=ub,
        nonlinear_eq=nonlinear_eq_constraint,
        linear_eq=linear_eq_constraint,
        iprint=verbosity_level,
        rhoend=1e-6,
    )

    println("Optimization Summary:")
    println("  Converged: ", issuccess(info))
    println("  Objective Value: ", info.fx)
    println("  Function Evaluations: ", info.nf)
    println("  Constraint Violation: ", info.cstrv)
    println("  Reason for Termination: ", PRIMA.reason(info))
    return info.fx, x, info
end

# ------------------------------------------------------------------------------
# 7) Extract and print solutions
# ------------------------------------------------------------------------------
function extract_and_print_optimized_values(params, P_initial, gene_psite_counts, beta_counts)
    """
    Extracts optimized alpha and beta values from the parameter vector and prints
    them in a readable format.

    # Arguments
    - `params::Vector{Float64}`: Vector of optimized parameters containing alpha and
      beta values.
    - `P_initial::Dict{Tuple{String, String}, Dict{String, Any}}`: Initial P data
      mapping genes and psites to kinases and time series.
    - `K_index::Dict{String, Vector{Float64}}`: Index mapping kinase-psite pairs to
      their time series vectors.
    - `gene_psite_counts::Vector{Int}`: Vector indicating the number of kinases per
      gene-psite pair.
    - `beta_counts::Dict{String, Int}`: Dictionary tracking beta counts for each
      kinase-psite pair.
    """
    # Extract alphas for gene-psite-kinase
    alpha_values = Dict{Tuple{String, String}, Dict{String, Float64}}()
    alpha_start = 1
    all_gene_psite_keys = collect(keys(P_initial))

    for ((gene, psite), count) in zip(all_gene_psite_keys, gene_psite_counts)
        kinases = P_initial[(gene, psite)]["Kinases"]
        alpha_values[(gene, psite)] = Dict(zip(kinases, params[alpha_start:alpha_start+count-1]))
        alpha_start += count
    end

    # Extract betas for kinase-psite
    beta_values = Dict{Tuple{String, String}, Float64}()
    beta_start = sum(gene_psite_counts) + 1

    for kpair in keys(beta_counts)
        if occursin("|", kpair)
            kinase, psite = split(kpair, "|", limit=2)
            beta_values[(kinase, psite)] = params[beta_start]
            beta_start += 1
        else
            println("Warning: Invalid key format in beta_counts: $kpair")
        end
    end

    # Display optimized alpha values
    println("\nOptimized Alpha:")
    for ((gene, psite), kinases) in sort(alpha_values)
        println("Protein: $gene \t Residue_Position: $psite")
        for (kinase, value) in sort(kinases)
            println("  Kinase: $kinase \t Value: $value")
        end
    end

    # Display optimized beta values
    println("\nOptimized Beta:")
    for ((kinase, psite), value) in sort(beta_values)
        println("Kinase: $kinase \t Residue_Position: $psite \t Value: $value")
    end

    return alpha_values, beta_values
end

# ------------------------------------------------------------------------------
# 8) Calculation of evaluation metrics
# ------------------------------------------------------------------------------
function calculate_metrics(true_values, predicted_values)
    """
    Calculates various evaluation metrics to assess the performance of the model,
    including RMSE, MAE, MAPE, R², and Adjusted R².

    # Arguments
    - `true_values::Matrix{Float64}`: Matrix of true observed phosphorylation levels.
    - `predicted_values::Matrix{Float64}`: Matrix of predicted phosphorylation levels.

    # Returns
    - `Dict{String, Float64}`: Dictionary containing calculated metrics:
        - `"RMSE"`: Root Mean Squared Error.
        - `"MAE"`: Mean Absolute Error.
        - `"MAPE"`: Mean Absolute Percentage Error.
        - `"R2"`: R-squared.
        - `"Adjusted R2"`: Adjusted R-squared.
    """
    # Ensure dimensions match
    @assert size(true_values) == size(predicted_values)

    # Residuals
    residuals = predicted_values - true_values

    # Metrics
    n = size(true_values, 1)  # Number of observations
    p = size(predicted_values, 2)  # Number of predictors

    mse = mean(residuals .^ 2)
    mae = mean(abs.(residuals))
    mape = mean(abs.(residuals ./ true_values)) * 100
    r2 = 1 - sum(residuals .^ 2) / sum((true_values .- mean(true_values, dims=1)) .^ 2)
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    return Dict(
        "RMSE" => sqrt(mse),
        "MAE" => mae,
        "MAPE" => mape,
        "R2" => r2,
        "Adjusted R2" => adjusted_r2
    )
end

# ------------------------------------------------------------------------------
# 5) Saving results in excel sheet
# ------------------------------------------------------------------------------
# Save optimization results to an Excel file

function save_results_to_excel(
    output_file,
    alpha_values,
    beta_values,
    metrics,
    residuals,
    P_estimated,
    P_initial_array,
    time_cols::Vector{String},
    interaction_df
)
    """
    Saves optimization results, including alpha/beta values, metrics, residuals,
    and observed/estimated values, into an Excel file with multiple sheets.

    # Arguments
    - `output_file`: Path to save the Excel file.
    - Various arguments represent optimization results and metrics.
    """
    XLSX.openxlsx(output_file, mode="w") do xf
        # Alpha Values Sheet
        alpha_data = DataFrame(
            Protein = String[],
            Psite = String[],
            Kinase = String[],
            Alpha = Float64[]
        )
        for ((gene, psite), kinases) in sort(alpha_values)
            for (kinase, value) in sort(kinases)
                push!(alpha_data, (gene, psite, kinase, value))
            end
        end
        XLSX.writetable!(XLSX.addsheet!(xf, "Alpha Values"), alpha_data)

        # Beta Values Sheet
        beta_data = DataFrame(
            Kinase = String[],
            Psite = String[],
            Beta = Float64[]
        )
        for ((kinase, psite), value) in sort(beta_values)
            push!(beta_data, (kinase, psite, value))
        end
        XLSX.writetable!(XLSX.addsheet!(xf, "Beta Values"), beta_data)

        # Optimization Summary Sheet
        summary_data = DataFrame(
            Metric = collect(keys(metrics)),
            Value = collect(values(metrics))
        )
        XLSX.writetable!(XLSX.addsheet!(xf, "Optimization Summary"), summary_data)

        # Residuals Sheet
        sheet_residuals = XLSX.addsheet!(xf, "Residuals")
        residuals_data = DataFrame(
            Dict(
                "Gene" => String[],
                "Psite" => String[],
                [timepoint => [] for timepoint in time_cols]...
            )
        )
        # Iterate through original data order
        for idx in axes(P_initial_array, 1)  # Match residual rows with the original order
            gene = interaction_df.GeneID[idx]
            psite = interaction_df.Psite[idx]

            row = Any[gene, psite]  # Create a row with gene, psite
            append!(row, residuals[idx, :]...)  # Append residuals
            push!(residuals_data, row)
        end
        # Sort and order columns
        rename!(
            residuals_data,
            Dict(
                names(residuals_data)[3:end] .=> time_cols
            )
        )
        XLSX.writetable!(sheet_residuals, residuals_data)

        # Estimated Values Sheet
        sheet_estimated = XLSX.addsheet!(xf, "Estimated Values")
        estimated_data = DataFrame(
            Dict(
                "Gene" => String[],
                "Psite" => String[],
                [timepoint => [] for timepoint in time_cols]...
            )
        )
        # Iterate through original data order
        for idx in axes(P_estimated, 1)  # Match estimated rows with the original order
            gene = interaction_df.GeneID[idx]
            psite = interaction_df.Psite[idx]

            row = Any[gene, psite]  # Create a row with gene, psite
            append!(row, P_estimated[idx, :]...)  # Append estimated values
            push!(estimated_data, row)
        end
        # Sort and order columns
        rename!(
            estimated_data,
            Dict(
                names(estimated_data)[3:end] .=> time_cols
            )
        )
        XLSX.writetable!(sheet_estimated, estimated_data)

        # Observed Values Sheet
        sheet_observed = XLSX.addsheet!(xf, "Observed Values")
        observed_data = DataFrame(
            Dict(
                "GeneID" => String[],
                "Psite" => String[],
                [timepoint => Float64[] for timepoint in time_cols]...  # Time columns with Float64 type
            )
        )
        # Use the original order from interaction_df to extract gene and psite
        for idx in eachindex(P_initial_array[:, 1])  # Iterate over rows of the array
            # Extract the corresponding gene and psite from the original data
            gene = interaction_df.GeneID[idx]
            psite = interaction_df.Psite[idx]

            # Construct the row
            row = Any[gene, psite]
            append!(row, P_initial_array[idx, :]...)  # Add the time series data
            push!(observed_data, row)
        end
        rename!(
            observed_data,
            Dict(
                names(observed_data)[3:end] .=> time_cols
            )
        )
        XLSX.writetable!(sheet_observed, observed_data)
    end
end

# ------------------------------------------------------------------------------
# 5) Saving plots
# ------------------------------------------------------------------------------
# Define the function to plot residuals for a gene
function plot_residuals_for_gene(
    gene,
    gene_data,
    time_points
)
    """
    Generates and saves various residual plots for a specific gene,
    including observed vs estimated plots with error bars and fills,
    cumulative residuals, histograms, and QQ plots.

    # Arguments
    - `gene::String`: The gene identifier for which to plot residuals.
    - `gene_data::Dict{String, Any}`: Dictionary containing data for the gene, including:
        - `"psites"`: Vector of psites associated with the gene.
        - `"observed"`: Vector of observed phosphorylation time series.
        - `"estimated"`: Vector of estimated phosphorylation time series.
        - `"residuals"`: Vector of residuals for each psite.
    - `time_points::Vector{Float64}`: Vector of time points corresponding to the phosphorylation measurements.
    """
    # Define colors for psites
    colors = ColorScheme(distinguishable_colors(length(gene_data["psites"]), transform=protanopic))

    # 1. Observed vs Estimated Plot with Error Bars
    plot(size=(800, 800))
    for (i, psite) in enumerate(gene_data["psites"])
        observed = gene_data["observed"][i]
        estimated = gene_data["estimated"][i]

        # Calculate standard error
        std_dev = std(observed)
        n = length(observed)
        se = std_dev / sqrt(n)

        # Plot observed data with error bars
        plot!(
            time_points[6:end],
            observed[6:end],
            ribbon=se,
            label=psite,
            color=colors[i],
            marker=(:square, 4, 0.8),
            markerstrokecolor = :black,
            linestyle=:dash,
            linewidth=1,
        )
        # Plot estimated series
        plot!(
            time_points,
            estimated,
            label="",
            color=colors[i],
            marker=(:circle, 4, 0.6),
            markerstrokecolor = :black,
            linestyle=:solid,
            linewidth=1,
        )
    end
    xlabel!("Time (hrs)")
    ylabel!("Phosphorylation Level (FC)")
    title!(gene)
    plot!(grid=true)
    savefig("abopt/results/fit_errorbars_$(gene).png")

    # 3. Cumulative Sum of Residuals
    plot(size=(800, 800))
    for (i, psite) in enumerate(gene_data["psites"])
        plot!(
            time_points,
            cumsum(gene_data["residuals"][i]),
            label=psite,
            color=colors[i],
            marker=(:circle, 4, 0.8),
            markerstrokecolor = :black,
            linestyle=:solid,
            linewidth=1,
        )
    end
    xlabel!("Time (hrs)")
    ylabel!("Cumulative Residuals")
    title!(gene)
    plot!(grid=true)
    savefig("abopt/results/cumulative_residuals_$(gene).png")


    # 4. Histogram of Residuals
    histogram(size=(800, 800))
    for (i, psite) in enumerate(gene_data["psites"])
        histogram!(
            gene_data["residuals"][i],
            bins=20,
            label=psite,
            color=colors[i],
            alpha=0.6,
            normalization=:pdf,
        )
    end
    xlabel!("Residuals")
    ylabel!("Frequency")
    title!(gene)
    savefig("abopt/results/histogram_residuals_$(gene).png")

    # 5. QQ Plot of Residuals
    plot(
        title="$gene",
        xlabel="Theoretical Quantiles",
        ylabel="Sample Quantiles",
        size=(800, 800)
    )
    for (i, psite) in enumerate(gene_data["psites"])
        residuals = gene_data["residuals"][i]
        sorted_data = sort(collect(residuals))  # Ensure it’s sorted
        theoretical_quantiles = quantile(Normal(0, 1), range(0, 1, length=length(sorted_data) + 2)[2:end-1])
        scatter!(
            theoretical_quantiles, sorted_data,
            label=psite, color=colors[i], marker=(:circle, 4, 0.6), markerstrokecolor=:black
        )
    end

    # Calculate limits for the reference line
    all_residuals = vcat([gene_data["residuals"][i] for i in eachindex(gene_data["psites"])]...)
    data_min, data_max = extrema(all_residuals)
    ref_min = min(data_min, -3)  # Adjust based on the desired theoretical range
    ref_max = max(data_max, 3)

    # Add a 45-degree reference line
    plot!(
        [ref_min, ref_max], [ref_min, ref_max],
        seriestype=:line,
        linestyle=:dash,
        color=:red,
    )

    savefig("abopt/results/qqplot_residuals_$(gene).png")

end

# ------------------------------------------------------------------------------
# 9) Main
# ------------------------------------------------------------------------------
function main()
    # Load input data
    full_hgnc_df, interaction_df = load_data("abopt/data/input1.csv", "abopt/data/input2.csv", estimate_missing_kinases)

    # Define the time points
    time_points = [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
    time_cols = [ "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14" ]

    # 1) Prepare P_initial and P_initial_array
    P_initial, P_initial_array = prepare_initial_arrays(full_hgnc_df, interaction_df, time_cols)

    # Define protein groups for psites and kinases
    n_pairs = length(P_initial)

    # 2) Prepare K_index, K_array, beta_counts
    K_index, K_array, beta_counts = prepare_kinase_data(full_hgnc_df, interaction_df, time_cols, estimate_missing_kinases)

    # 3) Gene–psite counts: how many kinases per gene–psite
    gene_psite_counts = [length(v["Kinases"]) for v in values(P_initial)]

    # Define the number of alpha and beta parameters
    num_alpha = sum(gene_psite_counts)
    num_beta = length(beta_counts)  # one parameter per key

    # Fix alpha bounds between 0 and 1
    alpha_bounds = [(0, 1) for _ in 1:num_alpha]

    # Allow user-defined bounds for beta
    beta_bounds = [(beta_lower, beta_upper) for _ in 1:num_beta]

    # 4) Combine alpha and beta bounds
    bounds = vcat(alpha_bounds, beta_bounds)

    println("Number of threads available: ", Threads.nthreads())

    # Perform constrained optimization
    min_f, optimized_params, info = constrained_optimization(
        P_initial_array,
        P_initial,
        K_index,
        K_array,
        gene_psite_counts,
        beta_counts,
        bounds,
        time_cols
    )

    println("\nMinimum Error Value: ", min_f)

    # Extract and print alpha and beta values
    alpha_values, beta_values = extract_and_print_optimized_values(optimized_params, P_initial, gene_psite_counts, beta_counts)

    # Calculate predicted values using optimized parameters
    P_estimated = calculate_estimated_series(
        optimized_params,
        P_initial,
        K_index,
        gene_psite_counts,
        beta_counts,
        time_cols
    )

    # Calculate and display the metrics
    metrics = calculate_metrics(P_initial_array, P_estimated)
    residuals = calculate_residuals(P_initial_array, P_estimated)

    println("\nPerformance Metrics:\n")
    for (metric, value) in metrics
        println("\t$metric: $value")
    end

    output_file = "abopt/results/results.xlsx"
    save_results_to_excel(
        output_file,
        alpha_values,
        beta_values,
        metrics,
        residuals,
        P_estimated,
        P_initial_array,
        time_cols,
        interaction_df
    )

    # Initialize a DefaultDict to group data by genes
    genes_data = DefaultDict(() -> Dict("psites" => [], "observed" => [], "estimated" => [], "residuals" => []))

    for (i, ((gene, psite), _)) in enumerate(P_initial)
        observed_series = P_initial_array[i, :]
        estimated_series = P_estimated[i, :]
        residuals_series = residuals[i, :]

        push!(genes_data[gene]["psites"], psite)
        push!(genes_data[gene]["observed"], observed_series)
        push!(genes_data[gene]["estimated"], estimated_series)
        push!(genes_data[gene]["residuals"], residuals_series)
    end

    # Plot residuals for each gene
    for (gene, gene_data) in genes_data
        plot_residuals_for_gene(gene, gene_data, time_points)
    end

    println("Completed.")
end

# Execute main() when the script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
