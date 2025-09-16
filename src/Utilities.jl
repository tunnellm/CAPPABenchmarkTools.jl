module Utilities

# Basic Linear Algebra
using LinearAlgebra
using SparseArrays

# Stable Random Number Generation
using StableRNGs
using Random    # shuffle

# Orderings
import AMD
import SymRCM
import Metis

# File I/O
import CSV
import DataFrames
using CodecZstd
using MatrixMarket

# Matrix Utilities
using Laplacians
using GenericArpack

export check_previous, load_matrix_names, do_log
export load_matrix, load_graph, package
export norm_information
export compute_tolerances, compute_operator_norms
export problem_interface
export is_laplacian, make_augmented, extract_negative, laplace_to_adj

"""
    norm_information

Stores various norm values for a given matrix, used for error estimation and convergence analysis.

# Fields
- `spectral_norm::Float64`: The spectral norm, representing the largest singular (eigen) value.
- `infinity_norm::Float64`: The infinity norm, defined as the maximum absolute row sum.
- `frobenius_norm::Float64`: The Frobenius norm, computed as the square root of the sum of squared elements.
- `jacobian::Float64`: The spectral radius of the Jacobi iteration matrix, used in iterative solver analysis.

# Notes
- These norms are used to compute relative residuals and norm-wise backward errors.
- The spectral norm is estimated using an iterative eigenvalue solver.
"""
struct norm_information
    spectral_norm::Float64
    infinity_norm::Float64
    frobenius_norm::Float64
    jacobian::Float64
    function norm_information(
        spectral_norm::Float64,
        infinity_norm::Float64,
        frobenius_norm::Float64,
        jacobian::Float64,
    )
        return new(spectral_norm, infinity_norm, frobenius_norm, jacobian)
    end
end

"""
    struct package

Represents a numerical linear system along with associated metadata for solving and preconditioning.

# Fields
- `A::SparseMatrixCSC`: The system matrix.
- `b::Vector`: The right-hand side vector.
- `seed::Vector`: A seed vector, often used for generating initial guesses or perturbations.
- `name::String`: A descriptive name for the problem instance.
- `norms::norm_information`: Precomputed norm information for `A`, including spectral, Frobenius, and infinity norms.
- `diagonally_dominant::Bool`: Indicates whether `A` is diagonally dominant.
- `nullspace::Int64`: The location of the removed degree of freedom in Laplacian and SPD problems, or `0` if not applicable.

# Notes
- This struct is primarily used in preconditioned iterative solvers.
- It provides an interface for handling different problem formulations, such as scaled and unscaled versions of `A`.
- The `nullspace` field helps track any constraints or removed degrees of freedom in Laplacian and SPD problems.
"""
struct package
    A::SparseMatrixCSC
    b::Vector
    seed::Vector
    name::String
    norms::norm_information
    diagonally_dominant::Bool
    nullspace::Int64
end

include("Tolerances.jl")


"""
    check_previous(path::String, preconditioner_name::String, check_tolerance::Bool; control_work=-1)

Checks the previous convergence data for a given preconditioner to determine whether the solver should continue or terminate early.

# Arguments
- `path::String`: Path to the directory containing convergence data files.
- `preconditioner_name::String`: Name of the preconditioner used in the solver.
- `check_tolerance::Bool`: Whether to check if the solution met the convergence tolerance.
- `control_work::Int` (optional, default `-1`): Maximum allowed computational cost before termination. If `-1`, this check is skipped.

# Returns
- `Int`: Returns:
  - `1` if the accumulated computational cost exceeds `control_work`.
  - `-1` if convergence data is missing, empty, or the minimum recorded tolerance is above the threshold.
  - The maximum iteration count from the available convergence data otherwise.

# Notes
- If `control_work` is positive and raw iteration data exists (`raw_data.csv.zstd`), the function checks if the combined base and preconditioner cost exceed `control_work`. If so, it returns the number of iterations recorded. If the work limit is not exceeded, the function proceeds to check the convergence data. If converged, it returns the highest iteration count recorded and returns -1 otherwise.
- If `check_tolerance` is enabled, the function verifies whether the minimum recorded tolerance meets the required threshold. If not met, it returns `-1`.
- Otherwise, the function returns the highest iteration count recorded in the available convergence data.
"""
function check_previous(
    path::String,
    preconditioner_name::String,
    check_tolerance::Bool;
    control_work = -1,
)

    if control_work > 0 && isfile(path * preconditioner_name * "raw_data.csv.zstd")
        io = open(path * preconditioner_name * "raw_data.csv.zstd", "r")
        stream = ZstdDecompressorStream(io)
        try
            df = CSV.read(stream, DataFrames.DataFrame)
            if DataFrames.nrow(df) <= 0
                return -1
            end
            if (df[end, " Base Cost"] + df[end, " Preconditioner Cost"]) > control_work
                # return check_previous(path, preconditioner_name, true)
                return size(df, 1) - 1
            else  # check for convergence, return iteration count if converged and -1 otherwise
                return df[end, " Relative Residual"] <= minimum(tolerances) ?
                       size(df, 1) - 1 : -1
            end
        catch e
            @warn "Error occurred while reading raw data: $e"
            open(joinpath(path, preconditioner_name, "log.txt"), "a") do f
                write(f, "$(time())\n")
                write(f, e)
                write(f, "\n")
            end
            return -1
        finally
            close(stream)
            close(io)
        end
    end

    convergence_data = ["relative_residual.csv"]  # we can check the other metrics but largely converge at a rate much faster than the relative residual
    location = path * preconditioner_name * "convergence_data_"
    local maximum_iterations = 0
    for file in convergence_data
        location_ = location * file
        if isfile(location_)
            data = CSV.read(location_, DataFrames.DataFrame)
            if !isempty(data)
                min_value = minimum(data[:, " Tolerance Met"])
            else
                return -1
            end
            if check_tolerance && min_value > minimum_tolerance
                return -1
            else
                maximum_iterations =
                    maximum([maximum_iterations maximum(data[:, "Iteration"])])
            end
        else
            return -1
        end
    end
    return maximum_iterations
end

"""
    do_log(location::String, msg::AbstractString)

Appends a plain string message to the log file at `location`, 
preceded by a timestamp.
"""
function do_log(location::String, msg::AbstractString)
    open(location, "a") do f
        write(f, string(time(), "\n"))
        write(f, msg, "\n\n")
    end
end

"""
    do_log(location::String, err::Exception)

Appends an exception report to the log file at `location`, 
preceded by a timestamp. Includes the full error message and stacktrace.
"""
function do_log(location::String, err::Exception)
    open(location, "a") do f
        write(f, string(time(), "\n"))
        showerror(f, err, catch_backtrace())
        write(f, "\n\n")
    end
end

"""
    do_log(location::String, x)

Fallback method. Converts `x` to a string and logs it as if it were a message.
"""
function do_log(location::String, x)
    do_log(location, string(x))
end




"""
    compute_tolerances(residual_norm::AbstractFloat, norm_b::AbstractFloat, norm_x::AbstractFloat, norms::norm_information)

Computes relative tolerances for convergence analysis based on the residual norm and matrix norms.

# Arguments
- `residual_norm::AbstractFloat`: The norm of the residual vector `r = b - Ax`.
- `norm_b::AbstractFloat`: The norm of the right-hand side vector `b`.
- `norm_x::AbstractFloat`: The norm of the current solution vector `x`.
- `norms::norm_information`: Struct containing precomputed matrix norms:
    - `spectral_norm`: Largest singular value of the matrix.
    - `infinity_norm`: Maximum absolute row sum.
    - `frobenius_norm`: Frobenius norm.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
    - `relative_residual`: `‖r‖ / ‖b‖`
    - `relative_normwise_2`: Backward error estimate using the spectral norm.
    - `relative_normwise_i`: Backward error estimate using the infinity norm.
    - `relative_normwise_f`: Backward error estimate using the Frobenius norm.

# Notes
- The normwise backward errors approximate how much `A` and `b` must be perturbed for `x` to be an exact solution.
- `relative_normwise_2` is typically the most reliable measure, but `relative_normwise_i` and `relative_normwise_f` can be useful for structured problems.
"""
function compute_tolerances(
    residual_norm::AbstractFloat,
    norm_b::AbstractFloat,
    norm_x::AbstractFloat,
    norms::norm_information,
)
    relative_residual = residual_norm / norm_b
    relative_normwise_2 = residual_norm / (norm_b + norms.spectral_norm * norm_x)
    relative_normwise_i = residual_norm / (norm_b + norms.infinity_norm * norm_x)
    relative_normwise_f = residual_norm / (norm_b + norms.frobenius_norm * norm_x)
    return (
        relative_residual,
        relative_normwise_2,
        relative_normwise_i,
        relative_normwise_f,
    )
end

"""
    compute_operator_norms(A::SparseMatrixCSC) -> norm_information

Computes various matrix norms used in numerical analysis and iterative solvers.

# Arguments
- `A::SparseMatrixCSC`: The input sparse matrix.

# Returns
- `norm_information`: A struct containing:
    - `spectral_norm::Float64`: The largest eigenvalue of `A`, estimated using the Lanczos method (`symeigs`).
    - `infinity_norm::Float64`: The maximum absolute row sum of `A`.
    - `frobenius_norm::Float64`: The Frobenius norm, computed as the square root of the sum of squared entries.
    - `jacobian::Float64`: The spectral radius of the Jacobi iteration matrix when `A` is symmetric positive definite (SPD).

# Notes
- The spectral norm is computed via `symeigs`, which uses an iterative eigenvalue solver.
- The parameter `p` controls the number of Lanczos basis vectors and is adaptively set based on `A`'s size.
- The Jacobi spectral radius is computed by scaling `A` using its diagonal and removing diagonal entries.
- If `A` is not SPD, the computed Jacobi spectral radius may not be meaningful and should be interpreted accordingly.
"""
function compute_operator_norms(A::SparseMatrixCSC)

    n = size(A, 1)

    p =
        n > 10^8 ? 125 :
        n > 2.5 * 10^7 ? 250 :
        n > 10^7 ? 400 :
        n > 2.5 * 10^6 ? 600 :
        n > 10^6 ? 550 :
        n > 500000 ? 500 : n > 250000 ? 400 : n > 100000 ? 300 : n > 50000 ? 200 : 100

    spectral_norm::Float64 =
        symeigs(A, 1; which = :LM, maxiter = div(n, 2), tol = 1e-6, ncv = p).values[1]
    infinity_norm::Float64 = maximum(sum(abs.(A), dims = 1))
    frobenius_norm::Float64 = sqrt(sum(x -> x^2, A.nzval))

    D = diag(A) |> D -> 1 ./ sqrt.(D) |> D -> Diagonal(D)

    J = copy(A)
    foreach(i -> J[i, i] = 0.0, 1:n)
    J = D * A
    J = J * D
    # this is the spectral radius of the Jacobi iteration matrix when A is SPD only
    # going to need to rewrite this for non-SPD matrices
    jacobian_norm::Float64 =
        symeigs(J, 1; which = :LM, maxiter = div(n, 2), tol = 1e-6, ncv = p).values[1]

    return norm_information(spectral_norm, infinity_norm, frobenius_norm, jacobian_norm)

end


"""
    load_matrix_names(list_location::String) -> Vector{String}

Loads matrix names from a file, extracting the first entry from each line.

# Arguments
- `list_location::String`: Path to the file containing a list of matrix names.

# Returns
- `Vector{String}`: A list of matrix names extracted from the file.

# Notes
- The file is expected to contain comma-separated values (CSV).
- Only the first column of each line is read, assuming it contains matrix names.
"""
function load_matrix_names(list_location::String)
    matrix_names = []
    open(list_location) do f
        for line in eachline(f)
            push!(matrix_names, String(split(line, ",")[1]))
        end
    end
    return matrix_names
end

"""
    load_graph(filename::String, reordering::String) -> problem_interface

Loads a graph from a Matrix Market file, processes its largest connected component, and constructs its Laplacian.

# Arguments
- `filename::String`: Name of the graph file (without extension) located in the `./graphs/` directory.
- `reordering::String`: Reordering scheme to apply to the Laplacian. Must be one of `"natural"`, `"rcm"`, `"amd", or "metis"`.

# Returns
- `problem_interface`: A struct containing the original, scaled, and unscaled Laplacian matrices with metadata.

# Notes
- The function reads an adjacency matrix from a `.mtx` file and extracts its largest connected component.
- The adjacency matrix is explicitly symmetrized to ensure numerical stability.
- The graph Laplacian is computed as `L = D - A`, where `D` is the degree matrix.
- The most "prolific" node (highest unweighted degree) is identified and removed to enforce a unique solution.
- If precomputed norms exist, they are loaded; otherwise, they are computed and stored.
- The function returns the problem in three formats:
    - `Original`: The full Laplacian matrix.
    - `Scaled`: The preconditioned and normalized SPD Laplacian.
    - `Special`: The unscaled SPD Laplacian.
"""
function load_graph(filename::String, reordering::String)

    reordering = lowercase(reordering)

    @assert reordering ∈ ["natural", "rcm", "amd", "metis"]#, "dissection"] # "dissection" is not yet implemented

    println("Loading $(filename)")

    base = "./graphs/"
    norms_exists =
        isfile(base * filename * "/scaled_norm.csv") &&
        isfile(base * filename * "/unscaled_norm.csv") &&
        isfile(base * filename * "/laplacian_norm.csv")

    filename_stripped = ""

    if contains(filename, "/")
        filename_stripped = String(split(filename, "/")[end])
    else
        filename_stripped = filename
    end

    Adjacency = MatrixMarket.mmread(base * filename * "/" * filename_stripped * ".mtx")

    println("Computing connected components for $(filename)")

    co = components(Adjacency)

    comps = vecToComps(co)

    m, loc = findmax(x -> length(x), comps)

    # Isolate the largest component
    Adjacency = Adjacency[comps[loc], comps[loc]]

    # Explicitly symmetrize the Adjacency matrix.
    Adjacency = Adjacency + Adjacency'
    Adjacency.nzval .*= 0.5

    # The resulting graph-Laplacian matrix is equivalent whether we set the diagonal to 0 or not.
    foreach(x -> Adjacency[x, x] = 0.0, 1:m)

    # Create the Laplacian matrix from the adjacency matrix
    Degree = vec(sum(Adjacency, dims = 2))
    Laplacian = Diagonal(Degree) - Adjacency

    println("Loaded $(filename) into Julia")

    if reordering != "natural"
        println("Computing $(reordering) reordering for $(filename)")
    end

    if reordering == "natural"
        permutation = collect(1:m)
    elseif reordering == "rcm"
        permutation = SymRCM.symrcm(Laplacian)
    elseif reordering == "amd"
        permutation = AMD.symamd(Laplacian)
    elseif reordering == "metrix"
        permutation, _ = Metis.permutation(Laplacian)
    end

    prolific = findmax(i -> Adjacency.colptr[i+1] - Adjacency.colptr[i], 1:m)[2]

    println("Computing Laplacian seed for $(filename)")
    seed = Vector{Float64}(undef, m)

    rng = StableRNG(123456789)
    # Not positive that we need to warm up the RNG but let's do it anyway.
    for _ = 1:round(Int, log2(m))
        seed .= rand(rng, m)
    end

    # # Set the element corresponding with the largest (unweighted) degree node to 0.
    seed[prolific] = 0.0

    # Find the indices of an even number of the largest elements in the seed
    amt = round(Int, log2(m))
    amt % 2 == 1 && (amt += 1)
    max = partialsortperm(seed, 1:amt, rev = true)

    # Set exactly half of the seed to 1.0 and the other half to -1.0
    seed .= 0.0
    seed[max] .= 1.0
    seed[shuffle(rng, max)[1:(div(amt, 2))]] .= -1.0

    L_RHS = Laplacian * seed

    println("Computing SPD Laplacian for $(filename)")

    # Reorder the Adjacency, Degree, Laplacian, seed, and RHS
    Adjacency = Adjacency[permutation, permutation]
    Degree = Degree[permutation]
    Laplacian = Laplacian[permutation, permutation]
    seed = seed[permutation]
    L_RHS = L_RHS[permutation]

    prolific_reordered = findfirst(x -> x == prolific, permutation)

    println("Most prolific node is at index $(prolific_reordered)")

    SPD_Laplacian = Laplacian[1:m.!=prolific_reordered, 1:m.!=prolific_reordered]
    SPD_RHS = L_RHS[1:m.!=prolific_reordered]

    SPD_Degree = Degree[1:m.!=prolific_reordered]

    SPD_Scaler = Diagonal(1 ./ sqrt.(SPD_Degree))

    SPD_Laplacian_Scaled = SPD_Scaler * SPD_Laplacian
    SPD_Laplacian_Scaled = SPD_Laplacian_Scaled * SPD_Scaler

    SPD_Laplacian_Scaled_SDD = true

    n = size(SPD_Laplacian_Scaled, 1)

    for i = 1:n

        if 2 * SPD_Laplacian_Scaled[i, i] < sum(abs, view(SPD_Laplacian_Scaled, :, i))
            SPD_Laplacian_Scaled_SDD = false
            break
        end

    end

    SPD_RHS_Scaled = SPD_Scaler * SPD_RHS

    # The diagonal should already be 1.0, we set it here explicitly in case of floating point error.
    foreach(x -> SPD_Laplacian_Scaled[x, x] = 1.0, 1:m-1)

    println("Symmetrizing SPD Laplacian for $(filename)")

    # We explicitly symmetrize the matrices here in case of floating point error.
    SPD_Laplacian_Scaled = SPD_Laplacian_Scaled + SPD_Laplacian_Scaled'
    SPD_Laplacian_Scaled.nzval .*= 0.5
    # We do not need to symmetrize the other unscaled matrices because we previously symmetrized the adjacency

    SPD_Laplacian_seed = seed[1:m.!=prolific_reordered]
    SPD_Laplacian_seed_scaled = Diagonal(sqrt.(SPD_Degree)) * SPD_Laplacian_seed

    println("Applied scaling and reordering to $(filename)")

    local scaled_norms
    local unscaled_norms
    local laplacian_norms

    println("Computing norms for $(filename)")

    if !norms_exists  # Compute norms using Scipy

        scaled_norms = compute_operator_norms(SPD_Laplacian_Scaled)
        unscaled_norms = compute_operator_norms(SPD_Laplacian)
        laplacian_norms = compute_operator_norms(Laplacian)

        open(base * filename * "/unscaled_norm.csv", "w") do f
            write(
                f,
                "$(unscaled_norms.spectral_norm),$(unscaled_norms.infinity_norm),$(unscaled_norms.frobenius_norm),$(unscaled_norms.jacobian)",
            )
        end
        open(base * filename * "/scaled_norm.csv", "w") do f
            write(
                f,
                "$(scaled_norms.spectral_norm),$(scaled_norms.infinity_norm),$(scaled_norms.frobenius_norm),$(scaled_norms.jacobian)",
            )
        end
        open(base * filename * "/laplacian_norm.csv", "w") do f
            write(
                f,
                "$(laplacian_norms.spectral_norm),$(laplacian_norms.infinity_norm),$(laplacian_norms.frobenius_norm),$(laplacian_norms.jacobian)",
            )
        end

    else  # Load norms from file
        println("Loading scaled norms for $(filename)")
        file = open(base * filename * "/scaled_norm.csv", "r")
        loaded = readline(file)
        s, i, f, j = split(loaded, ",")
        close(file)

        scaled_norms = norm_information(
            parse(Float64, s),
            parse(Float64, i),
            parse(Float64, f),
            parse(Float64, j),
        )

        println("Loading unscaled norms for $(filename)")
        file = open(base * filename * "/unscaled_norm.csv", "r")
        loaded = readline(file)
        s, i, f, j = split(loaded, ",")
        close(file)

        unscaled_norms = norm_information(
            parse(Float64, s),
            parse(Float64, i),
            parse(Float64, f),
            parse(Float64, j),
        )

        println("Loading Laplacian norms for $(filename)")
        file = open(base * filename * "/laplacian_norm.csv", "r")
        loaded = readline(file)
        s, i, f, j = split(loaded, ",")
        close(file)

        laplacian_norms = norm_information(
            parse(Float64, s),
            parse(Float64, i),
            parse(Float64, f),
            parse(Float64, j),
        )
    end

    return problem_interface(
        package(
            Laplacian,
            L_RHS,
            seed,
            filename_stripped,
            laplacian_norms,
            true,
            prolific_reordered,
        ),
        package(
            SPD_Laplacian_Scaled,
            SPD_RHS_Scaled,
            SPD_Laplacian_seed_scaled,
            filename_stripped,
            scaled_norms,
            SPD_Laplacian_Scaled_SDD,
            0,
        ),
        package(
            SPD_Laplacian,
            SPD_RHS,
            SPD_Laplacian_seed,
            filename_stripped,
            unscaled_norms,
            true,
            0,
        ),
        reordering,
        :symmetric_graph,
        filename_stripped,
    )

end

"""
    problem_interface(
        Original::package,
        Scaled::package,
        Special::package,
        reordering::String,
        type::Symbol
    )

Represents a structured problem instance with different representations of the system matrix.

# Fields
- `Original::package`: The unmodified system, typically the raw input matrix and right-hand side.
- `Scaled::package`: The preconditioned or scaled version of the system, often used to improve numerical stability.
- `Special::package`: A variant of the system with specific modifications, such as padding for SDD enforcement.
- `reordering::String`: The reordering scheme applied to the system, e.g., `"natural"`, `"rcm"`, `"amd"`.
- `type::Symbol`: The type of problem, such as `:SPD_matrix` or `:symmetric_graph`.
- `name::String`: A descriptive name for the problem instance.

# Notes
- This struct allows switching between different representations of the same problem for solver comparison.
- The `reordering` field indicates whether node permutations were applied for better numerical properties.
- The `type` field distinguishes between standard SPD matrices and graph-based problems.
"""
struct problem_interface
    Original::package
    Scaled::package
    Special::package
    reordering::String
    type::Symbol
    name::String
end

"""
    load_matrix(filename::String, reordering::String, SPD::Bool=true) -> problem_interface

Loads a sparse matrix from a Matrix Market file, applies reordering, and constructs different representations of the system.

# Arguments
- `filename::String`: Name of the matrix file (without extension) located in the `./matrices/` directory.
- `reordering::String`: Reordering scheme to apply to the matrix. Must be one of `"natural"`, `"rcm"`, or `"amd"`.
- `SPD::Bool` (optional, default `true`): If `true`, symmetrizes the matrix to ensure it is symmetric positive definite (SPD).

# Returns
- `problem_interface`: A struct containing three representations of the system:
    - `Original`: The raw matrix and its right-hand side vector.
    - `Scaled`: A scaled version for improved numerical stability.
    - `Special`: A padded version ensuring symmetric diagonal dominance (SDD).

# Notes
- Reads a sparse matrix from a `.mtx` file and constructs the right-hand side `b` using a randomized seed.
- If `SPD` is `true`, the matrix is explicitly symmetrized.
- The matrix is reordered according to the specified scheme to improve solver efficiency.
- A scaling transformation is applied for stability, and a padded variant is created if the system is not SDD.
- Precomputed norm information is loaded if available; otherwise, norms are computed and stored.
"""
function load_matrix(filename::String, reordering::String, SPD::Bool = true)

    reordering = lowercase(reordering)

    @assert reordering ∈ ["natural", "rcm", "amd", "metis"] # "dissection" is not yet implemented

    println("Loading $(filename)")

    base = "./matrices/"
    norms_exists =
        isfile(base * filename * "/scaled_norm.csv") &&
        isfile(base * filename * "/unscaled_norm.csv")

    filename_stripped = ""

    if contains(filename, "/")
        filename_stripped = String(split(filename, "/")[end])
    else
        filename_stripped = filename
    end

    println("Load Matrix Market file of $(filename)")

    A_ = MatrixMarket.mmread(base * filename * "/" * filename_stripped * ".mtx")

    m = size(A_, 1)

    seed = ones(m)

    rng = StableRNG(123456789)
    # Not positive that we need to warm up the RNG but let's do it anyway.
    for _ = 1:round(Int, log2(m))
        seed .= rand(rng, m)
    end

    max = partialsortperm(seed, 1:round(Int, log2(m) + 1), rev = true)

    seed .= 0

    seed[max] .= 1 .* (-1) .^ rand(rng, Bool, length(max))

    b = A_ * seed

    if SPD
        A_ .= A_ + A_'
        A_.nzval .= 0.5 .* A_.nzval
    end

    if reordering != "Natural"
        println("Computing $(reordering) reordering for $(filename)")
    end

    if reordering == "natural"
        permutation = collect(1:size(A_, 1))
    elseif reordering == "rcm"
        permutation = SymRCM.symrcm(A_ + A_')  # Some matrices are not symmetric at the bit level
    elseif reordering == "amd"
        permutation = AMD.symamd(A_)
    elseif reordering == "metis"
        permutation, _ = Metis.permutation(A_)
    end

    A_ = copy(A_[permutation, permutation])

    A_Scaled = copy(A_)

    println("Loaded $(filename) into Julia")

    A_SDD = true
    for i in axes(A_, 2)
        if 2 * A_[i, i] < sum(abs, view(A_, :, i))
            A_SDD = false
            break
        end
    end

    # Apply scaling =
    if SPD
        println("Scaling $(filename).")
        A_Unscaler = sqrt.(abs.(diag(A_Scaled, 0)))
        A_Scaler = 1.0 ./ A_Unscaler |> spdiagm
        A_Unscaler = A_Unscaler |> spdiagm
        A_Scaled = A_Scaler * A_Scaled * A_Scaler
        for i = 1:m
            A_Scaled[i, i] = 1.0
        end
        A_Scaled .= A_Scaled + A_Scaled'
        A_Scaled.nzval .= 0.5 .* A_Scaled.nzval
    else
        println("Scaling $(filename).")
        A_Unscaler = diag(A_Scaled, 0)
        A_Scaler = 1.0 ./ A_Unscaler |> spdiagm
        A_Unscaler = A_Unscaler |> spdiagm
        A_Scaled = A_Scaler * A_Scaled
    end


    A_Scaled_SDD = true
    for i in axes(A_Scaled, 2)
        if 2 * A_Scaled[i, i] < sum(abs, view(A_Scaled, :, i))
            A_Scaled_SDD = false
            break
        end
    end

    # Pad the diagonal
    A_Scaled_Padded = copy(A_Scaled)
    if !A_Scaled_SDD && !A_SDD
        for i in axes(A_Scaled_Padded, 2)
            temp = sum(abs, view(A_Scaled_Padded, :, i))
            if 2 * A_Scaled_Padded[i, i] < temp
                A_Scaled_Padded[i, i] = temp + eps(1.0)
            end
        end
        println("Created scaled and padded version of $(filename)")
    elseif A_Scaled_SDD
        println("Scaled version of $(filename) is SDD")
    else
        println("Original $(filename) is SDD")
    end

    # A.data .= A_Scaled.nzval

    local scaled_norms
    local unscaled_norms

    if !SPD
        scaled_norms = norm_information(0.0, 0.0, 0.0, 0.0)
        unscaled_norms = norm_information(0.0, 0.0, 0.0, 0.0)
    elseif !norms_exists  # Compute norms using Scipy

        println("Computing scaled norms")
        scaled_norms = compute_operator_norms(A_Scaled)

        open(base * filename * "/scaled_norm.csv", "w") do f
            write(
                f,
                "$(scaled_norms.spectral_norm),$(scaled_norms.infinity_norm),$(scaled_norms.frobenius_norm),$(scaled_norms.jacobian)",
            )
        end

        unscaled_norms = compute_operator_norms(A_)

        open(base * filename * "/unscaled_norm.csv", "w") do f
            write(
                f,
                "$(unscaled_norms.spectral_norm),$(unscaled_norms.infinity_norm),$(unscaled_norms.frobenius_norm),$(unscaled_norms.jacobian)",
            )
        end


    else  # Load norms from file
        println("Loading norms for $(filename)")
        file = open(base * filename * "/scaled_norm.csv", "r")
        loaded = readline(file)
        s, i, f, j = split(loaded, ",")
        close(file)

        scaled_norms = norm_information(
            parse(Float64, s),
            parse(Float64, i),
            parse(Float64, f),
            parse(Float64, j),
        )

        file = open(base * filename * "/unscaled_norm.csv", "r")
        loaded = readline(file)
        s, i, f, j = split(loaded, ",")
        close(file)

        unscaled_norms = norm_information(
            parse(Float64, s),
            parse(Float64, i),
            parse(Float64, f),
            parse(Float64, j),
        )

        close(file)
    end
    println("Loading RHS for $(filename)")

    b .= b[permutation]
    b_scaled = A_Scaler * b

    seed .= seed[permutation]
    seed_scaled = A_Unscaler * seed

    println("Loaded RHS for $(filename)")

    return problem_interface(
        package(A_, b, seed, filename_stripped, unscaled_norms, A_SDD, 0),
        package(
            A_Scaled,
            b_scaled,
            seed_scaled,
            filename_stripped,
            scaled_norms,
            A_Scaled_SDD,
            0,
        ),
        package(
            A_Scaled_Padded,
            b_scaled,
            seed_scaled,
            filename_stripped,
            scaled_norms,
            true,
            0,
        ),
        reordering,
        :SPD_matrix,
        filename_stripped,
    )
end


"""
    extract_negative(A::SparseMatrixCSC{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

Extracts the negative elements from the given sparse matrix `A`.

# Arguments
- `A::SparseMatrixCSC{Tv, Ti}`: The input sparse matrix.

# Returns
- `SparseMatrixCSC{Tv, Ti}`: A sparse matrix containing only the negative elements of `A`, with the same dimensions as `A`.

# Notes
- This function identifies and isolates negative values while preserving their original positions in the matrix.
- The resulting matrix has zeros in place of all non-negative elements.
"""
function extract_negative(A::SparseMatrixCSC{Tv,Ti})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}

    A = SparseMatrixCSC(A)

    m, n = size(A)
    rows, cols, vals = findnz(A)

    neg_indices = findall(x -> x < 0, vals)
    neg_vals = vals[neg_indices]
    neg_rows = rows[neg_indices]
    neg_cols = cols[neg_indices]

    return sparse(neg_rows, neg_cols, neg_vals, m, n)
end


"""
    make_augmented(A::SparseMatrixCSC{Tv, Ti}, b::Vector{Tv}) -> Tuple{SparseMatrixCSC{Tv, Ti}, Vector{Tv}}

Constructs an augmented system from a symmetric diagonally dominant (SDD) matrix `A`, producing a graph-Laplacian of double the size.

# Arguments
- `A::SparseMatrixCSC{Tv, Ti}`: The input symmetric diagonally dominant (SDD) matrix.
- `b::Vector{Tv}`: The right-hand side vector.

# Returns
- `Tuple{SparseMatrixCSC{Tv, Ti}, Vector{Tv}}`: A tuple containing:
    - The augmented matrix, which is a graph-Laplacian of size `2m × 2m` (where `m` is the size of `A`).
    - The augmented right-hand side vector of length `2m`.

# Notes
- The function decomposes `A` into its positive (`Ap`), negative (`An`), and diagonal (`D1`, `D2`) components.
- `Ap` contains only the positive off-diagonal values, while `An` contains only the negative off-diagonal values.
- `D1` is constructed to ensure the augmented system maintains Laplacian properties.
- The final augmented matrix is assembled as:
    ```[ D1 + (D2 / 2) + An   - (D2 / 2) - Ap ]
    [ - (D2 / 2) - Ap      D1 + (D2 / 2) + An ]```
- The augmented right-hand side is formed as `[b; -b]`, preserving solution consistency.
"""
function make_augmented(
    A::SparseMatrixCSC{Tv,Ti},
    b::Vector{Tv},
)::Tuple{SparseMatrixCSC{Tv,Ti},Vector{Tv}} where {Tv,Ti}

    A = SparseMatrixCSC(copy(A))

    diagonal = diag(A) |> spdiagm

    m, _ = size(A)

    e = ones(m)

    An = extract_negative(A)

    for i in An.nzval
        if i > 0
            throw("An is not negative")
        end
    end

    A = convert(SparseMatrixCSC{Float64,Int64}, A)
    An = convert(SparseMatrixCSC{Float64,Int64}, An)
    diagonal = convert(SparseMatrixCSC{Float64,Int64}, diagonal)

    Ap = A - An - diagonal

    for i in Ap.nzval
        if i < 0
            throw("Ap is not positive")
        end
    end

    D1 = (Ap * e - An * e) |> spdiagm

    D2 = A - Ap - An - D1

    for i in diag(D2)
        if i < -eps(1.0)
            print(i)
            throw("Diag D2 is not positive")
        end
    end

    for i in D2.nzval
        if i < -eps(1.0)
            throw("D2 is not positive")
        end
    end

    augmented_system = [(D1+(D2./2)+An) (-(D2 ./ 2)-Ap); (-(D2 ./ 2)-Ap) (D1+(D2./2)+An)]

    augmented_b = [b; -b]

    return augmented_system, augmented_b
end

"""
    laplace_to_adj(A::SparseMatrixCSC{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

Converts a Laplacian matrix `A` to its corresponding adjacency matrix.

# Arguments
- `A::SparseMatrixCSC{Tv, Ti}`: The Laplacian matrix to be converted.

# Returns
- `SparseMatrixCSC{Tv, Ti}`: The adjacency matrix corresponding to `A`.

# Notes
- The adjacency matrix is computed as `B = D - A`, where `D` is the diagonal degree matrix of `A`.
- This conversion assumes `A` is a valid Laplacian, meaning its diagonal entries represent node degrees, and off-diagonal entries represent negative edge weights.
- The resulting adjacency matrix is symmetric and retains the sparsity pattern of `A`.
"""
function laplace_to_adj(A::SparseMatrixCSC{Tv,Ti})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}

    m, n = size(A)

    diagonal = diag(A)

    B = spdiagm(diagonal) - A

    return B

end

"""
    is_laplacian(A::SparseMatrixCSC{Tv, Ti}) -> Bool

Checks whether the given sparse matrix `A` is a valid graph Laplacian.

# Arguments
- `A::SparseMatrixCSC{Tv, Ti}`: The matrix to check.

# Returns
- `Bool`: `true` if `A` satisfies the properties of a Laplacian matrix, otherwise `false`.

# Notes
- A Laplacian matrix `L` is defined as `L = D - A`, where:
  - `D` is the diagonal degree matrix.
  - `A` is the adjacency matrix with non-positive off-diagonal elements.
- The function checks that:
  1. All off-diagonal elements are non-positive.
  2. The row sums of `A` (negated) match the diagonal entries of `A`.
- If any off-diagonal element is positive, the function returns `false`.
"""
function is_laplacian(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

    diagonal = convert(SparseMatrixCSC{Tv,Ti}, diag(A) |> spdiagm)

    m, _ = size(A)

    B = A - diagonal
    B = convert(SparseMatrixCSC{Tv,Ti}, B)

    e = ones(m)
    e = convert(Vector{Tv}, e)

    D = -sum(B, dims = 2)

    # println(D .≈ diag(A))

    for i in B.nzval
        if i > 0
            return false
        end
    end

    return D ≈ diag(A)

end

"""
    is_sdd(B::SparseMatrixCSC) -> Bool

Checks whether the given sparse matrix `B` is symmetric and diagonally dominant (SDD).

# Arguments
- `B::SparseMatrixCSC`: The sparse matrix to check.

# Returns
- `Bool`: `true` if `B` is symmetric and diagonally dominant, otherwise `false`.

# Notes
- A matrix `B` is considered **symmetric diagonally dominant (SDD)** if:
  - It is symmetric, meaning `B == B'`.
  - For every row `i`, the diagonal entry satisfies `2 * B[i, i] ≥ sum(abs, B[:, i])`.
- The function iterates over all columns to verify diagonal dominance.
- If any row violates the SDD condition, the function returns `false` immediately.
"""
function is_sdd(B::SparseMatrixCSC)::Bool

    for i in axes(B, 2)
        if sum(abs, view(B, :, i)) <= 2 * B[i, i]
            return false
        end
    end

    return true
end

end
