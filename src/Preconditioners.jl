module Preconditioners

using ..Algorithms: sparse_triangular_solve
using ..Utilities: norm_information, make_augmented, laplace_to_adj, problem_interface, package

using SparseArrays
using LinearAlgebra
using Statistics

import Laplacians
import CombinatorialMultigrid
using LDLFactorizations

import RandomizedPreconditioners
import LimitedLDLFactorizations

import CondaPkg
import PythonCall
import MATLAB

using CSV
using DataFrames

export Preconditioner  # Preconditioner
export Control  # Identity
export TruncatedNeumann  # Truncated Neumann
export SOR, SSOR, GaussSeidel, SymmetricGaussSeidel  # Iterative methods
export IncompleteCholesky, ModifiedIncompleteCholesky  # Incomplete Cholesky (MATLAB)
export SuperILU, SuperLLT  # SuperLU 
export SymmetricSPAI  # Saunders (MATLAB)
export AMG_rube_stuben, AMG_smoothed_aggregation  # PyAmg
export LaplaceStripped # Laplacians.jl
export CombinatorialMG  # CombinatorialMultigrid.jl
export SteepestDescent  # Basic iterative method
export RandomizedNystrom, LimitedLDL
export conda_install_dependencies
 
# Path to the dependencies directory, used for locating external scripts and data files.
const DEP_DIR = joinpath(@__DIR__, "..", "DEPENDENCIES")

"""
    Preconditioner(LinearOperator::Function, num_multiplications::UInt64, system::package)

Represents a preconditioner for an iterative solver.

# Arguments
- `LinearOperator::Function`: A function that applies the preconditioner to a given vector.
- `num_multiplications::UInt64`: The estimated number of nonzero multiplications performed by the preconditioner.
- `generation_work::Float64`: An estimate of the computational work required to generate the preconditioner. If unknown, use `Inf`.
- `system::package`: The system associated with the preconditioner, including the matrix and problem metadata.

# Notes
- The `LinearOperator` function takes a solution vector and a right-hand side vector as inputs and modifies the solution in place.
- `num_multiplications` provides a measure of the computational cost of applying the preconditioner.
- `generation_work::Float64` provides an estimate of the computational work required to generate the preconditioner. Inf is used when unknown.
- This struct is used to encapsulate different types of preconditioners while maintaining a consistent interface for iterative solvers.
"""
struct Preconditioner
    LinearOperator::Function
    num_multiplications::UInt64
    generation_work::Float64
    system::package
end

function conda_install_dependencies()
    CondaPkg.add("scipy")
    CondaPkg.add("PyAMG")
end

"""
    SOR(mat::problem_interface, ω::Float64, num_iters::Int64) -> Preconditioner

Constructs a Successive Over-Relaxation (SOR) preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `ω::Float64`: The relaxation parameter. If set to `-1`, an optimal value is computed based on the Jacobi spectral radius.
- `num_iters::Int64`: The number of SOR iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the SOR preconditioner for iterative solvers.

# Notes
- This function applies SOR on the `Scaled` version of the problem interface.
- If `ω < 0`, an automatic computation of `ω` is performed based on the spectral properties of the matrix.
- The preconditioner modifies an initial guess iteratively to approximate the solution.
"""
function SOR(mat::problem_interface, ω::Float64, num_iters::Int64)
    return SOR(mat.Scaled, ω, num_iters)
end


"""
    SOR(input::package, ω::Float64, num_iters::Int64) -> Preconditioner

Constructs a Successive Over-Relaxation (SOR) preconditioner for the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `ω::Float64`: The relaxation parameter. If `ω < 0`, it is automatically computed based on the Jacobi spectral radius.
- `num_iters::Int64`: The number of SOR iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the SOR preconditioner for iterative solvers.

# Notes
- The SOR iteration updates the solution in-place using a forward sweep through the rows of `A`.
- The function extracts the compressed sparse column (CSC) representation of `A` for efficient traversal.
- The relaxation parameter `ω` is computed if negative, using an expression related to the spectral radius of the Jacobi iteration matrix.
- The preconditioner applies `num_iters` sweeps over the system, modifying the initial guess iteratively.
- The number of multiplications is estimated as `(nnz(A) + 2 * size(A, 1)) * num_iters`, accounting for all updates.
"""
function SOR(input::package, ω::Float64, num_iters::Int64)
    
    ω = ω < 0. ? 1.0 + (Float64(input.norms.jacobian) / (1.0 + sqrt(1.0 - Float64(input.norms.jacobian)^2)))^2 : ω

    ψ = 1. - ω

    matrix = input.A

    sz = size(matrix, 1)

    colptr = matrix.colptr
    rowval = matrix.rowval
    nzval = matrix.nzval

    function LinearOperator(StartingGuess, b)

        StartingGuess .= 0.
        
        for _ in 1:num_iters
            for i in 1:sz
                row_sum = 0.
                ii_val = 0.
                for j in colptr[i]:colptr[i+1]-1
                    idx = rowval[j]
                    if idx == i
                        ii_val = nzval[j]
                    else
                        row_sum += nzval[j] * StartingGuess[idx]
                    end
                end
                StartingGuess[i] = ψ * StartingGuess[i] + (ω * (b[i] - row_sum)) / ii_val
            end
        end
    end
    num_multiplications = (nnz(matrix) + 2 * sz) * num_iters
    println("Created SOR Preconditioner for $(input.name) with ω = $ω and $num_iters iterations.")
    return Preconditioner(LinearOperator, num_multiplications, 0., input)
end


"""
    SteepestDescent(mat::problem_interface, num_iters::Int64) -> Preconditioner

Constructs a Steepest Descent preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `num_iters::Int64`: The number of iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the Steepest Descent preconditioner.

# Notes
- This function applies the Steepest Descent method on the `Scaled` version of the problem.
- Iteratively updates the solution in the direction of the negative residual, with an adaptive step size.
- Convergence depends on the conditioning of the system matrix.
"""
function SteepestDescent(mat::problem_interface, num_iters::Int64)
    return SteepestDescent(mat.Scaled, num_iters)
end


"""
    SteepestDescent(input::package, num_iters::Int64) -> Preconditioner

Constructs a Steepest Descent preconditioner for the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `num_iters::Int64`: The number of iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the Steepest Descent preconditioner.

# Notes
- The Steepest Descent method iteratively updates the solution in the direction of the negative residual.
- The step size `α` is computed adaptively as `(r' * r) / (p' * r)`, where `r` is the residual and `p = A'r` is the projected residual.
- The preconditioner modifies an initial guess iteratively to approximate the solution.
- The number of multiplications is estimated as `nnz(A) * num_iters + 4 * size(A, 1) * num_iters - size(A, 1)`.
"""
function SteepestDescent(input::package, num_iters::Int64)

    r = zeros(size(input.A, 1))
    p = zeros(size(input.A, 1))

    function LinearOperator(x, b)
        x .= 0
        r .= b
        for _ in 1:num_iters-1
            mul!(p, input.A', r)
            α = (r' * r) / (p' * r)
            x .+= α .* r
            r .-= α .* p
        end
        mul!(p, input.A', r)
        α = (r' * r) / (p' * r)
        x .+= α .* r
    end

    num_multiplications = nnz(input.A) * num_iters + 4 * size(input.A, 1) * num_iters - size(input.A, 1)

    println("Created Steepest Descent Preconditioner for $(input.name) with $num_iters iterations.")

    return Preconditioner(LinearOperator, num_multiplications, 0., input)
end


"""
    SSOR(mat::problem_interface, ω::Float64, num_iters::Int64) -> Preconditioner

Constructs a Symmetric Successive Over-Relaxation (SSOR) preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `ω::Float64`: The relaxation parameter. If set to `-1`, an optimal value is computed based on the Jacobi spectral radius.
- `num_iters::Int64`: The number of SSOR iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the SSOR preconditioner for iterative solvers.

# Notes
- This function applies SSOR on the `Scaled` version of the problem interface.
- The SSOR method consists of a forward sweep followed by a backward sweep to ensure symmetry.
- If `ω < 0`, an automatic computation of `ω` is performed based on the spectral properties of the matrix.
- The preconditioner modifies an initial guess iteratively to approximate the solution.
"""
function SSOR(mat::problem_interface, ω::Float64, num_iters::Int64)
    return SSOR(mat.Scaled, ω, num_iters)
end


"""
    SSOR(input::package, ω::Float64, num_iters::Int64) -> Preconditioner

Constructs a Symmetric Successive Over-Relaxation (SSOR) preconditioner for the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `ω::Float64`: The relaxation parameter. If `ω == -1`, it is automatically computed based on the Jacobi spectral radius.
- `num_iters::Int64`: The number of SSOR iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the SSOR preconditioner for iterative solvers.

# Notes
- The SSOR method consists of a forward sweep followed by a backward sweep to enforce symmetry.
- If `ω < 0`, the relaxation parameter is computed using an expression related to the spectral radius of the Jacobi iteration matrix.
- The function extracts the compressed sparse column (CSC) representation of `A` for efficient traversal.
- Iteratively modifies the solution estimate using `num_iters` sweeps.
- The total number of multiplications is estimated as `2 * (nnz(A) + 2 * size(A, 1)) * num_iters`, accounting for both forward and backward passes.
"""
function SSOR(input::package, ω::Float64, num_iters::Int64)

    ω = ω == -1. ? 1.0 + (Float64(input.norms.jacobian) / (1.0 + sqrt(1.0 - Float64(input.norms.jacobian)^2)))^2 : ω

    ψ = 1. - ω

    matrix = input.A

    sz = size(matrix, 1)

    colptr = matrix.colptr
    rowval = matrix.rowval
    nzval = matrix.nzval
    
    function LinearOperator(StartingGuess, b)
        StartingGuess .= 0
        @inbounds for _ in 1:num_iters
            for i in 1:sz
                row_sum = 0.
                ii_val = 0.
                for j in colptr[i]:colptr[i+1]-1
                    idx = rowval[j]
                    if idx == i
                        ii_val = nzval[j]
                    else
                        row_sum += nzval[j] * StartingGuess[idx]
                    end
                end
                StartingGuess[i] = ψ * StartingGuess[i] + (ω * (b[i] - row_sum)) / ii_val
            end
            for i in sz:-1:1
                row_sum = 0.
                ii_val = 0.
                for j in colptr[i]:colptr[i+1]-1
                    idx = rowval[j]
                    if idx == i
                        ii_val = nzval[j]
                    else
                        row_sum += nzval[j] * StartingGuess[idx]
                    end
                end
                StartingGuess[i] = ψ * StartingGuess[i] + (ω * (b[i] - row_sum)) / ii_val
            end
        end
    end
    num_multiplications = 2 * (nnz(matrix) + 2 * sz) * num_iters
    println("Created SSOR Preconditioner for $(input.name) with ω = $ω and $num_iters iterations.")
    return Preconditioner(LinearOperator, num_multiplications, 0., input)
end


"""
    GaussSeidel(mat::problem_interface, num_iters::Int64) -> Preconditioner

Constructs a Gauss-Seidel preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `num_iters::Int64`: The number of iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the Gauss-Seidel preconditioner.

# Notes
- This function applies Gauss-Seidel on the `Scaled` version of the problem interface.
- Gauss-Seidel iteratively updates the solution by performing a forward sweep through the rows of `A`.
- Convergence depends on the spectral properties of the system matrix.
"""
function GaussSeidel(mat::problem_interface, num_iters::Int64)
    return GaussSeidel(mat.Scaled, num_iters)
end


"""
    GaussSeidel(input::package, num_iters::Int64) -> Preconditioner

Constructs a Gauss-Seidel preconditioner for the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `num_iters::Int64`: The number of Gauss-Seidel iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the Gauss-Seidel preconditioner for iterative solvers.

# Notes
- The Gauss-Seidel method iteratively updates the solution using a forward sweep through the rows of `A`.
- Each iteration modifies the solution `x` in place based on the equation:
    `xᵢ = (bᵢ - Σ Aᵢⱼ xⱼ) / Aᵢᵢ`
    where `Σ` sums over all `j ≠ i`.
- The function extracts the compressed sparse column (CSC) representation of `A` for efficient traversal.
- The preconditioner applies `num_iters` forward sweeps over the system.
- The total number of multiplications is estimated as `nnz(A) * num_iters`.
"""
function GaussSeidel(input::package, num_iters::Int64)

    matrix = input.A

    sz = size(matrix, 1)

    colptr = matrix.colptr
    rowval = matrix.rowval
    nzval = matrix.nzval

    function LinearOperator(StartingGuess, b)
        StartingGuess .= 0.
        @inbounds  for _ in 1:num_iters
            for i in 1:sz
                row_sum = 0.
                ii_val = 0.
                for j in colptr[i]:colptr[i+1]-1
                    idx = rowval[j]
                    if idx == i
                        ii_val = nzval[j]
                    else
                        row_sum += nzval[j] * StartingGuess[idx]
                    end
                end
                StartingGuess[i] =  (b[i] - row_sum) / ii_val
            end
        end
    end
    num_multiplications = nnz(matrix) * num_iters
    println("Created Gauss-Seidel Preconditioner for $(input.name) with $num_iters iterations.")
    return Preconditioner(LinearOperator, num_multiplications, 0., input)
end


"""
    SymmetricGaussSeidel(mat::problem_interface, num_iters::Int64) -> Preconditioner

Constructs a Symmetric Gauss-Seidel preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `num_iters::Int64`: The number of iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the Symmetric Gauss-Seidel preconditioner.

# Notes
- This function applies Symmetric Gauss-Seidel on the `Scaled` version of the problem interface.
- The method consists of a forward sweep followed by a backward sweep to enforce symmetry.
- Convergence depends on the spectral properties of the system matrix.
"""
function SymmetricGaussSeidel(mat::problem_interface, num_iters::Int64)
    return SymmetricGaussSeidel(mat.Scaled, num_iters)
end


"""
    SymmetricGaussSeidel(input::package, num_iters::Int64) -> Preconditioner

Constructs a Symmetric Gauss-Seidel preconditioner for the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `num_iters::Int64`: The number of iterations to perform.

# Returns
- `Preconditioner`: A struct encapsulating the Symmetric Gauss-Seidel preconditioner.

# Notes
- The Symmetric Gauss-Seidel method consists of a forward sweep followed by a backward sweep to enforce symmetry.
- Each iteration modifies the solution `x` using:
    `xᵢ = (bᵢ - Σ Aᵢⱼ xⱼ) / Aᵢᵢ`
where `Σ` sums over all `j ≠ i`.
- The function extracts the compressed sparse column (CSC) representation of `A` for efficient traversal.
- Applies `num_iters` symmetric sweeps over the system.
- The total number of multiplications is estimated as `2 * nnz(A) * num_iters`, accounting for both forward and backward passes.
"""

function SymmetricGaussSeidel(input::package, num_iters::Int64)

    matrix = input.A

    colptr = matrix.colptr
    rowval = matrix.rowval
    nzval = matrix.nzval

    sz = size(matrix, 1)    
    function LinearOperator(StartingGuess, b)
        StartingGuess .= 0.
        @inbounds for _ in 1:num_iters
            for i in 1:sz
                row_sum = 0.
                ii_val = 0.
                for j in colptr[i]:colptr[i+1]-1
                    idx = rowval[j]
                    if idx == i
                        ii_val = nzval[j]
                    else
                        row_sum += nzval[j] * StartingGuess[idx]
                    end
                end
                StartingGuess[i] = (b[i] - row_sum) / ii_val
            end
            for i in sz:-1:1
                row_sum = 0.
                ii_val = 0.
                for j in colptr[i]:colptr[i+1]-1
                    idx = rowval[j]
                    if idx == i
                        ii_val = nzval[j]
                    else
                        row_sum += nzval[j] * StartingGuess[idx]
                    end
                end
                StartingGuess[i] = (b[i] - row_sum) / ii_val
            end
        end
    end
    num_multiplications = 2 * nnz(matrix) * num_iters
    println("Created Symmetric Gauss-Seidel Preconditioner for $(input.name) with $num_iters iterations.")
    return Preconditioner(LinearOperator, num_multiplications, 0., input)
end


"""
    IncompleteCholesky(mat::problem_interface, drop_tol::Float64, fill::Int64, michol::Bool=false) -> Preconditioner

Constructs an Incomplete Cholesky (IC) preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `drop_tol::Float64`: The drop tolerance, controlling the sparsity of the factorization.
- `fill::Int64`: The level of fill-in allowed in the factorization. If `-1`, a no-fill version is used.
- `michol::Bool` (optional, default `false`): If `true`, enables modified incomplete Cholesky (MIC) factorization.

# Returns
- `Preconditioner`: A struct encapsulating the Incomplete Cholesky preconditioner.

# Notes
- This function applies Incomplete Cholesky on the `Scaled` version of the problem interface.
- If `fill == -1`, the factorization is performed without additional fill-in.
- The MIC variant is enabled when `michol` is set to `true`, modifying the factorization to improve numerical stability.
- This function serves as a wrapper for calling the MATLAB-based Incomplete Cholesky implementation.
"""
function IncompleteCholesky(mat::problem_interface, drop_tol::Float64, fill::Int64, michol::Bool=false)
    return IncompleteCholesky(mat.Scaled, drop_tol, fill, michol)
end


"""
    IncompleteCholesky(input::package, drop_tol::Float64, fill::Int64, michol::Bool=false) -> Preconditioner

Constructs an Incomplete Cholesky (IC) preconditioner using MATLAB's `ichol` function.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `drop_tol::Float64`: Drop tolerance controlling the sparsity of the factorization.
- `fill::Int64`: Level of fill-in allowed. If `-1`, a no-fill version is used.
- `michol::Bool` (optional, default `false`): If `true`, enables the modified Incomplete Cholesky (MIC) factorization.

# Returns
- `Preconditioner`: A struct encapsulating the Incomplete Cholesky preconditioner.

# Notes
- This function requires MATLAB and calls `ichol(A, options)`.
- If `fill == -1`, the no-fill variant of `ichol` is used.
- The factorized matrix `u` is retrieved from MATLAB and its transpose `l` is used for solving.
- The preconditioner applies a sparse triangular solve using `l` and `u`.
- The function ensures that the MATLAB session is properly managed, closing it after execution.
"""
function IncompleteCholesky(input::package, drop_tol::Float64, fill::Int64, michol::Bool=false)

    local session
    try
        session = MATLAB.MSession(0)
    catch e
        println("Error creating MATLAB session.")
        rethrow(e)
    end

    try

        MATLAB.put_variable(session, :A, input.A)

        eval_string = ""

        if fill == -1
            eval_string *= "options.type = 'nofill'; "
        else
            eval_string *= "options.type = 'ict'; "
            eval_string *= "options.droptol = $drop_tol; "
        end
        eval_string *= "options.michol = $(michol ? "\'on\'" : "\'off\'");"

        MATLAB.eval_string(session, eval_string)
        MATLAB.eval_string(session, "u = ichol(A, options);")

        u = MATLAB.get_mvariable(session, :u) |> MATLAB.jsparse
        l = copy(transpose(u))

        function LinearOperator(y, r)
            y .= sparse_triangular_solve(l, u, r)
        end
        num_multiplications = 2 * nnz(u)

        # the following is a coarse approximation of generation work. This will be an undercount for thresholded factorizations.
        generation_work = sum(x -> x^2, u.colptr[2:end] .- u.colptr[1:end-1])

        println("Created Incomplete Cholesky Preconditioner for $(input.name).")
        return Preconditioner(LinearOperator, num_multiplications, Float64(generation_work), input)
    catch y
        println("Error creating the Incomplete Cholesky Preconditioner for $(input.name).")
        rethrow(y)
    finally
        MATLAB.close(session)
    end
end

"""
    ModifiedIncompleteCholesky(mat::problem_interface, drop_tol::Float64, fill::Int64) -> Preconditioner

Constructs a Modified Incomplete Cholesky (MIC) preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `drop_tol::Float64`: Drop tolerance controlling the sparsity of the factorization.
- `fill::Int64`: Level of fill-in allowed. If `-1`, a no-fill version is used.

# Returns
- `Preconditioner`: A struct encapsulating the Modified Incomplete Cholesky preconditioner.

# Notes
- This function is a wrapper around `IncompleteCholesky`, enabling the `michol` option.
- The MIC variant improves numerical stability by modifying the diagonal entries during factorization.
- The factorization is applied to the `Scaled` version of the problem.
"""
function ModifiedIncompleteCholesky(mat::problem_interface, drop_tol::Float64, fill::Int64)
    return IncompleteCholesky(mat, drop_tol, fill, true)
end


"""
    SuperILU(mat::problem_interface, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL") -> Preconditioner

Constructs an incomplete LU (ILU) preconditioner using SuperLU, applied to the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `drop_tolerance::Float64`: Drop tolerance controlling the sparsity of the ILU factorization.
- `fill::Int64`: Level of fill-in allowed in the factorization. Must be between `1` and `10`.
- `ordering::String` (optional, default `"NATURAL"`): Column ordering strategy for factorization. Must be one of `"NATURAL"`, `"MMD_ATA"`, or `"MMD_AT_PLUS_A"`.

# Returns
- `Preconditioner`: A struct encapsulating the ILU preconditioner.

# Notes
- This function applies ILU on the `Scaled` version of the problem.
- The preconditioner is constructed using SuperLU's `spilu` function from SciPy.
- The ILU factors returned are **not necessarily symmetric**, even if the input matrix is symmetric.
- The function checks that `fill` is within the allowed range and throws an error otherwise.
- The number of multiplications is estimated as the total number of nonzeros in the ILU factors.
"""
function SuperILU(mat::problem_interface, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL")
    return SuperILU(mat.Scaled, drop_tolerance, fill, ordering)
end


"""
    SuperILU(input::package, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL") -> Preconditioner

Constructs an incomplete LU (ILU) preconditioner using SuperLU, applied to the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `drop_tolerance::Float64`: Drop tolerance controlling the sparsity of the ILU factorization.
- `fill::Int64`: Level of fill-in allowed in the factorization. Must be between `1` and `10`.
- `ordering::String` (optional, default `"NATURAL"`): Column ordering strategy for factorization. Must be one of `"NATURAL"`, `"MMD_ATA"`, or `"MMD_AT_PLUS_A"`.

# Returns
- `Preconditioner`: A struct encapsulating the ILU preconditioner.

# Notes
- This function applies ILU on `input.A` using the SuperLU `spilu` function from SciPy.
- The ILU factorization is controlled by the `drop_tolerance` and `fill` parameters, determining sparsity and accuracy.
- The function ensures `fill` is within the allowed range and raises an error otherwise.
- The matrix is converted into SciPy's compressed sparse column (CSC) format before factorization.
- The preconditioner solves systems using the SuperLU `ilu.solve` function.
- The number of multiplications is estimated as the sum of nonzeros in `L` and `U`, capturing the factorization complexity.
"""
function SuperILU(input::package, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL")

    ordering = uppercase(ordering)

    @assert ordering ∈ ["NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]

    # Assert that fill is between 1 and 10
    if fill < 1 || fill > 10
        throw(ArgumentError("Fill must be between 1 and 10"))
    end
    
    scipy = PythonCall.pyimport("scipy.sparse")
    # Convert the matrix to a python object
    A = scipy.csc_matrix((input.A.nzval, input.A.rowval .- 1, input.A.colptr .- 1), shape=size(input.A))
    dc = PythonCall.pybuiltins.dict
    ilu = scipy.linalg.spilu(A, fill_factor=fill, drop_tol=drop_tolerance, 
        # options=dc(Equil=false,RowPerm="NOROWPERM", ColPerm="MMD_AT_PLUS_A",
        options=dc(Equil=false,RowPerm="NOROWPERM", ColPerm=ordering,
        # ColPerm="MMD_AT_PLUS_A",
        # ColPerm="MMD_AT_PLUS_A",
        # SymmetricMode=true,
        DiagPivotThresh=0.)
        )
    function LinearOperator(y, r)
        y .= ilu.solve(r)
    end
    num_multiplications = ilu.L.nnz + ilu.U.nnz

    # Estimate the generation work as the sum of nonzeros in L and U
    # This is a coarse approximation, as it does not account for thresholding effects.
    generation_work = sum(x -> x, (ilu.L.colptr[2:end] .- ilu.L.colptr[1:end-1]) .* (ilu.U.colptr[2:end] .- ilu.U.colptr[1:end-1]))
    
    println("Created SuperLU Preconditioner for $(input.name) with fill factor $fill.")
    return Preconditioner(LinearOperator, num_multiplications, Float64(generation_work), input)
end


"""
    SuperLLT(mat::problem_interface, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL") -> Preconditioner

Constructs an incomplete Cholesky-like factorization (`LLᵀ`) preconditioner using SuperLU, applied to the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `drop_tolerance::Float64`: Drop tolerance controlling the sparsity of the factorization.
- `fill::Int64`: Level of fill-in allowed in the factorization. Must be between `1` and `10`.
- `ordering::String` (optional, default `"NATURAL"`): Column ordering strategy for factorization. Must be one of `"NATURAL"`, `"MMD_ATA"`, or `"MMD_AT_PLUS_A"`.

# Returns
- `Preconditioner`: A struct encapsulating the `LLᵀ` preconditioner.

# Notes
- This function applies the factorization on the `Scaled` version of the problem.
- The factorization is computed using SuperLU's `spilu` function, but only the `L^T` factor is used for preconditioning, which results in a symmetric preconditioner.
- The resulting factors are **not necessarily symmetric**, even if the input matrix is symmetric.
- The function checks that `fill` is within the allowed range and throws an error otherwise.
"""
function SuperLLT(mat::problem_interface, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL")
    return SuperLLT(mat.Scaled, drop_tolerance, fill, ordering)
end

"""
    SuperLLT(input::package, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL") -> Preconditioner

Constructs an incomplete Cholesky-like factorization (`LLᵀ`) preconditioner using SuperLU, applied to the given system.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `drop_tolerance::Float64`: Drop tolerance controlling the sparsity of the factorization.
- `fill::Int64`: Level of fill-in allowed in the factorization. Must be between `1` and `10`.
- `ordering::String` (optional, default `"NATURAL"`): Column ordering strategy for factorization. Must be one of `"NATURAL"`, `"MMD_ATA"`, or `"MMD_AT_PLUS_A"`.

# Returns
- `Preconditioner`: A struct encapsulating the `LLᵀ` preconditioner.

# Notes
- This function applies an incomplete factorization on `input.A` using SuperLU's `spilu` function.
- The ILU factorization is computed with `SymmetricMode=true`, but **the resulting factors are not necessarily symmetric**, even if `A` is symmetric.
- The `L` factor is used for preconditioning, scaled by the diagonal of `U`, resulting in a symmetric preconditioner.
- The function ensures `fill` is within the allowed range and raises an error otherwise.
- The preconditioner solves systems using a sparse triangular solve with `L'` and `L^T`.
- The total number of multiplications is estimated as `2 * nnz(U)`, accounting for forward and backward solves.
"""
function SuperLLT(input::package, drop_tolerance::Float64, fill::Int64, ordering::String="NATURAL")

    ordering = uppercase(ordering)

    @assert ordering ∈ ["NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]

    # Assert that fill is between 1 and 10
    if fill < 1 || fill > 10
        throw(ArgumentError("Fill must be between 1 and 10"))
    end

    scipy = PythonCall.pyimport("scipy.sparse")
    # Convert the matrix to a python object
    A = scipy.csc_matrix((input.A.nzval, input.A.rowval .- 1, input.A.colptr .- 1), shape=size(input.A))
    dc = PythonCall.pybuiltins.dict
    ilu = scipy.linalg.spilu(A, fill_factor=fill, drop_tol=drop_tolerance, 
        # options=dc(Equil=false,RowPerm="NOROWPERM", ColPerm="MMD_AT_PLUS_A",
        options=dc(
            Equil=false,
            RowPerm="NOROWPERM", 
            ColPerm="NATURAL",
        # ColPerm="MMD_AT_PLUS_A",
        # ColPerm="MMD_AT_PLUS_A",
        SymmetricMode=true,
        DiagPivotThresh=0.)
    )

    # lp = ilu.perm_r .+ 1
    # rp = ilu.perm_c .+ 1

    U_ = ilu.U
    L_ = ilu.L

    U = SparseMatrixCSC(U_.shape..., U_.indptr .+ 1, U_.indices .+ 1, U_.data);
    L = SparseMatrixCSC(L_.shape..., L_.indptr .+ 1, L_.indices .+ 1, L_.data);

    # We can use U too, but L appears to work better in practice.
    # To use U, we need to remove half of the scaling.
    # dia = 1 ./ sqrt.(diag(U))
    # U = Diagonal(dia) * U

    # To use L, we add half of the scaling from U.
    dia = sqrt.(diag(U))
    L = L * Diagonal(dia)

    function LinearOperator(y, r)  # recall we use the transpose here
        y .= sparse_triangular_solve(copy(L'), copy(L), r)
    end
    num_multiplications = 2 * ilu.U.nnz

    generation_work = sum(x -> x, (L.colptr[2:end] .- L.colptr[1:end-1]) .* (U.colptr[2:end] .- U.colptr[1:end-1]))

    println("Created SuperLU Preconditioner for $(input.name) with fill factor $fill.")
    return Preconditioner(LinearOperator, num_multiplications, Float64(generation_work), input)
end


"""
    TruncatedNeumann(mat::problem_interface, num_iters::Int64, norm::Float64) -> Preconditioner

Constructs a Truncated Neumann series preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `num_iters::Int64`: The number of iterations for the truncated Neumann series expansion.
- `norm_::Float64`: The norm used for scaling. If `0`, the spectral norm is estimated and used.

# Returns
- `Preconditioner`: A struct encapsulating the Truncated Neumann preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- The preconditioner approximates the inverse of `A` using a truncated Neumann series expansion:
    `M⁻¹ ≈ α (I - αA + α²A² - … + αⁿAⁿ)`
    where `α` is chosen as `1/‖A‖` for stability.
- If `norm == 0`, `α` is computed as `1`, otherwise `α = 1 / norm`.
- The number of multiplications is estimated as `num_iters * nnz(A)`, accounting for all matrix-vector products.
"""
function TruncatedNeumann(mat::problem_interface, num_iters::Int64, norm_::Float64)
    return TruncatedNeumann(mat.Scaled, num_iters, norm_)
end


"""
    TruncatedNeumann(mat::problem_interface, num_iters::Int64, norm::Float64) -> Preconditioner

Constructs a Truncated Neumann series preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `num_iters::Int64`: The number of iterations for the truncated Neumann series expansion.
- `norm_::Float64`: The norm used for scaling. If `0`, the spectral norm is estimated and used.

# Returns
- `Preconditioner`: A struct encapsulating the Truncated Neumann preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- The preconditioner approximates the inverse of `A` using a truncated Neumann series expansion:
    `M⁻¹ ≈ α (I - αA + α²A² - … + αⁿAⁿ)`
where `α` is chosen as `1/‖A‖` for stability.
- If `norm == 0`, the spectral norm is estimated, otherwise `α = 1 / norm`.
- The function performs iterative matrix-vector multiplications to construct the preconditioner.
- The total number of multiplications is estimated as `num_iters * nnz(A)`, accounting for all matrix-vector products.
"""
function TruncatedNeumann(input::package, num_iters::Int64, norm_::Float64)

    β = norm_ == -1.0 ? 0.5 * input.norms.spectral_norm : norm_ == 0.0 ? 1.0 : norm(input.A, norm_)
    α = 1.0 / β

    w = zeros(size(input.A, 1))
    temp = zeros(size(input.A, 1))

    matrix = α * input.A

    function LinearOperator(product, x)
        w .= x
        product .= w
        for _ in 1:num_iters
            mul!(temp, matrix, product)
            product .= w .+ product .- temp
        end
        product .*= α
    end
    num_multiplications = num_iters * nnz(input.A)
    println("Created Truncated Neumann Preconditioner for $(input.name) with $num_iters iterations.")
    return Preconditioner(LinearOperator, num_multiplications, 0., input)
end


"""
    SymmetricSPAI(mat::problem_interface, fill_factor::Float64) -> Preconditioner

Constructs a Symmetric Sparse Approximate Inverse (SPAI) preconditioner using the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `fill_factor::Float64`: The fill factor controlling the sparsity of the approximate inverse.

# Returns
- `Preconditioner`: A struct encapsulating the Symmetric SPAI preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- The preconditioner is computed using MATLAB's `SSAI3` function, requiring MATLAB support.
- The `fill_factor` determines the trade-off between accuracy and sparsity.
"""
function SymmetricSPAI(mat::problem_interface, fill_factor::Float64)
    return SymmetricSPAI(mat.Scaled, fill_factor)
end


"""
    SymmetricSPAI(input::package, fill_factor::Float64) -> Preconditioner

Constructs a Symmetric Sparse Approximate Inverse (SPAI) preconditioner using MATLAB's `SSAI3` function.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `fill_factor::Float64`: The fill factor controlling the sparsity of the approximate inverse.

# Returns
- `Preconditioner`: A struct encapsulating the Symmetric SPAI preconditioner.

# Notes
- This function applies the preconditioner to `input.A` and requires MATLAB support.
- The preconditioner is computed using the `SSAI3` function from MATLAB, which approximates `A⁻¹`.
- The `fill_factor` determines the trade-off between sparsity and accuracy.
- The preconditioner applies `M' * r` as an approximate inverse application.
- The MATLAB session is managed internally, ensuring cleanup upon failure.
"""
function SymmetricSPAI(input::package, fill_factor::Float64)
    
    local session
    try
        session = MATLAB.MSession(0)
    catch e
        println("Error creating MATLAB session.")
        rethrow(e)
    end

    try

        MATLAB.eval_string(session, "addpath(\"$(DEP_DIR)\")")
        
        MATLAB.put_variable(session, :sz, Float64(size(input.A, 1)))
        MATLAB.put_variable(session, :fl, ceil(fill_factor * Float64(ceil(nnz(input.A) / size(input.A, 1)))))
        MATLAB.put_variable(session, :A, input.A)

        MATLAB.eval_string(session, "[M, generation_cost] = SSAI3(A, sz, fl)")
        M = MATLAB.get_mvariable(session, :M) |> MATLAB.jsparse
        generation_work = MATLAB.get_mvariable(session, :generation_cost) |> MATLAB.jscalar

        println(generation_work)

        function LinearOperator(y, r)
            mul!(y, M', r)
        end

        num_multiplications = nnz(M)
        
        println("Created Symmetric Sparse Approximate Inverse Preconditioner for $(input.name).")
        return Preconditioner(LinearOperator, num_multiplications, Float64(generation_work), input)
    catch y
        println("Error creating the Symmetric Sparse Approximate Inverse Preconditioner for $(input.name).")
        rethrow(y)
    finally
        MATLAB.close(session)
    end
end


"""
    AMG_rube_stuben(mat::problem_interface, amg_cycle::Char, num_cycles::Int64) -> Preconditioner

Constructs an Algebraic Multigrid (AMG) preconditioner using the Ruge-Stuben method from PyAMG, applied to the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `amg_cycle::Char`: The type of multigrid cycle (`'V'`, `'W'`, etc.).
- `num_cycles::Int64`: The number of cycles to apply.

# Returns
- `Preconditioner`: A struct encapsulating the AMG Ruge-Stuben preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- This is a wrapper around PyAMG's Ruge-Stuben AMG implementation.
"""
function AMG_rube_stuben(mat::problem_interface, amg_cycle::Char, num_cycles::Int64)
    return AMG_rube_stuben(mat, string(amg_cycle), num_cycles)
end


"""
    AMG_rube_stuben(mat::problem_interface, amg_cycle::Char, num_cycles::Int64) -> Preconditioner

Constructs an Algebraic Multigrid (AMG) preconditioner using the Ruge-Stuben method from PyAMG, applied to the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `amg_cycle::String`: The type of multigrid cycle (`'V'`, `'W'`, etc.).
- `num_cycles::Int64`: The number of cycles to apply.

# Returns
- `Preconditioner`: A struct encapsulating the AMG Ruge-Stuben preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- This is a wrapper around PyAMG's Ruge-Stuben AMG implementation.
"""
function AMG_rube_stuben(mat::problem_interface, amg_cycle::String, num_cycles::Int64)
    return AMG_rube_stuben(mat.Scaled, string(amg_cycle), num_cycles)
end


"""
    AMG_rube_stuben(input::package, amg_cycle::String, num_cycles::Int64) -> Preconditioner

Constructs an Algebraic Multigrid (AMG) preconditioner using the Ruge-Stuben method from PyAMG.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `amg_cycle::String`: The type of multigrid cycle (`"V"`, `"W"`, etc.).
- `num_cycles::Int64`: The number of cycles to apply.

# Returns
- `Preconditioner`: A struct encapsulating the AMG Ruge-Stuben preconditioner.

# Notes
- This function wraps PyAMG's `ruge_stuben_solver` for efficient multigrid preconditioning.
- The matrix `A` is converted to a SciPy CSR format before applying the solver.
- The preconditioner solves systems using the specified AMG cycle (`amg_cycle`) for `num_cycles` iterations.
- The number of multiplications is estimated based on the cycle complexity and nonzero entries in the coarse-grid matrix.
- If an error occurs, the function prints an error message and rethrows the exception.
"""
function AMG_rube_stuben(input::package, amg_cycle::String, num_cycles::Int64)
    
    try
        amg_cycle = string(amg_cycle)

        sp = PythonCall.pyimport("scipy.sparse")
        amg = PythonCall.pyimport("pyamg")

        println("this is new one")

        # Because the matrix is symmetric, we can simply pretend it is already in CSR format
        A = sp.csr_matrix((input.A.nzval, input.A.rowval .- 1, input.A.colptr .- 1), shape=size(input.A))

        ml = amg.ruge_stuben_solver(A)

        function LinearOperator(y, x)
            y .= ml.solve(x, maxiter=num_cycles, cycle=amg_cycle, tol=1e-16)
        end
        
        num_multiplications = PythonCall.pyconvert(Int64, PythonCall.pybuiltins.round(ml.cycle_complexity(cycle=amg_cycle) * ml.levels[0].A.nnz)) * num_cycles

        println("Created AMG Ruge-Stuben Preconditioner for $(input.name) with $(num_cycles) cycle$(num_cycles == 1 ? " " : "s ")of type $amg_cycle.")
        return Preconditioner(LinearOperator, num_multiplications, Inf, input)
    catch y
        println("Error creating AMG Ruge-Stuben Preconditioner for $(input.name) with $(num_cycles) cycle$(num_cycles == 1 ? " " : "s ")of type $amg_cycle.")
        rethrow(y)
    end

end


"""
    AMG_smoothed_aggregation(mat::problem_interface, amg_cycle::Char, num_cycles::Int64) -> Preconditioner

Constructs an Algebraic Multigrid (AMG) preconditioner using the Smoothed Aggregation method from PyAMG, applied to the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `amg_cycle::Char`: The type of multigrid cycle (`'V'`, `'W'`, etc.).
- `num_cycles::Int64`: The number of cycles to apply.

# Returns
- `Preconditioner`: A struct encapsulating the AMG Smoothed Aggregation preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- It is a wrapper around PyAMG's Smoothed Aggregation AMG implementation.
- The preconditioner is constructed using PyAMG's `smoothed_aggregation_solver`, which generates a hierarchy of coarse grids for multigrid acceleration.
"""
function AMG_smoothed_aggregation(mat::problem_interface, amg_cycle::Char, num_cycles::Int64)
    return AMG_smoothed_aggregation(mat, string(amg_cycle), num_cycles)
end


"""
    AMG_smoothed_aggregation(mat::problem_interface, amg_cycle::Char, num_cycles::Int64) -> Preconditioner

Constructs an Algebraic Multigrid (AMG) preconditioner using the Smoothed Aggregation method from PyAMG, applied to the scaled version of the given problem.

# Arguments
- `mat::problem_interface`: The problem interface containing different representations of the system.
- `amg_cycle::String`: The type of multigrid cycle (`'V'`, `'W'`, etc.).
- `num_cycles::Int64`: The number of cycles to apply.

# Returns
- `Preconditioner`: A struct encapsulating the AMG Smoothed Aggregation preconditioner.

# Notes
- This function applies the preconditioner to the `Scaled` version of the problem interface.
- It is a wrapper around PyAMG's Smoothed Aggregation AMG implementation.
- The preconditioner is constructed using PyAMG's `smoothed_aggregation_solver`, which generates a hierarchy of coarse grids for multigrid acceleration.
"""
function AMG_smoothed_aggregation(mat::problem_interface, amg_cycle::String, num_cycles::Int64)
    return AMG_smoothed_aggregation(mat.Scaled, string(amg_cycle), num_cycles)
end


"""
    AMG_smoothed_aggregation(input::package, amg_cycle::String, num_cycles::Int64) -> Preconditioner

Constructs an Algebraic Multigrid (AMG) preconditioner using the Smoothed Aggregation method from PyAMG.

# Arguments
- `input::package`: The system to be preconditioned, including the matrix `A` and metadata.
- `amg_cycle::String`: The type of multigrid cycle (`"V"`, `"W"`, etc.).
- `num_cycles::Int64`: The number of cycles to apply.

# Returns
- `Preconditioner`: A struct encapsulating the AMG Smoothed Aggregation preconditioner.

# Notes
- This function wraps PyAMG's `smoothed_aggregation_solver` for efficient multigrid preconditioning.
- The matrix `A` is converted to a SciPy CSR format before applying the solver.
- The preconditioner solves systems using the specified AMG cycle (`amg_cycle`) for `num_cycles` iterations.
- The number of multiplications is estimated based on the cycle complexity and nonzero entries in the coarse-grid matrix.
- If an error occurs, the function prints an error message and rethrows the exception.
"""
function AMG_smoothed_aggregation(input::package, amg_cycle::String, num_cycles::Int64)

    try
        amg_cycle = string(amg_cycle)

        sp = PythonCall.pyimport("scipy.sparse")
        amg = PythonCall.pyimport("pyamg")

        # Because the matrix is symmetric, we can simply pretend it is already in CSR format
        A = sp.csr_matrix((input.A.nzval, input.A.rowval .- 1, input.A.colptr .- 1), shape=size(input.A))

        ml = amg.smoothed_aggregation_solver(A)

        function LinearOperator(y, x)
            y .= ml.solve(x, maxiter=num_cycles, cycle=amg_cycle, tol=1e-16)
        end

        num_multiplications = PythonCall.pyconvert(Int64, PythonCall.pybuiltins.round(ml.cycle_complexity(cycle=amg_cycle) * ml.levels[0].A.nnz)) * num_cycles

        println("Created AMG Smoothed Aggregation Preconditioner for $(input.name) with $(num_cycles) cycle$(num_cycles == 1 ? " " : "s ")of type $amg_cycle.")
        return Preconditioner(LinearOperator, num_multiplications, Inf, input)
    catch y
        println("Error creating AMG Smoothed Aggregation Preconditioner for $(input.name) with $(num_cycles) cycle$(num_cycles == 1 ? " " : "s ")of type $amg_cycle.")
        rethrow(y)
    end
end



function LimitedLDL(input::problem_interface, memory::Integer, droptol::Real)
    return LimitedLDL(input.Scaled, memory, droptol)
end

function LimitedLDL(input::package, memory::Integer, droptol::Real)

    nznum = (nnz(input.A) - size(input.A, 1)) / 2

    nzpercol = nznum / size(input.A, 1)

    memory_val = ceil(Int, nzpercol * memory) 

    try
        
        ldl = LimitedLDLFactorizations.lldl(input.A; P = collect(1:size(input.A, 1)), 
            memory = memory_val, droptol = droptol)

        function LinearOperator(y, r)
            y .= ldl \ r
        end

        num_multiplications = 2 * length(ldl.Lnzvals) + size(input.A, 1)

        println("Created Limited LDL Preconditioner for $(input.name) with memory $memory and droptol $droptol.")

        generation_work = sum(x -> x^2, ldl.colptr[2:end] .- ldl.colptr[1:end-1])

        return Preconditioner(LinearOperator, num_multiplications, Float64(generation_work), input)

    catch
        println("Error creating Limited LDL Preconditioner for $(input.name) with memory $memory and droptol $droptol.")
        rethrow()
    end
end

function RandomizedNystrom(input::problem_interface, rank::Integer, truncation::Integer, μ::Real, dummy_variable::Int=0)

    @assert truncation ≤ rank "Truncation must be less than or equal to rank"

    return RandomizedNystrom(input.Scaled, rank, truncation, μ, dummy_variable)

end

function RandomizedNystrom(input::package, rank::Integer, truncation::Integer, μ::Real, dummy_variable::Int=0)

    @assert truncation ≤ rank "Truncation must be less than or equal to rank"

    try
        rny = RandomizedPreconditioners.NystromSketch(input.A, truncation, rank)

        rnyinv = RandomizedPreconditioners.NystromPreconditionerInverse(rny, μ)

        function LinearOperator(y, r)
            mul!(y, rnyinv, r)
        end

        n = size(input.A, 1)

        num_multiplications = 2*n*truncation + truncation

        generation_cost = nnz(input.A) * rank + 3*n*rank^2 + (rank^3-3)/6 + (n+1)*rank*(rank-1)/2

        println("Created Randomized Nystrom Preconditioner for $(input.name) with rank $rank and truncation $truncation, run $dummy_variable.")

        return Preconditioner(LinearOperator, num_multiplications, Float64(generation_cost), input)

    catch e
        println("Error creating Randomized Nystrom Preconditioner for $(input.name) with rank $rank and truncation $truncation, run $dummy_variable.")
        rethrow(e)
    end

end


"""
    getLapChol(a::SparseMatrixCSC; split=0, merge=0, fixedOrder=false)

Computes an approximate Cholesky factorization for a Laplacian matrix, handling multiple connected components.

# Arguments
- `a::SparseMatrixCSC`: The input Laplacian matrix.
- `split::Integer` (optional, default `0`): The number of levels for recursive splitting in the approximation.
- `merge::Integer` (optional, default `0`): The number of levels for merging in the approximation.
- `fixedOrder::Bool` (optional, default `false`): If `true`, enforces a fixed ordering for the factorization.

# Returns
- `Tuple{Int, Vector{Vector{Int}}, Vector{Function}, Vector{Int}}`:
    - `length(solvers)`: The number of solver instances created.
    - `comps`: A list of connected components in the Laplacian.
    - `solvers`: A list of solver functions, one for each component.
    - `num_mult_list`: A list containing the estimated number of multiplications required for each solver.

# Notes
- If the matrix has multiple connected components, each component is factored separately.
- Small components (size < 50) are solved directly using a dense Cholesky factorization.
- Larger components use an approximate Cholesky factorization from `Laplacians.jl`, which supports recursive splitting and merging.
- The function first identifies connected components using `Laplacians.components(a)`.
- For single-component cases, the entire matrix is factored as one system.
- If `fixedOrder` is `true`, a fixed ordering strategy is used for `Laplacians.LLMatOrd`.
"""
function getLapChol(a::SparseMatrixCSC; split=0, merge=0, fixedOrder=false)

    # This function is derived from Laplacians.jl:
    # Copyright (c) 2015-2016 Daniel A. Spielman and other contributors.
    # Licensed under the MIT "Expat" License.

    # Permission is hereby granted, free of charge, to any person obtaining
    # a copy of this software and associated documentation files (the
    # "Software"), to deal in the Software without restriction, including
    # without limitation the rights to use, copy, modify, merge, publish,
    # distribute, sublicense, and/or sell copies of the Software, and to
    # permit persons to whom the Software is furnished to do so, subject to
    # the following conditions:

    # The above copyright notice and this permission notice shall be
    # included in all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    # CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    local comps = Array{Array{Int,1}}(undef, 0)
    local solvers = Array{Function}(undef, 0)
    local num_mult_list = Array{Int, 1}(undef, 0)

    co = Laplacians.components(a)

    if maximum(co) > 1

        comps = Laplacians.vecToComps(co)
        comps = Laplacians.vecToComps(co)
        nullComps = findall(x -> length(x) == 1, comps)
        hasNullComps = length(nullComps) > 0
        start = 1
        if hasNullComps

            start = 2
            flatNullComps = vcat(comps[nullComps]...)
            comps = vcat([flatNullComps], comps[setdiff(1:end, nullComps)])
            push!(solvers, Laplacians.nullSolver)

        end

        for i in start:length(comps)

            ind = comps[i]
            asub = a[ind, ind]

            if length(ind) < 50

                la_forced = Laplacians.forceLap(asub)
                N = size(la_forced)[1]

                ind_local = findmax(diag(la_forced))[2]
                leave = [1:(ind_local-1);(ind_local+1):N]

                la_forced_sub = la_forced[leave, leave]
 
                ldli = cholesky(la_forced_sub)

                push!(num_mult_list, 2 * nnz(sparse(ldli.L)))

                F(b) = begin
                    bs = b[leave] .- mean(b)


                    xs = ldli \ bs

                    x = zeros(size(b))
                    x[leave] = xs
                    return x

                end

                push!(solvers, F)

            else

                local ldli, llmat

                if split >= 1 && merge < 1
                    llmat = Laplacians.LLmatp(asub, split)
                    ldli = Laplacians.approxChol(llmat, split)
                elseif split >= 1 && merge >= 1
                    llmat = Laplacians.LLmatp(asub, split)
                    ldli = Laplacians.approxChol(llmat, split, merge)
                else
                    llmat = Laplacians.LLmatp(asub)
                    ldli = Laplacians.approxChol(llmat)
                end

                push!(num_mult_list, 2 * length(ldli.fval) + size(ldli.d, 1))

                push!(solvers, b -> Laplacians.LDLsolver(ldli, b))

            end
            
        end

    else

        local ldli = nothing

        if fixedOrder
            if split >= 1 && merge == split
                llmat = Laplacians.LLMatOrd(a,split)
                ldli = Laplacians.approxChol(llmat,split)
            elseif split >= 1 && merge != split
                println("Fixed-order solve with merge != split not implemented here. Abort.")
                return 0, 0, true # bool error 
            else
                llmat = Laplacians.LLMatOrd(a)
                ldli = Laplacians.approxChol(llmat)
            end 
        else
            if split >= 1 && merge < 1
                llmat = Laplacians.LLmatp(a, split)
                ldli = Laplacians.approxChol(llmat, split)
            elseif split >= 1 && merge >= 1
                llmat = Laplacians.LLmatp(a, split)
                ldli = Laplacians.approxChol(llmat,  split, merge)
            else
                llmat = Laplacians.LLmatp(a)
                ldli = Laplacians.approxChol(llmat)
            end 
        end

        subSolver(b) = Laplacians.LDLsolver(ldli, b)

        push!(num_mult_list, 2 * length(ldli.fval) + size(ldli.d, 1))

        push!(solvers, subSolver)
    end
    return length(solvers), comps, solvers, num_mult_list
end


function LaplaceStripped(mat::problem_interface, split::Integer, merge::Integer)
    mat.type == :symmetric_graph && return LaplaceStripped(mat.Original, split, merge) # run on the Laplacian if undirected graph
    mat.Scaled.diagonally_dominant && return LaplaceStripped(mat.Scaled, split, merge, mat.Scaled) # run augmented on scaled if SDD
    mat.Original.diagonally_dominant && return LaplaceStripped(mat.Original, split, merge, mat.Original) # run augmented on original if SDD
    return LaplaceStripped(mat.Special, split, merge, mat.Scaled) # run augmented on padded, apply to scaled
end

function LaplaceStripped(input::package, split::Integer, merge::Integer)

    a = laplace_to_adj(input.A)

    len, components, solvers, num_mult_list = getLapChol(a; split = split, merge = merge)

    solver = solvers[1]

    function LinearOperator(y, r)
        y .= solver(r)
    end

    return Preconditioner(LinearOperator, num_mult_list[1], input)

end


function LaplaceStripped(input::package, split::Integer, merge::Integer, run_on::package)

    A_aug_lap, _ = make_augmented(input.A, input.b)

    local sz = size(input.A, 1)
    local sol_ = zeros(size(A_aug_lap, 1))
    local rhs = zeros(size(A_aug_lap, 1))
    
    a = laplace_to_adj(A_aug_lap)

    len, components, solvers, num_mult_list = getLapChol(a; split = split, merge = merge)

    work_space = [zeros(length(component)) for component in components]

    function LinearOperator(sol, r)

        sol .= r

        view(rhs, 1:sz) .= sol
        view(rhs, sz+1:length(rhs)) .= .-sol

        if len > 1
            for i in 1:len
                work_space[i] .= view(rhs, components[i])
                sol_[components[i]] .= solvers[i](work_space[i])
            end
        else
            sol_ .= solvers[1](rhs)
        end

        sol .= (view(sol_, 1:sz) .- view(sol_, sz+1:length(sol_))) ./ 2
        
    end

    num_multiplications = sum(num_mult_list)

    return Preconditioner(LinearOperator, num_multiplications, run_on)
end


## Combinatorial Multigrid Derivative Code
# This next block of code is a modification of CombinatorialMultigrid to output the work done by the preconditioner.
# The package code must also be modified to output the H, M, W, X variables
# in the CombinatorialMultigrid.cmg_preconditioner_lap function.

# MIT License

# Copyright (c) 2021 Bodhi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# struct Hierarchy
#     A::SparseMatrixCSC{Float64, Int64}
#     invD::Vector{Float64}
#     cI::Vector{Int64}
#     #chol::CholT
#     chol::LDLFactorizations.LDLFactorization{Float64, Int64, Int64, Int64}
# end

# struct Workspace
#     x::Vector{Float64}
#     b::Vector{Float64}
#     tmp::Vector{Float64}
# end

# mutable struct LevelAux
#     fwd::Bool
#     rc::Int32
#     repeat::Int32
#     #
#     sd::Bool
#     islast::Bool
#     iterative::Bool
#     n::Int64
#     nc::Int64
# end

# function preconditioner_i(H, W, X, b::Vector{Float64})
#     level = Int64(1)
#     #@inbounds W[1].b = b
#     # BLAS.blascopy!(length(b),b,1,W[1].b,1)

#     num_mult = 0

#     while level > 0
#         x = W[level].x
#         b = W[level].b
#         tmp = W[level].tmp

#         invD = H[level].invD
#         A = H[level].A
#         cI = H[level].cI

#         if X[level].islast && !X[level].iterative
#             # @inbounds ldiv!(x, H[level].chol,b)
#             num_mult += 2 * length(H[level].chol.Li) + length(H[level].chol.d)
#             level -= 1
#         elseif X[level].islast && X[level].iterative
#             num_mult += length(b)
#             # W[level].x .= b .* invD
#             level -= 1
#         elseif !X[level].islast && X[level].fwd
#             repeat = X[level].repeat   #number of 'recursive' calls

#             if X[level].rc > repeat
#                 X[level].rc = 1
#                 level -= 1
#             else
#                 if X[level].rc == 1
#                     # W[level].x .= b.* invD
#                     num_mult += length(b)
#                 else
#                     mul!(tmp, A, x)
#                     tmp .-= b
#                     # tmp .*= -invD
#                     num_mult += length(b)
#                     W[level].x .+= tmp
#                 end

#                 mul!(tmp, A, x)
#                 tmp .-= b
#                 X[level].fwd = false
#                 # interpolate!(W[level+1].b, cI, -tmp)
#                 level += 1
#             end
#         elseif !X[level].islast && !X[level].fwd
#             z = W[level+1].x
#             W[level].x .+= z[cI]
#             # mul!(tmp, A, x)
#             num_mult += nnz(A)
#             tmp .-= b
#             tmp .*= -invD
#             num_mult += length(b)
#             W[level].x .+= tmp
#             X[level].rc += 1
#             X[level].fwd = true
#         end
#     end

#     return num_mult
# end

## End Combinatorial Multigrid Derivative Code


function CombinatorialMG(mat::problem_interface)
    mat.type == :symmetric_graph && return CombinatorialMG(mat.Original) # run on the Laplacian if undirected graph
    mat.Scaled.diagonally_dominant && return CombinatorialMG(mat.Scaled, mat.Scaled) # run on the scaled if SDD
    mat.Original.diagonally_dominant && return CombinatorialMG(mat.Original, mat.Original) # run on the original if SDD
    return CombinatorialMG(mat.Special, mat.Scaled) # run on the padded, apply to scaled
end

function CombinatorialMG(mat::package)
    try
        work_lookup = CSV.read(joinpath(DEP_DIR, "CombinatorialWorkCost.csv"), DataFrame)

        idx = findfirst(x -> x == input.name, work_lookup[!, "Matrix"])

        if idx === nothing
            throw(ArgumentError("Matrix not found in the lookup table."))
        end

        num_multiplications = work_lookup[idx, "Work"]

        pfunc, _ = CombinatorialMultigrid.cmg_preconditioner_lap(mat.Laplacian)

        # pfunc, H, M, W, X = CombinatorialMultigrid.cmg_preconditioner_lap(mat.Laplacian)
        # num_multiplications = preconditioner_i(H, W, X, mat.b)

        # open(cmg_path, "a") do io
        #     write(io, "$(mat.name),$num_multiplications\n")
        # end


        function LinearOperator(y, r)
            y .= pfunc(r)
        end

        return Preconditioner(LinearOperator, num_multiplications, mat)
        
    catch y
        println("Error creating Combinatorial Multigrid Preconditioner for $(mat.name).")
        rethrow(y)
    end
end


function CombinatorialMG(input::package, run_on::package)
    try 
        work_lookup = CSV.read(joinpath(DEP_DIR, "CombinatorialWorkCost.csv"), DataFrame)

        idx = findfirst(x -> x == input.name, work_lookup[!, "Matrix"])

        if idx === nothing
            throw(ArgumentError("Matrix not found in the lookup table."))
        end

        num_multiplications = work_lookup[idx, "Work"]

        A_, b = make_augmented(input.A, input.b)
        pfunc, _ = CombinatorialMultigrid.cmg_preconditioner_lap(A_)

        sz = Integer(size(input.A, 1))
        
        sol_ = zeros(size(A_, 1))
        rhs = zeros(size(A_, 1))

        function LinearOperator(sol, r)

            sol .= r

            rhs[1:sz] .= sol
            rhs[sz+1:end] .= .-sol

            sol_ .= pfunc(rhs)

            sol .= view(sol_, 1:sz) .- view(sol_, sz+1:length(sol_))

        end

        return Preconditioner(LinearOperator, num_multiplications, run_on)


    catch y
        println("Error creating Combinatorial Multigrid Preconditioner for $(input.name).")
        rethrow(y)
    end
end


function Control(mat::problem_interface)
    return _Control(mat.Scaled)
end


function _Control(input::package)
    function LinearOperator(y, r)
        y .= r
    end
    num_multiplications = 0
    println("Created Control Preconditioner for $(input.name).")
    return Preconditioner(LinearOperator, num_multiplications, input)
end

end