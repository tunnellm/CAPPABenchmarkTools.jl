module Solvers

using SparseArrays
using LinearAlgebra

export preconditioned_conjugate_gradient, preconditioned_conjugate_residual, preconditioned_minres

using ..Utilities: norm_information, compute_tolerances
using ..Preconditioners: Preconditioner

# File I/O
using CodecZstd, TranscodingStreams, BufferedStreams

include("Tolerances.jl")

# Constants for the raw data output
const comma = b", "
const newline = b"\n"

"""
    pcg_print(
        stream,
        i,
        relative_residual,
        relative_normwise_2,
        relative_normwise_i,
        relative_normwise_f,
        error,
        base_cost,
        prec_cost,
        μ,
        r1,
        r1z1_new,
        pprod,
        np,
        Ax,
        x,
        b
    )

Writes iteration metrics for the Preconditioned Conjugate Gradient (PCG) algorithm to the specified stream.

# Arguments
- `stream`: Output stream where the iteration data is written.
- `i::Int`: Current iteration number.
- `relative_residual::Float64`: Relative residual of the solution.
- `relative_normwise_2::Float64`: Relative 2-norm of the solution.
- `relative_normwise_i::Float64`: Relative norm-wise error at initialization.
- `relative_normwise_f::Float64`: Relative norm-wise error at final iteration.
- `error::Float64`: Current computed error.
- `base_cost::Float64`: Accumulated computational cost of matrix-vector products.
- `prec_cost::Float64`: Accumulated computational cost of preconditioning operations.
- `μ::Float64`: Step size parameter for the PCG iteration.
- `r1::Vector`: Residual vector at the current iteration.
- `r1z1_new::Float64`: Inner product of the residual and preconditioned residual.
- `pprod::Float64`: Inner product of the conjugate direction vector.
- `np::Float64`: Norm of the conjugate direction vector.
- `Ax::Vector`: Result of matrix-vector multiplication `A * x`.
- `x::Vector`: Current solution estimate.
- `b::Vector`: Right-hand side vector of the linear system.

"""
function pcg_print(
    stream,
    i,
    relative_residual,
    relative_normwise_2,
    relative_normwise_i,
    relative_normwise_f,
    error,
    base_cost,
    prec_cost,
    μ,
    r1,
    r1z1_new,
    pprod,
    np,
    Ax,
    x,
    b,
)

    write(stream, string(i))
    write(stream, comma)
    write(stream, string(relative_residual))
    write(stream, comma)
    write(stream, string(error))
    write(stream, comma)
    write(stream, string(relative_normwise_2))
    write(stream, comma)
    write(stream, string(relative_normwise_i))
    write(stream, comma)
    write(stream, string(relative_normwise_f))
    write(stream, comma)
    write(stream, string(base_cost * i))
    write(stream, comma)
    write(stream, string(prec_cost * (i + 1)))
    write(stream, comma)
    write(stream, string(abs(μ)))
    write(stream, comma)
    write(stream, string(norm(r1)))
    write(stream, comma)
    write(stream, string(r1z1_new))
    write(stream, comma)
    write(stream, string(pprod))
    write(stream, comma)
    write(stream, string(np))
    write(stream, comma)
    write(stream, string(0.5 * (x' * Ax) - x' * b))
    write(stream, newline)

end

"""
    preconditioned_conjugate_gradient(
        preconditioner::Preconditioner,
        maxiters::Int64,
        norms::norm_information,
        raw_data_out::String,
        convergence_data_out::String;
        work_limit::Int64 = -1
    )

Solves the linear system `Ax = b` using the Preconditioned Conjugate Gradient (PCG) method with a given preconditioner.

# Arguments
- `preconditioner::Preconditioner`: The preconditioner, which includes a `package` struct containing the system to be solved.
- `maxiters::Int64`: The maximum number of iterations.
- `raw_data_out::String`: Path to the file where raw iteration data will be stored. This file will be compressed using Zstd.
- `convergence_data_out::String`: Path prefix for convergence data output files.
- `work_limit::Int64` (optional, default `-1`): Maximum allowed computational work before termination. If `-1`, no limit is set.

# Returns
- `iter::Int`: The number of iterations performed.

# Notes
- The function extracts the system matrix `A` and right-hand side `b` from `preconditioner.package.system`.
- Uses the preconditioner to accelerate convergence.
- Iteration data, including relative residuals and norm-wise relative backward errors, is logged to the specified output files.
- The algorithm terminates when convergence is reached, the iteration limit is hit, or computational work exceeds `work_limit`.
"""
function preconditioned_conjugate_gradient(
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
)

    base_cost = nnz(preconditioner.system.A)
    prec_cost = preconditioner.num_multiplications

    raw_data = open(raw_data_out, "w")
    raw_data_stream = BufferedOutputStream(raw_data, buffer_size)
    compressed_data_stream = ZstdCompressorStream(raw_data_stream, level = 1)

    write(compressed_data_stream, raw_header)

    relative_residual_out = open(convergence_data_out * "_relative_residual.csv", "w")
    write(relative_residual_out, convergence_header)
    relative_normwise_2_out = open(convergence_data_out * "_relative_normwise_2.csv", "w")
    write(relative_normwise_2_out, convergence_header)
    relative_normwise_i_out = open(convergence_data_out * "_relative_normwise_i.csv", "w")
    write(relative_normwise_i_out, convergence_header)
    relative_normwise_f_out = open(convergence_data_out * "_relative_normwise_f.csv", "w")
    write(relative_normwise_f_out, convergence_header)

    relative_residual_tolerances = copy(tolerances)
    relative_normwise_2_tolerances = copy(tolerances)
    relative_normwise_i_tolerances = copy(tolerances)
    relative_normwise_f_tolerances = copy(tolerances)

    norm_b = norm(preconditioner.system.b)

    x = zeros(size(preconditioner.system.A, 1))
    x_true = copy(x)
    r = copy(preconditioner.system.b)

    z = zeros(size(preconditioner.system.A, 1))

    preconditioner.LinearOperator(z, r)

    p = copy(z)

    iter = 0
    Ax = zeros(size(preconditioner.system.b))

    prod = preconditioner.system.A' * p

    pprod = p' * prod

    rz_old = r' * z

    μ = rz_old / pprod

    r_true = copy(preconditioner.system.b)

    np = norm(p)

    pcg_print(
        compressed_data_stream,
        0,
        compute_tolerances(norm(r), norm_b, norm(x), preconditioner.system.norms)...,
        norm(preconditioner.system.seed),
        base_cost,
        prec_cost,
        μ,
        r,
        rz_old,
        pprod,
        np,
        Ax,
        x,
        preconditioner.system.b,
    )

    for i = 1:maxiters

        iter = i

        rz_new = r' * z

        if i > 1

            τ = rz_new / rz_old
            p .= p .* τ .+ z

        end
        mul!(prod, preconditioner.system.A', p)



        pprod = p' * prod

        μ = rz_new / pprod

        x .+= μ .* p

        r .-= μ .* prod

        preconditioner.LinearOperator(z, r)



        rz_old = rz_new

        if preconditioner.system.nullspace > 0
            x_true .= x .- x[preconditioner.system.nullspace]
        else
            x_true .= x
        end



        mul!(Ax, preconditioner.system.A', x_true)

        r_true .= preconditioner.system.b .- Ax

        relative_residual, relative_normwise_2, relative_normwise_i, relative_normwise_f =
            compute_tolerances(
                norm(r_true),
                norm_b,
                norm(x_true),
                preconditioner.system.norms,
            )

        np = norm(p)

        pcg_print(
            compressed_data_stream,
            i,
            relative_residual,
            relative_normwise_2,
            relative_normwise_i,
            relative_normwise_f,
            norm(preconditioner.system.seed .- x_true),
            base_cost,
            prec_cost,
            μ,
            r,
            rz_new,
            pprod,
            np,
            Ax,
            x_true,
            preconditioner.system.b,
        )


        if isnan(μ)
            close(compressed_data_stream)
            close(raw_data_stream)
            close(raw_data)
            close(relative_residual_out)
            close(relative_normwise_2_out)
            close(relative_normwise_i_out)
            close(relative_normwise_f_out)
            throw(ArgumentError("NaN Detected in Conjugate Gradient"))
        end


        @label relres
        for tol in relative_residual_tolerances
            if relative_residual <= tol
                deleteat!(
                    relative_residual_tolerances,
                    findfirst(x -> x == tol, relative_residual_tolerances),
                )
                write(relative_residual_out, string(i))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(tol))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(relative_residual))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(base_cost * i))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(prec_cost * (i + 1)))
                write(relative_residual_out, newline)
                @goto relres
            end
        end
        @label relnormwise2
        for tol in relative_normwise_2_tolerances
            if relative_normwise_2 <= tol
                deleteat!(
                    relative_normwise_2_tolerances,
                    findfirst(x -> x == tol, relative_normwise_2_tolerances),
                )
                write(relative_normwise_2_out, string(i))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(tol))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(relative_normwise_2))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(base_cost * i))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(prec_cost * (i + 1)))
                write(relative_normwise_2_out, newline)
                @goto relnormwise2
            end
        end
        @label relnormwisei
        for tol in relative_normwise_i_tolerances
            if relative_normwise_i <= tol
                deleteat!(
                    relative_normwise_i_tolerances,
                    findfirst(x -> x == tol, relative_normwise_i_tolerances),
                )
                write(relative_normwise_i_out, string(i))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(tol))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(relative_normwise_i))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(base_cost * i))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(prec_cost * (i + 1)))
                write(relative_normwise_i_out, newline)
                @goto relnormwisei
            end
        end
        @label relnormwisef
        for tol in relative_normwise_f_tolerances
            if relative_normwise_f <= tol
                deleteat!(
                    relative_normwise_f_tolerances,
                    findfirst(x -> x == tol, relative_normwise_f_tolerances),
                )
                write(relative_normwise_f_out, string(i))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(tol))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(relative_normwise_f))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(base_cost * i))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(prec_cost * (i + 1)))
                write(relative_normwise_f_out, newline)
                @goto relnormwisef
            end
        end

        if (
            isempty(relative_residual_tolerances) &&
            isempty(relative_normwise_2_tolerances) &&
            isempty(relative_normwise_i_tolerances) &&
            isempty(relative_normwise_f_tolerances)
        )
            break
        end


        if work_limit > 0 && (base_cost * i + prec_cost * (i + 1)) > work_limit
            iter = -2
            break
        end

    end
    close(compressed_data_stream)
    close(raw_data_stream)
    close(raw_data)
    close(relative_residual_out)
    close(relative_normwise_2_out)
    close(relative_normwise_i_out)
    close(relative_normwise_f_out)
    return iter
end


"""
    pcr_print(
        stream,
        i,
        relative_residual,
        relative_normwise_2,
        relative_normwise_i,
        relative_normwise_f,
        error,
        base_cost,
        prec_cost,
        α,
        r1,
        zAz,
        ApAp,
        np,
        Ax,
        x,
        b
    )

Writes iteration metrics for the Preconditioned Conjugate Residual (PCR) algorithm to the specified stream.

# Arguments
- `stream`: Output stream where the iteration data is written.
- `i::Int`: Current iteration number.
- `relative_residual::Float64`: Relative residual of the solution.
- `relative_normwise_2::Float64`: Relative 2-norm of the solution.
- `relative_normwise_i::Float64`: Relative norm-wise error at initialization.
- `relative_normwise_f::Float64`: Relative norm-wise error at final iteration.
- `error::Float64`: Current computed error.
- `base_cost::Float64`: Accumulated computational cost of matrix-vector products.
- `prec_cost::Float64`: Accumulated computational cost of preconditioning operations.
- `α::Float64`: Step size parameter for the PCR iteration.
- `r1::Vector`: Residual vector at the current iteration.
- `zAz::Float64`: Inner product of the preconditioned residual and A times preconditioned residual.
- `ApAp::Float64`: Inner product of A times the conjugate direction vector.
- `np::Float64`: Norm of the conjugate direction vector.
- `Ax::Vector`: Result of matrix-vector multiplication `A * x`.
- `x::Vector`: Current solution estimate.
- `b::Vector`: Right-hand side vector of the linear system.

"""
function pcr_print(
    stream,
    i,
    relative_residual,
    relative_normwise_2,
    relative_normwise_i,
    relative_normwise_f,
    error,
    base_cost,
    prec_cost,
    α,
    r1,
    zAz,
    ApAp,
    np,
    Ax,
    x,
    b,
)

    write(stream, string(i))
    write(stream, comma)
    write(stream, string(relative_residual))
    write(stream, comma)
    write(stream, string(error))
    write(stream, comma)
    write(stream, string(relative_normwise_2))
    write(stream, comma)
    write(stream, string(relative_normwise_i))
    write(stream, comma)
    write(stream, string(relative_normwise_f))
    write(stream, comma)
    write(stream, string(base_cost * (i + 1)))
    write(stream, comma)
    write(stream, string(prec_cost * (i + 1)))
    write(stream, comma)
    write(stream, string(abs(α)))
    write(stream, comma)
    write(stream, string(norm(r1)))
    write(stream, comma)
    write(stream, string(zAz))
    write(stream, comma)
    write(stream, string(ApAp))
    write(stream, comma)
    write(stream, string(np))
    write(stream, comma)
    write(stream, string(0.5 * (x' * Ax) - x' * b))
    write(stream, newline)

end

"""
    preconditioned_conjugate_residual(
        preconditioner::Preconditioner,
        maxiters::Int64,
        raw_data_out::String,
        convergence_data_out::String;
        work_limit::Int64 = -1
    )

Solves the linear system `Ax = b` using the Preconditioned Conjugate Residual (PCR) method with a given preconditioner.

# Arguments
- `preconditioner::Preconditioner`: The preconditioner, which includes a `package` struct containing the system to be solved.
- `maxiters::Int64`: The maximum number of iterations.
- `raw_data_out::String`: Path to the file where raw iteration data will be stored. This file will be compressed using Zstd.
- `convergence_data_out::String`: Path prefix for convergence data output files.
- `work_limit::Int64` (optional, default `-1`): Maximum allowed computational work before termination. If `-1`, no limit is set.

# Returns
- `iter::Int`: The number of iterations performed.

# Notes
- The function extracts the system matrix `A` and right-hand side `b` from `preconditioner.package.system`.
- Uses the preconditioner to accelerate convergence.
- Unlike PCG which minimizes the A-norm of the error, PCR minimizes the 2-norm of the residual.
- Iteration data, including relative residuals and norm-wise relative backward errors, is logged to the specified output files.
- The algorithm terminates when convergence is reached, the iteration limit is hit, or computational work exceeds `work_limit`.
"""
function preconditioned_conjugate_residual(
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
)

    base_cost = nnz(preconditioner.system.A)
    prec_cost = preconditioner.num_multiplications

    raw_data = open(raw_data_out, "w")
    raw_data_stream = BufferedOutputStream(raw_data, buffer_size)
    compressed_data_stream = ZstdCompressorStream(raw_data_stream, level = 1)

    write(compressed_data_stream, raw_header)

    relative_residual_out = open(convergence_data_out * "_relative_residual.csv", "w")
    write(relative_residual_out, convergence_header)
    relative_normwise_2_out = open(convergence_data_out * "_relative_normwise_2.csv", "w")
    write(relative_normwise_2_out, convergence_header)
    relative_normwise_i_out = open(convergence_data_out * "_relative_normwise_i.csv", "w")
    write(relative_normwise_i_out, convergence_header)
    relative_normwise_f_out = open(convergence_data_out * "_relative_normwise_f.csv", "w")
    write(relative_normwise_f_out, convergence_header)

    relative_residual_tolerances = copy(tolerances)
    relative_normwise_2_tolerances = copy(tolerances)
    relative_normwise_i_tolerances = copy(tolerances)
    relative_normwise_f_tolerances = copy(tolerances)

    norm_b = norm(preconditioner.system.b)

    x = zeros(size(preconditioner.system.A, 1))
    x_true = copy(x)
    r = copy(preconditioner.system.b)

    z = zeros(size(preconditioner.system.A, 1))

    preconditioner.LinearOperator(z, r)

    # For CR, we need Az (A times preconditioned residual)
    Az = zeros(size(preconditioner.system.A, 1))
    mul!(Az, preconditioner.system.A', z)

    p = copy(z)
    Ap = copy(Az)

    iter = 0
    Ax = zeros(size(preconditioner.system.b))

    # CR uses z'*Az for numerator instead of r'*z
    zAz_old = z' * Az

    # CR uses (Ap)'*(Ap) for denominator instead of p'*Ap
    ApAp = Ap' * Ap

    α = zAz_old / ApAp

    r_true = copy(preconditioner.system.b)

    np = norm(p)

    pcr_print(
        compressed_data_stream,
        0,
        compute_tolerances(norm(r), norm_b, norm(x), preconditioner.system.norms)...,
        norm(preconditioner.system.seed),
        base_cost,
        prec_cost,
        α,
        r,
        zAz_old,
        ApAp,
        np,
        Ax,
        x,
        preconditioner.system.b,
    )

    for i = 1:maxiters

        iter = i

        zAz_new = z' * Az

        if i > 1

            β = zAz_new / zAz_old
            p .= z .+ β .* p
            Ap .= Az .+ β .* Ap

        end

        ApAp = Ap' * Ap

        α = zAz_new / ApAp

        x .+= α .* p

        r .-= α .* Ap

        preconditioner.LinearOperator(z, r)

        mul!(Az, preconditioner.system.A', z)

        zAz_old = zAz_new

        if preconditioner.system.nullspace > 0
            x_true .= x .- x[preconditioner.system.nullspace]
        else
            x_true .= x
        end

        mul!(Ax, preconditioner.system.A', x_true)

        r_true .= preconditioner.system.b .- Ax

        relative_residual, relative_normwise_2, relative_normwise_i, relative_normwise_f =
            compute_tolerances(
                norm(r_true),
                norm_b,
                norm(x_true),
                preconditioner.system.norms,
            )

        np = norm(p)

        pcr_print(
            compressed_data_stream,
            i,
            relative_residual,
            relative_normwise_2,
            relative_normwise_i,
            relative_normwise_f,
            norm(preconditioner.system.seed .- x_true),
            base_cost,
            prec_cost,
            α,
            r,
            zAz_new,
            ApAp,
            np,
            Ax,
            x_true,
            preconditioner.system.b,
        )


        if isnan(α)
            close(compressed_data_stream)
            close(raw_data_stream)
            close(raw_data)
            close(relative_residual_out)
            close(relative_normwise_2_out)
            close(relative_normwise_i_out)
            close(relative_normwise_f_out)
            throw(ArgumentError("NaN Detected in Conjugate Residual"))
        end


        @label relres
        for tol in relative_residual_tolerances
            if relative_residual <= tol
                deleteat!(
                    relative_residual_tolerances,
                    findfirst(x -> x == tol, relative_residual_tolerances),
                )
                write(relative_residual_out, string(i))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(tol))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(relative_residual))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(base_cost * (i + 1)))
                write(relative_residual_out, comma)
                write(relative_residual_out, string(prec_cost * (i + 1)))
                write(relative_residual_out, newline)
                @goto relres
            end
        end
        @label relnormwise2
        for tol in relative_normwise_2_tolerances
            if relative_normwise_2 <= tol
                deleteat!(
                    relative_normwise_2_tolerances,
                    findfirst(x -> x == tol, relative_normwise_2_tolerances),
                )
                write(relative_normwise_2_out, string(i))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(tol))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(relative_normwise_2))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(base_cost * (i + 1)))
                write(relative_normwise_2_out, comma)
                write(relative_normwise_2_out, string(prec_cost * (i + 1)))
                write(relative_normwise_2_out, newline)
                @goto relnormwise2
            end
        end
        @label relnormwisei
        for tol in relative_normwise_i_tolerances
            if relative_normwise_i <= tol
                deleteat!(
                    relative_normwise_i_tolerances,
                    findfirst(x -> x == tol, relative_normwise_i_tolerances),
                )
                write(relative_normwise_i_out, string(i))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(tol))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(relative_normwise_i))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(base_cost * (i + 1)))
                write(relative_normwise_i_out, comma)
                write(relative_normwise_i_out, string(prec_cost * (i + 1)))
                write(relative_normwise_i_out, newline)
                @goto relnormwisei
            end
        end
        @label relnormwisef
        for tol in relative_normwise_f_tolerances
            if relative_normwise_f <= tol
                deleteat!(
                    relative_normwise_f_tolerances,
                    findfirst(x -> x == tol, relative_normwise_f_tolerances),
                )
                write(relative_normwise_f_out, string(i))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(tol))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(relative_normwise_f))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(base_cost * (i + 1)))
                write(relative_normwise_f_out, comma)
                write(relative_normwise_f_out, string(prec_cost * (i + 1)))
                write(relative_normwise_f_out, newline)
                @goto relnormwisef
            end
        end

        if (
            isempty(relative_residual_tolerances) &&
            isempty(relative_normwise_2_tolerances) &&
            isempty(relative_normwise_i_tolerances) &&
            isempty(relative_normwise_f_tolerances)
        )
            break
        end


        if work_limit > 0 && (base_cost * (i + 1) + prec_cost * (i + 1)) > work_limit
            iter = -2
            break
        end

    end
    close(compressed_data_stream)
    close(raw_data_stream)
    close(raw_data)
    close(relative_residual_out)
    close(relative_normwise_2_out)
    close(relative_normwise_i_out)
    close(relative_normwise_f_out)
    return iter
end


"""
    minres_print(
        stream,
        i,
        relative_residual,
        relative_normwise_2,
        relative_normwise_i,
        relative_normwise_f,
        error,
        base_cost,
        prec_cost,
        phi,
        r1,
        r2y,
        beta,
        alfa,
        anorm,
        acond,
        dxnorm,
        rnorm,
        epsx,
        epsr,
        norm_w,
        Ax,
        x,
        b
    )

Writes iteration metrics for the Preconditioned Minimal Residual (MINRES) algorithm to the specified stream.

# Arguments
- `stream`: Output stream where the iteration data is written.
- `i::Int`: Current iteration number.
- `relative_residual::Float64`: Relative residual of the solution.
- `relative_normwise_2::Float64`: Relative 2-norm of the solution.
- `relative_normwise_i::Float64`: Relative norm-wise error at initialization.
- `relative_normwise_f::Float64`: Relative norm-wise error at final iteration.
- `error::Float64`: Current computed error.
- `base_cost::Float64`: Accumulated computational cost of matrix-vector products.
- `prec_cost::Float64`: Accumulated computational cost of preconditioning operations.
- `phi::Float64`: Lanczos tridiagonalization parameter used in MINRES.
- `r1::Vector`: Residual vector at the current iteration.
- `r2y::Float64`: Inner product of the second residual vector and the preconditioned residual.
- `beta::Float64`: Lanczos recurrence coefficient.
- `alfa::Float64`: Lanczos diagonal term.
- `anorm::Float64`: Estimated norm of the system matrix.
- `acond::Float64`: Condition number estimate of the system matrix.
- `dxnorm::Float64`: Norm of the update step in the solution.
- `rnorm::Float64`: Norm of the residual.
- `epsx::Float64`: Computed stopping tolerance based on solution norm.
- `epsr::Float64`: Computed stopping tolerance based on residual norm.
- `norm_w::Float64`: Norm of the conjugate search direction.
- `Ax::Vector`: Result of matrix-vector multiplication `A * x`.
- `x::Vector`: Current solution estimate.
- `b::Vector`: Right-hand side vector of the linear system.

# Notes
- The function writes each metric as a comma-separated value (CSV) entry.
- `base_cost` and `prec_cost` track computational costs per iteration.
- The last value written corresponds to the objective function value `(1/2) x'Ax - x'b`.
"""
function minres_print(
    stream,
    i,
    relative_residual,
    relative_normwise_2,
    relative_normwise_i,
    relative_normwise_f,
    error,
    base_cost,
    prec_cost,
    phi,
    r1,
    r2y,
    beta,
    alfa,
    anorm,
    acond,
    dxnorm,
    rnorm,
    epsx,
    epsr,
    norm_w,
    Ax,
    x,
    b,
)

    write(stream, string(i))
    write(stream, comma)
    write(stream, string(relative_residual))
    write(stream, comma)
    write(stream, string(error))
    write(stream, comma)
    write(stream, string(relative_normwise_2))
    write(stream, comma)
    write(stream, string(relative_normwise_i))
    write(stream, comma)
    write(stream, string(relative_normwise_f))
    write(stream, comma)
    write(stream, string(base_cost * i))
    write(stream, comma)
    write(stream, string(prec_cost * (i + 1)))
    write(stream, comma)
    write(stream, string(phi))
    write(stream, comma)
    write(stream, string(norm(r1)))
    write(stream, comma)
    write(stream, string(r2y))
    write(stream, comma)
    write(stream, string(beta))
    write(stream, comma)
    write(stream, string(alfa))
    write(stream, comma)
    write(stream, string(anorm))
    write(stream, comma)
    write(stream, string(acond))
    write(stream, comma)
    write(stream, string(dxnorm))
    write(stream, comma)
    write(stream, string(rnorm))
    write(stream, comma)
    write(stream, string(epsx))
    write(stream, comma)
    write(stream, string(epsr))
    write(stream, comma)
    write(stream, string(norm_w))
    write(stream, comma)
    write(stream, string(0.5 * (x' * Ax) - x' * b))
    write(stream, newline)

end

"""
    preconditioned_minres(
        preconditioner::Preconditioner,
        maxiters::Int64,
        norms::norm_information,
        raw_data_out::String,
        convergence_data_out::String;
        work_limit::Int64 = -1,
        nullspace::Int64 = 0
    )

Solves the linear system `Ax = b` using the Preconditioned Minimal Residual (MINRES) method with a given preconditioner.

# Arguments
- `preconditioner::Preconditioner`: The preconditioner, which includes a `package` struct containing the system to be solved.
- `maxiters::Int64`: The maximum number of iterations.
- `raw_data_out::String`: Path to the file where raw iteration data will be stored. This file will be compressed using Zstd.
- `convergence_data_out::String`: Path prefix for convergence data output files.
- `work_limit::Int64` (optional, default `-1`): Maximum allowed computational work before termination. If `-1`, no limit is set.

# Returns
- `iter::Int`: The number of iterations performed, or `-1` if termination criteria were met before convergence.

# Notes
- The function extracts the system matrix `A` and right-hand side `b` from `preconditioner.package.system`.
- Uses the preconditioner to accelerate convergence.
- Applies the Lanczos tridiagonalization process to approximate the solution while minimizing the residual.
- Iteration data, including relative residuals and norm-wise errors, is logged to the specified output files.
- The algorithm terminates when convergence is reached, the iteration limit is hit, computational work exceeds `work_limit`, or the preconditioner fails to maintain positive definiteness.
"""
function preconditioned_minres(
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
)

    rtol = 1e-16

    raw_data = open(raw_data_out, "w")
    raw_data_stream = BufferedOutputStream(raw_data, buffer_size)
    compressed_data_stream = ZstdCompressorStream(raw_data_stream, level = 1)

    base_cost = nnz(preconditioner.system.A)
    prec_cost = preconditioner.num_multiplications

    write(compressed_data_stream, minres_raw_header)

    relative_residual_out = open(convergence_data_out * "_relative_residual.csv", "w")
    write(relative_residual_out, convergence_header)
    relative_normwise_2_out = open(convergence_data_out * "_relative_normwise_2.csv", "w")
    write(relative_normwise_2_out, convergence_header)
    relative_normwise_i_out = open(convergence_data_out * "_relative_normwise_i.csv", "w")
    write(relative_normwise_i_out, convergence_header)
    relative_normwise_f_out = open(convergence_data_out * "_relative_normwise_f.csv", "w")
    write(relative_normwise_f_out, convergence_header)

    relative_residual_tolerances = copy(tolerances)
    relative_normwise_2_tolerances = copy(tolerances)
    relative_normwise_i_tolerances = copy(tolerances)
    relative_normwise_f_tolerances = copy(tolerances)

    n = length(preconditioner.system.b)
    r0 = copy(preconditioner.system.b)
    bnorm = norm(preconditioner.system.b)
    rtol0 = copy(rtol)
    rnorm = 0.0
    arnorm = 0.0
    anorm = 0.0
    acond = 0.0
    dxnorm = 0.0
    x = zeros(n)
    x_true = copy(x)
    r1 = copy(r0)
    y = zeros(n)
    preconditioner.LinearOperator(y, r0)
    beta1 = r0' * y
    if beta1 < 0.0
        close(compressed_data_stream)
        close(raw_data_stream)
        close(raw_data)
        close(relative_residual_out)
        close(relative_normwise_2_out)
        close(relative_normwise_i_out)
        close(relative_normwise_f_out)
        throw(ArgumentError("Preconditioner must be positive definite"))
    end

    numrtol = 1
    beta1 = sqrt(beta1)
    istop = 0
    oldb = 0.0
    beta = copy(beta1)
    dbar = 0.0
    epsln = 0.0
    phibar = copy(beta1)
    rhs1 = copy(beta1)
    rhs2 = 0.0
    tnorm2 = 0.0
    cs = -1.0
    sn = 0.0
    gmax = 0.0
    gmin = typemax(Float64)
    r2 = copy(r0)
    w = zeros(n)
    w1 = zeros(n)
    w2 = zeros(n)
    x0norm = 0.0

    iter = 0

    v = zeros(n)

    r_true = zeros(n)
    Ax = zeros(n)

    minres_print(
        compressed_data_stream,
        0,
        compute_tolerances(norm(r0), bnorm, norm(x), preconditioner.system.norms)...,
        norm(preconditioner.system.seed),
        base_cost,
        prec_cost,
        0.0,
        r0,
        r2' * y,
        beta,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        Ax,
        x,
        preconditioner.system.b,
    )

    for i = 1:maxiters
        s = 1 / beta
        v .= s .* y
        mul!(y, preconditioner.system.A', v)
        # y = A' * v
        if i > 1
            y .-= (beta / oldb) .* r1
        end

        alfa = v' * y
        y .-= (alfa / beta) .* r2
        r1 .= r2
        r2 .= y
        preconditioner.LinearOperator(y, r2)
        oldb = copy(beta)
        beta = r2' * y

        # Check if any of these things are NaN
        if isnan(alfa) ||
           isnan(beta) ||
           isnan(epsln) ||
           isnan(dbar) ||
           isnan(phibar) ||
           isnan(rhs1) ||
           isnan(rhs2) ||
           isnan(tnorm2) ||
           isnan(arnorm) ||
           isnan(anorm) ||
           isnan(acond) ||
           isnan(dxnorm) ||
           isnan(rnorm) ||
           isnan(gmax) ||
           isnan(gmin)
            close(compressed_data_stream)
            close(raw_data_stream)
            close(raw_data)
            close(relative_residual_out)
            close(relative_normwise_2_out)
            close(relative_normwise_i_out)
            close(relative_normwise_f_out)
            throw(ArgumentError("NaN Detected in MINRES"))
        end

        if beta < 0
            close(compressed_data_stream)
            close(raw_data_stream)
            close(raw_data)
            close(relative_residual_out)
            close(relative_normwise_2_out)
            close(relative_normwise_i_out)
            close(relative_normwise_f_out)
            throw(ArgumentError("Preconditioner must be positive definite"))
        end

        beta = sqrt(beta)
        tnorm2 = tnorm2 + alfa^2 + oldb^2 + beta^2
        oldeps = copy(epsln)
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        # root = norm([gbar; dbar])
        root = sqrt(gbar^2 + dbar^2)
        arnorm = phibar * root

        # gamma = norm([gbar; beta])
        gamma = sqrt(gbar^2 + beta^2)
        gamma = max(gamma, eps(1.0))
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        denom = 1 / gamma

        w1 .= w2

        w2 .= w
        w .= (v .- (oldeps .* w1) .- (delta .* w2)) .* denom

        x .+= phi .* w

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z


        anorm = sqrt(tnorm2)
        dxnorm = norm(x)
        rnorm = copy(phibar)

        acond = gmax / gmin

        epsx = (anorm * dxnorm + beta1) * eps(1.0)
        epsr = (anorm * dxnorm + beta1) * rtol

        # test1 = rnorm / (anorm * dxnorm + bnorm)
        # test2 = arnorm / (anorm * (rnorm + eps(1.0)))
        # t1 = 1 + test1
        # t2 = 1 + test2

        # if t2 <= 1
        #     println("t2")
        #     istop = 2
        # end

        # if t1 <= 1
        #     println("t1")
        #     istop = 1
        # end

        if i >= maxiters
            # println("maxiters")
            istop = 6
        end

        if acond >= 0.1 / eps(1.0)
            # println("acond")
            istop = 5
        end

        if epsx >= beta1
            # println("epsx")
            istop = 3
        end

        # if test2 <= rtol
        #     println("test2")
        #     istop = 2
        # end

        # if test1 <= rtol
        #     println("test1")
        #     istop = 1
        # end

        # println("istop = $istop")

        if preconditioner.num_multiplications != 0 && istop > 0 && istop <= 5
            if x0norm == 0
                xnorm = dxnorm
                rnormk = norm(preconditioner.system.b - preconditioner.system.A' * x)
            else
                xt = x0 + x
                xnorm = norm(xt)
                rnormk = norm(preconditioner.system.b - preconditioner.system.A' * xt)
            end
            epsr = (anorm * xnorm + bnorm) * rtol0
            if rnormk <= epsr
                istop = 1
            elseif numrtol < 5 && rtol > eps(1.0)
                numrtol = numrtol + 1
                rtol = rtol / 10
                istop = 0
            else
                istop = 10
            end
        end

        if preconditioner.system.nullspace > 0
            x_true .= x .- x[preconditioner.system.nullspace]
        else
            x_true .= x
        end

        mul!(Ax, preconditioner.system.A', x_true)
        r_true .= preconditioner.system.b .- Ax


        relative_residual, relative_normwise_2, relative_normwise_i, relative_normwise_f =
            compute_tolerances(
                norm(r_true),
                bnorm,
                norm(x_true),
                preconditioner.system.norms,
            )

        minres_print(
            compressed_data_stream,
            i,
            relative_residual,
            relative_normwise_2,
            relative_normwise_i,
            relative_normwise_f,
            norm(preconditioner.system.seed .- x_true),
            base_cost,
            prec_cost,
            phi,
            r1,
            r2' * y,
            beta,
            alfa,
            anorm,
            acond,
            dxnorm,
            rnorm,
            epsx,
            epsr,
            norm(w),
            Ax,
            x_true,
            preconditioner.system.b,
        )

        # write(raw_data, "$i, $relative_residual, $relative_normwise_2, $relative_normwise_i, $relative_normwise_f, $(base_cost * i), $(prec_cost * (i + 1))\n")
        @label relres
        for tol in relative_residual_tolerances
            if relative_residual <= tol
                deleteat!(
                    relative_residual_tolerances,
                    findfirst(x -> x == tol, relative_residual_tolerances),
                )
                write(
                    relative_residual_out,
                    "$i, $tol, $relative_residual, $(base_cost * i), $(prec_cost * (i + 1))\n",
                )
                @goto relres
            end
        end
        @label relnormwise2
        for tol in relative_normwise_2_tolerances
            if relative_normwise_2 <= tol
                deleteat!(
                    relative_normwise_2_tolerances,
                    findfirst(x -> x == tol, relative_normwise_2_tolerances),
                )
                write(
                    relative_normwise_2_out,
                    "$i, $tol, $relative_normwise_2, $(base_cost * i), $(prec_cost * (i + 1))\n",
                )
                @goto relnormwise2
            end
        end
        @label relnormwisei
        for tol in relative_normwise_i_tolerances
            if relative_normwise_i <= tol
                deleteat!(
                    relative_normwise_i_tolerances,
                    findfirst(x -> x == tol, relative_normwise_i_tolerances),
                )
                write(
                    relative_normwise_i_out,
                    "$i, $tol, $relative_normwise_i, $(base_cost * i), $(prec_cost * (i + 1))\n",
                )
                @goto relnormwisei
            end
        end
        @label relnormwisef
        for tol in relative_normwise_f_tolerances
            if relative_normwise_f <= tol
                deleteat!(
                    relative_normwise_f_tolerances,
                    findfirst(x -> x == tol, relative_normwise_f_tolerances),
                )
                write(
                    relative_normwise_f_out,
                    "$i, $tol, $relative_normwise_f, $(base_cost * i), $(prec_cost * (i + 1))\n",
                )
                @goto relnormwisef
            end
        end
        iter = i
        if isempty(relative_residual_tolerances) &&
           isempty(relative_normwise_2_tolerances) &&
           isempty(relative_normwise_i_tolerances) &&
           isempty(relative_normwise_f_tolerances)
            break
        end

        if work_limit > 0 && (base_cost * i + prec_cost * (i + 1)) > work_limit
            iter = -2
            break
        end

        if istop == 1
            break
        end

        if istop > 0
            iter = -1
            break
        end
    end
    close(compressed_data_stream)
    close(raw_data_stream)
    close(raw_data)
    close(relative_residual_out)
    close(relative_normwise_2_out)
    close(relative_normwise_i_out)
    close(relative_normwise_f_out)
    return iter#, x
end

end
