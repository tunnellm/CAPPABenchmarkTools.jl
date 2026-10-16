module Preconditioners

using ..Utilities: reduce_to_laplacian, laplace_to_adj, estimate_eigrange, tridiag_eigvals!
using ..PreconditionerTypes: Preconditioner, OptionedPreconditioner
using ..Hypre: HypreBoomerAMG, HypreFSAI, HypreCheby, HypreParaSails, HypreSchwarz
using ..ThreadedUtilities: threaded_spmv!, _sleep_julia_workers!, _sleep_julia_workers_for_blas!

using SparseArrays
using LinearAlgebra
using Statistics
using Random
using StableRNGs

import PythonCall

import Laplacians

const EXTERNAL_DIR = joinpath(@__DIR__, "..", "External")
include(joinpath(EXTERNAL_DIR, "FastSSAI3", "src", "FastSSAI3.jl"))
using .FastSSAI3: ssai3
include(joinpath(EXTERNAL_DIR, "LimitedLDLFactorizations", "src", "LimitedLDLFactorizations.jl"))
import .LimitedLDLFactorizations as LLDL
include(joinpath(EXTERNAL_DIR, "ILUK", "src", "ILUK.jl"))
import .ILUK

export Preconditioner, OptionedPreconditioner
export LaplaciansDeg, LaplaciansGiven, LaplaciansWdeg, LaplaciansRandom
export PyAMGRuStu, PyAMGSmAgg, PyAMGRootnode, PyAMGPairwise
export Control, TruncatedNeumann, TruncatedNeumannAdaptive
export Chebyshev, ChebyshevAdaptive, SSOR, SGS, SSAI
export LimitedLDLShiftPD, LimitedLDLAbsPD
export ILUKShiftPD, ILUKAbsPD
export HypreBoomerAMG, HypreFSAI, HypreCheby, HypreParaSails, HypreSchwarz


"""
    LaplaciansDeg(A; split=0, merge=0) -> Preconditioner

Approximate Cholesky preconditioner using greedy minimum-degree ordering.

The matrix `A` is automatically reduced to a graph Laplacian via
`reduce_to_laplacian` (SDDM ground node or SDD double cover), then
factored using `Laplacians.approxchol_prec_deg`.

`split=0` (default) uses the basic algorithm. `split>=2` enables edge
splitting for robustness (merge is set equal to split internally).
"""
function LaplaciansDeg(A::SparseMatrixCSC; split::Integer=0)
    split == 1 && error("split=1 is not recommended: uses split data structures with no actual splitting. Use split=0 or split>=2.")
    components, wrap_solver = reduce_to_laplacian(A)
    prec, meta = _build_preconditioner(components, wrap_solver) do adj
        Laplacians.approxchol_prec_deg(adj; split=split, merge=split)
    end
    meta["config"] = Dict{String,Any}("split" => split)
    return prec, meta
end

"""
    LaplaciansGiven(A) -> (Preconditioner, Dict)

Approximate Cholesky preconditioner using fixed natural ordering.
"""
function LaplaciansGiven(A::SparseMatrixCSC)
    components, wrap_solver = reduce_to_laplacian(A)
    prec, meta = _build_preconditioner(components, wrap_solver) do adj
        Laplacians.approxchol_prec_given(adj)
    end
    meta["config"] = Dict{String,Any}()
    return prec, meta
end

"""
    LaplaciansWdeg(A; seed::Integer) -> (Preconditioner, Dict)

Approximate Cholesky preconditioner using weighted-degree ordering with
random perturbation. `seed` is required for reproducibility.
"""
function LaplaciansWdeg(A::SparseMatrixCSC; seed::Integer)
    Random.seed!(seed)
    components, wrap_solver = reduce_to_laplacian(A)
    prec, meta = _build_preconditioner(components, wrap_solver) do adj
        Laplacians.approxchol_prec_wdeg(adj)
    end
    meta["config"] = Dict{String,Any}("seed" => seed)
    return prec, meta
end

"""
    LaplaciansRandom(A; seed::Integer) -> (Preconditioner, Dict)

Approximate Cholesky preconditioner using random permutation ordering.
`seed` is required for reproducibility.
"""
function LaplaciansRandom(A::SparseMatrixCSC; seed::Integer)
    Random.seed!(seed)
    components, wrap_solver = reduce_to_laplacian(A)
    prec, meta = _build_preconditioner(components, wrap_solver) do adj
        Laplacians.approxchol_prec_random(adj)
    end
    meta["config"] = Dict{String,Any}("seed" => seed)
    return prec, meta
end

"""
    _build_preconditioner(factor_fn, components, wrap_solver) -> Preconditioner

Internal helper. Factors each component's adjacency matrix using `factor_fn`,
accumulates work counters, and wraps everything into a `Preconditioner`.

`factor_fn(adj)` must return `(F!, wc, num_mults)`.
"""
function _build_preconditioner(factor_fn, components, wrap_solver)
    solvers = Vector{Any}(undef, length(components))
    total_apply_fma = UInt64(0)
    total_gen_fma = UInt64(0)
    total_gen_graph = UInt64(0)

    for (k, (adj, _)) in enumerate(components)
        F!, wc, num_mults = factor_fn(adj)
        solvers[k] = F!
        total_apply_fma += UInt64(num_mults)
        total_gen_fma += UInt64(wc.flops)
        total_gen_graph += UInt64(wc.graph_ops)
    end

    LinearOperator = wrap_solver(solvers, total_apply_fma)
    prec = Preconditioner(LinearOperator, total_gen_fma, total_gen_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => total_apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => total_gen_fma,
        "generation_graph" => total_gen_graph,
        "num_components" => length(components),
    )
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# Identity (Control) Preconditioner
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Control(A) -> (Preconditioner, Dict)

Identity preconditioner: applies `z = r` with zero work.
Useful as a baseline for benchmarking.
"""
function Control(A::SparseMatrixCSC)
    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        z .= r
        return 0.0
    end
    prec = Preconditioner(apply!, UInt64(0), UInt64(0))
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => UInt64(0),
        "apply_cost_kind" => "fixed",
        "generation_fma" => UInt64(0),
        "generation_graph" => UInt64(0),
        "config" => Dict{String,Any}(),
    )
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# Truncated Neumann Series Preconditioner
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TruncatedNeumann(A::SparseMatrixCSC; num_iters::Int,
                     lanczos_k::Int, method::Symbol) -> (Preconditioner, Dict)

Truncated Neumann series preconditioner using Chebyshev-optimal scaling:

    M⁻¹ ≈ α · Σᵢ₌₀ᵏ (I - αA)ⁱ

where α = 2/(λ_min + λ_max) is the Chebyshev center of the spectrum.
The Chebyshev center minimizes the spectral radius ρ(I - αA), giving
ρ = (κ-1)/(κ+1) vs ρ = 1 - 1/κ for the naive choice α = 1/λ_max.

The series converges when ρ(I - αA) < 1, i.e., when all eigenvalues of A are
positive and the spectral bounds are reasonable.

# Arguments
- `num_iters::Int`: number of Neumann terms (higher = better approximation, more work)
- `lanczos_k::Int`: Lanczos iterations for eigenvalue estimation. Must be > 0 for
  Lanczos-based methods, must be 0 for norm-based methods.
- `method::Symbol`: spectral bound estimation method:
  - `:ritz` — use extreme Ritz values directly (optimistic, tight when converged)
  - `:conservative` — widen Ritz values by residual bounds (Bauer-Fike)
  - `:heuristic` — λ_min = θ_min / 30 (simple safety margin)
  - `:extrapolation` — Aitken Δ² acceleration of Ritz value sequences
  - `:one_norm` — α = 2/‖A‖₁ (max absolute column sum, O(nnz), no Lanczos)
  - `:frobenius` — α = 2/‖A‖_F (Frobenius norm, O(nnz), no Lanczos)
  - `:norm_min` — α = 2/min(‖A‖₁, ‖A‖_F), both norms in one pass
  - `:norm_safe` — α = 2/(1.1 min(‖A‖₁, ‖A‖_F))
  - `:lanczos_ritz` — unguarded Ritz scale with the timing binary's start
  - `:lanczos_guarded` — α = 2/(θ_min + 1.1(θ_max + ρ_max))
"""
function TruncatedNeumann(A::SparseMatrixCSC;
                          num_iters::Int,
                          lanczos_k::Int, method::Symbol)
    n = size(A, 1)
    nz = nnz(A)

    norm_upper = NaN
    if method == :norm_min || method == :norm_safe
        @assert lanczos_k == 0 "lanczos_k must be 0 for method=:$method"
        one_norm = 0.0
        frobenius_sq = 0.0
        @inbounds for col in 1:n
            column_sum = 0.0
            for p in A.colptr[col]:(A.colptr[col + 1] - 1)
                value = A.nzval[p]
                column_sum += abs(value)
                frobenius_sq += value * value
            end
            one_norm = max(one_norm, column_sum)
        end
        norm_upper = min(one_norm, sqrt(frobenius_sq))
        @assert isfinite(norm_upper) && norm_upper > 0
        α = 2.0 / ((method == :norm_safe ? 1.1 : 1.0) * norm_upper)
        generation_fma = UInt64(2) * UInt64(nz)
        λ_min = NaN; λ_max = norm_upper
        ritz_min_res = NaN; ritz_max_res = NaN
        ritz_min_gap = NaN; ritz_max_gap = NaN
    elseif method == :one_norm
        @assert lanczos_k == 0 "lanczos_k must be 0 for method=:one_norm"
        norm_val = opnorm(A, 1)
        α = 2.0 / norm_val
        generation_fma = UInt64(nz)
        λ_min = NaN; λ_max = norm_val
        ritz_min_res = NaN; ritz_max_res = NaN
        ritz_min_gap = NaN; ritz_max_gap = NaN
    elseif method == :frobenius
        @assert lanczos_k == 0 "lanczos_k must be 0 for method=:frobenius"
        _sleep_julia_workers_for_blas!()
        norm_val = norm(A)
        α = 2.0 / norm_val
        generation_fma = UInt64(nz)
        λ_min = NaN; λ_max = norm_val
        ritz_min_res = NaN; ritz_max_res = NaN
        ritz_min_gap = NaN; ritz_max_gap = NaN
    else
        @assert lanczos_k > 0 "lanczos_k must be > 0 for Lanczos-based method=:$method"
        estimator_method = method == :lanczos_guarded ? :guarded :
                           method == :lanczos_ritz ? :portable_ritz : method
        λ_min, λ_max, eigrange_fma, ritz_min_res, ritz_max_res,
            ritz_min_gap, ritz_max_gap = estimate_eigrange(A, lanczos_k;
                                                           method = estimator_method)
        α = method == :lanczos_guarded ?
            2.0 / (λ_min + 1.1 * (λ_max + ritz_max_res)) :
            2.0 / (λ_min + λ_max)
        generation_fma = eigrange_fma
    end

    # Apply cost per call:
    #   num_iters SpMVs: num_iters * nnz(A) FMA
    #   num_iters vector updates (w - α*temp + z): 2n FMA each (fused multiply-sub + add)
    #   final scaling (z *= α): n FMA
    apply_fma = UInt64(num_iters) * UInt64(nz) + UInt64(2 * num_iters + 1) * UInt64(n)

    # Workspace (closed over by apply!)
    w    = zeros(n)
    temp = zeros(n)

    apply! = let A = A, α = α, apply_fma = apply_fma, n = n, num_iters = num_iters,
                 temp = temp, w = w
        function apply_inner(z::Vector{Float64}, r::Vector{Float64})::UInt64
            @inbounds for i in 1:n
                w[i] = r[i]
                z[i] = r[i]
            end
            for _ in 1:num_iters
                threaded_spmv!(temp, A, z)
                @inbounds for i in 1:n
                    z[i] = w[i] + z[i] - α * temp[i]
                end
            end
            @inbounds for i in 1:n
                z[i] *= α
            end
            return apply_fma
        end
        apply_inner
    end

    prec = Preconditioner(apply!, UInt64(generation_fma), UInt64(0))
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => UInt64(generation_fma),
        "generation_graph" => UInt64(0),
        "lambda_min" => λ_min,
        "lambda_max" => λ_max,
        "alpha" => α,
        "norm_upper" => norm_upper,
        "guard_factor" => method in (:norm_safe, :lanczos_guarded) ? 1.1 : 1.0,
        "ritz_min_residual" => ritz_min_res,
        "ritz_max_residual" => ritz_max_res,
        "ritz_min_half_gap" => ritz_min_gap,
        "ritz_max_half_gap" => ritz_max_gap,
        "config" => Dict{String,Any}(
            "num_iters" => num_iters,
            "lanczos_k" => lanczos_k,
            "method" => String(method),
        ),
    )
    return prec, meta
end

"""
    TruncatedNeumannAdaptive(A::SparseMatrixCSC; num_iters::Int,
                              lanczos_k::Symbol, method::Symbol) -> (Preconditioner, Dict)

Adaptive wrapper around `TruncatedNeumann` that computes `lanczos_k` from the
matrix dimension.

# Arguments
- `num_iters::Int`: number of Neumann terms
- `lanczos_k::Symbol`: adaptive scheme for Lanczos iteration count:
  - `:log2_scaled` — `ceil(30 * log2(n))`, grows slowly with dimension
- `method::Symbol`: spectral bound estimation method (see `TruncatedNeumann`)
"""
function TruncatedNeumannAdaptive(A::SparseMatrixCSC;
                                   num_iters::Int, lanczos_k::Symbol,
                                   method::Symbol)
    n = size(A, 1)
    if method in (:one_norm, :frobenius, :norm_min, :norm_safe)
        k = 0
    elseif lanczos_k == :log2_scaled
        k = ceil(Int, 30 * log2(n))
    else
        error("Unknown adaptive lanczos_k: $lanczos_k")
    end
    prec, meta = TruncatedNeumann(A; num_iters=num_iters, lanczos_k=k, method=method)
    meta["config"]["lanczos_k_scheme"] = String(lanczos_k)
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chebyshev Polynomial Preconditioner (via hypre)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Chebyshev(A::SparseMatrixCSC; order::Int, lanczos_k::Int,
              method::Symbol, variant::Symbol) -> (Preconditioner, Dict)

Chebyshev polynomial preconditioner using hypre's implementation with spectral
bounds estimated via Lanczos + DSTERF.

Upstream hypre clamps the requested degree to 4. Our hypre fork preserves its
degree-1 through degree-4 implementation and extends the CPU implementation to
higher degrees (including 8 and 16) using Chebyshev semi-iteration, with matching
work accounting. Higher degrees are not silently replaced by degree 4.

The spectral bounds [λ_min, λ_max] are computed by `estimate_eigrange` using
the specified `method` and `lanczos_k` Lanczos iterations, then passed to
hypre's Chebyshev solver. `eig_ratio = 0` covers the full estimated interval
[λ_min, 1.1 × λ_max]. A positive ratio raises its lower endpoint by that
fraction of the interval width. Older wrappers silently retained hypre's 0.3
default; corrected runs have an explicit ratio in their configuration identity.

Generation cost includes both the Lanczos eigenvalue estimation and hypre's
polynomial coefficient setup. Apply cost is reported by hypre.

# Arguments
- `order::Int`: Chebyshev polynomial degree
- `eig_ratio::Real = 0.0`: lower-endpoint interpolation fraction in [0, 1]
- `lanczos_k::Int`: Lanczos iterations for eigenvalue estimation (must be > 0)
- `method::Symbol`: eigenvalue estimation method — `:ritz`, `:conservative`,
  `:heuristic`, or `:extrapolation`
- `variant::Symbol`: Chebyshev polynomial variant:
  - `:standard_residual` — standard Chebyshev residual polynomial r(t) = 1 - t·s(t)
  - `:alternative` — for degree m >= 2, the degree-(m-1) normalized Chebyshev
    residual polynomial times `(1 - t/upper_bound)`; degree 1 uses `1 - t/theta`
"""
function Chebyshev(A::SparseMatrixCSC;
                   order::Int,
                   lanczos_k::Int, method::Symbol,
                   variant::Symbol, eig_ratio::Real = 0.0)
    @assert lanczos_k > 0 "lanczos_k must be > 0"
    isfinite(eig_ratio) && 0 <= eig_ratio <= 1 ||
        throw(ArgumentError("eig_ratio must be finite and in [0, 1]"))

    # Estimate spectral bounds
    λ_min, λ_max, eigrange_fma, ritz_min_res, ritz_max_res,
        ritz_min_gap, ritz_max_gap = estimate_eigrange(A, lanczos_k; method = method)

    P, cheby_apply_fma = HypreCheby(A; order = order, variant = variant, scale = false,
                                     eig_ratio = eig_ratio, min_eig_est = λ_min, max_eig_est = λ_max)

    # Sum our Lanczos cost with hypre's residual setup cost
    total_gen_fma = eigrange_fma + P.generation_fma

    prec = Preconditioner(P.LinearOperator, UInt64(total_gen_fma), P.generation_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => cheby_apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => UInt64(total_gen_fma),
        "generation_graph" => P.generation_graph,
        "lambda_min" => λ_min,
        "lambda_max" => λ_max,
        "ritz_min_residual" => ritz_min_res,
        "ritz_max_residual" => ritz_max_res,
        "ritz_min_half_gap" => ritz_min_gap,
        "ritz_max_half_gap" => ritz_max_gap,
        "chebyshev_implementation" => order > 4 ? "hypre_host_semi_iteration_v1" : "hypre_legacy_horner",
        "effective_order" => order,
        "config" => Dict{String,Any}(
            "order" => order,
            "eig_ratio" => Float64(eig_ratio),
            "lanczos_k" => lanczos_k,
            "method" => String(method),
            "variant" => String(variant),
        ),
    )
    return prec, meta
end

"""
    ChebyshevAdaptive(A::SparseMatrixCSC; order::Int, lanczos_k::Symbol,
                       method::Symbol, variant::Symbol) -> (Preconditioner, Dict)

Adaptive wrapper around `Chebyshev` that computes `lanczos_k` from the
matrix dimension.

# Arguments
- `order::Int`: Chebyshev polynomial degree
- `lanczos_k::Symbol`: adaptive scheme for Lanczos iteration count:
  - `:log2_scaled` — `ceil(30 * log2(n))`, grows slowly with dimension
- `method::Symbol`: eigenvalue estimation method (see `Chebyshev`)
- `variant::Symbol`: Chebyshev variant (see `Chebyshev`)
- `eig_ratio::Real = 0.0`: lower-endpoint interpolation (see `Chebyshev`)
"""
function ChebyshevAdaptive(A::SparseMatrixCSC;
                            order::Int, lanczos_k::Symbol,
                            method::Symbol, variant::Symbol, eig_ratio::Real = 0.0)
    n = size(A, 1)
    if lanczos_k == :log2_scaled
        k = ceil(Int, 30 * log2(n))
    else
        error("Unknown adaptive lanczos_k: $lanczos_k")
    end
    prec, meta = Chebyshev(A; order=order, lanczos_k=k, method=method, variant=variant, eig_ratio=eig_ratio)
    meta["config"]["lanczos_k_scheme"] = String(lanczos_k)
    return prec, meta
end

# The corrected default must not reuse a pre-fix result directory when callers
# omit the keyword. Other preconditioners retain their existing identities.
_preconditioner_identity_kwargs(preconditioner, kwargs) = kwargs
function _preconditioner_identity_kwargs(
        ::Union{typeof(Chebyshev),typeof(ChebyshevAdaptive)}, kwargs)
    ratio = Float64(get(kwargs, "eig_ratio", 0.0))
    isfinite(ratio) && 0 <= ratio <= 1 ||
        throw(ArgumentError("eig_ratio must be finite and in [0, 1]"))
    return merge(kwargs, Dict{String,Any}("eig_ratio" => ratio))
end

# ═══════════════════════════════════════════════════════════════════════════════
# SSOR Preconditioner
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SSOR(A::SparseMatrixCSC; num_iters::Int, ω::Float64) -> Preconditioner

Symmetric Successive Over-Relaxation preconditioner on a scaled (unit diagonal)
SPD matrix. Each iteration performs a forward then backward Gauss-Seidel sweep
with relaxation parameter ω:

    x[i] = (1-ω)·x[i] + ω·(b[i] - Σ_{j≠i} a[i,j]·x[j])

(no diagonal divide since the input is assumed scaled to unit diagonal).

# FMA accounting

Per iteration (forward + backward sweep):
- Off-diagonal dot products: 2·(nnz - n) FMA
- b[i] - row_sum: 2n adds (counted as FMA)
- ω·(...): 2n multiplies
- (1-ω)·x[i] + ...: 2n FMA
→ Total per iteration: 2·(nnz + n) FMA

# Arguments
- `num_iters::Int`: number of SSOR iterations
- `ω::Float64`: relaxation parameter
"""
function SSOR(A::SparseMatrixCSC;
              num_iters::Int,
              ω::Real)
    ω = Float64(ω)
    n = size(A, 1)
    nz = nnz(A)

    colptr = A.colptr
    rowval = A.rowval
    nzval  = A.nzval

    ω_val = ω

    # Apply cost: 2·(nnz + n) per iteration
    apply_fma = UInt64(2) * UInt64(nz + n) * UInt64(num_iters)

    # Branchless inner loop: include the unit diagonal in the @simd sum,
    # then absorb the extra ω·z[i] diagonal contribution into ψ:
    #   z[i] = (1-ω)·z[i] + ω·(r[i] - Σ_{j≠i} a[i,j]·z[j])       (standard)
    #        = (1-ω)·z[i] + ω·(r[i] - full_sum + z[i])             (a[i,i]=1)
    #        = z[i] + ω·(r[i] - full_sum)                           ((1-ω)+ω=1)
    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        z .= 0.0
        @inbounds for _ = 1:num_iters
            # Forward sweep
            for i = 1:n
                full_sum = 0.0
                @simd for j = colptr[i]:(colptr[i+1] - 1)
                    full_sum = muladd(nzval[j], z[rowval[j]], full_sum)
                end
                z[i] = z[i] + ω_val * (r[i] - full_sum)
            end
            # Backward sweep
            for i = n:-1:1
                full_sum = 0.0
                @simd for j = colptr[i]:(colptr[i+1] - 1)
                    full_sum = muladd(nzval[j], z[rowval[j]], full_sum)
                end
                z[i] = z[i] + ω_val * (r[i] - full_sum)
            end
        end
        return apply_fma
    end

    prec = Preconditioner(apply!, UInt64(0), UInt64(0))
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => UInt64(0),
        "generation_graph" => UInt64(0),
        "config" => Dict{String,Any}(
            "num_iters" => num_iters,
            "omega" => ω_val,
        ),
    )
    return prec, meta
end


# ═══════════════════════════════════════════════════════════════════════════════
# Symmetric Gauss-Seidel Preconditioner
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SGS(A::SparseMatrixCSC; num_iters::Int) -> Preconditioner

Symmetric Gauss-Seidel preconditioner on a scaled (unit diagonal) SPD matrix.
This is the specialized ω=1 case of SSOR with no relaxation weighting and no
diagonal divide, giving the minimal-cost symmetric sweep:

    x[i] = b[i] - Σ_{j≠i} a[i,j]·x[j]

# FMA accounting

Per iteration (forward + backward sweep):
- Off-diagonal dot products: 2·(nnz - n) FMA
- b[i] - row_sum: 2n adds (counted as FMA)
→ Total per iteration: 2·nnz FMA

No generation cost — no Lanczos, no parameter computation.
"""
function SGS(A::SparseMatrixCSC; num_iters::Int)
    n = size(A, 1)
    nz = nnz(A)

    colptr = A.colptr
    rowval = A.rowval
    nzval  = A.nzval

    # Apply cost: 2·nnz per iteration
    apply_fma = UInt64(2) * UInt64(nz) * UInt64(num_iters)

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        z .= 0.0
        @inbounds for _ = 1:num_iters
            # Forward sweep (branchless: include diagonal=1 in sum, add z[i] back)
            for i = 1:n
                full_sum = 0.0
                @simd for j = colptr[i]:(colptr[i+1] - 1)
                    full_sum = muladd(nzval[j], z[rowval[j]], full_sum)
                end
                z[i] = r[i] - full_sum + z[i]
            end
            # Backward sweep
            for i = n:-1:1
                full_sum = 0.0
                @simd for j = colptr[i]:(colptr[i+1] - 1)
                    full_sum = muladd(nzval[j], z[rowval[j]], full_sum)
                end
                z[i] = r[i] - full_sum + z[i]
            end
        end
        return apply_fma
    end

    prec = Preconditioner(apply!, UInt64(0), UInt64(0))
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => UInt64(0),
        "generation_graph" => UInt64(0),
        "config" => Dict{String,Any}(
            "num_iters" => num_iters,
        ),
    )
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# SSAI — Symmetric Sparse Approximate Inverse
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SSAI(A::SparseMatrixCSC; fill_factor::Real) -> Preconditioner

Symmetric Sparse Approximate Inverse preconditioner (SSAI3 algorithm).

Computes a sparse symmetric matrix M ≈ A⁻¹ such that applying the preconditioner
is a single SpMV: z = M · r.

Algorithm by Shaked Regev and Michael Saunders, ICME, Stanford University.
Reference: https://stanford.edu/group/SOL/reports/20SSAI.pdf

# Arguments
- `fill_factor::Real`: multiplier for average column degree to determine target
  fill level in M. Higher = denser approximation, more setup work, better quality.

# FMA accounting
- Generation: reported by FastSSAI3 (includes all Gram-Schmidt, triangular solves,
  and column-norm precomputation)
- Generation graph: sorting and sparsity pattern operations (reported by FastSSAI3)
- Apply: nnz(M) FMA (single SpMV)
"""
function SSAI(A::SparseMatrixCSC; fill_factor::Real)
    result = ssai3(A; fill_factor = fill_factor)
    M = result.M

    apply_fma = UInt64(result.solve_flops)
    generation_fma = UInt64(result.flops)
    generation_graph = UInt64(result.graph_ops)

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        threaded_spmv!(z, M, r)
        return apply_fma
    end

    prec = Preconditioner(apply!, generation_fma, generation_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => generation_fma,
        "generation_graph" => generation_graph,
        "config" => Dict{String,Any}(
            "fill_factor" => fill_factor,
        ),
    )
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# Limited-Memory LDLᵀ Preconditioners
# ═══════════════════════════════════════════════════════════════════════════════

"""
    LimitedLDLShiftPD(A::SparseMatrixCSC; memory_factor::Integer, droptol::Real,
                      max_retries::Int=20) -> (Preconditioner, Dict)

Limited-memory LDLᵀ preconditioner with positive definiteness enforced via
diagonal shifting. Uses the vendored LimitedLDLFactorizations fork with
flop and graph-op tracking.

When a pivot becomes non-positive during factorization, the library adds a
shift α to the diagonal and retries (α *= 10 each time, starting from
√ε ≈ 1.5e-8). Up to `max_retries` attempts are made internally.

# Arguments
- `memory_factor::Integer`: fill factor multiplier (target nnz per column = memory_factor × avg degree)
- `droptol::Real`: drop tolerance for incomplete factorization
- `max_retries::Int = 20`: maximum shift-increase attempts before giving up

# Returns
- `Preconditioner`: the preconditioner
- `Dict`: diagnostics with keys `"num_retries"`, `"α_final"`, `"factorized"`, `"method"`
"""
function LimitedLDLShiftPD(A::SparseMatrixCSC;
                           memory_factor::Integer, droptol::Real,
                           max_retries::Int = 20)
    n = size(A, 1)
    nzpercol = ((nnz(A) - n) / 2) / n
    memory_val = ceil(Int, nzpercol * memory_factor)
    P = collect(1:n)

    ldl = LLDL.lldl_spd(A; P = P, memory = memory_val, droptol = droptol,
                         α = 0.0, max_increase_α = max_retries)

    LLDL.factorized(ldl) || error("LimitedLDLShiftPD failed after $(ldl.num_retries) retries, α=$(ldl.α_out)")

    apply_fma = UInt64(ldl.solve_flops)
    generation_fma = UInt64(ldl.flops)
    generation_graph = UInt64(ldl.graph_ops)

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        ldiv!(z, ldl, r)
        return apply_fma
    end

    nnz_L = length(ldl.Lnzvals)
    nnz_lower_A = (nnz(A) - n) ÷ 2

    prec = Preconditioner(apply!, generation_fma, generation_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => generation_fma,
        "generation_graph" => generation_graph,
        "num_retries" => ldl.num_retries,
        "alpha_final" => ldl.α_out,
        "fill_ratio" => nnz_L / nnz_lower_A,
        "config" => Dict{String,Any}(
            "memory_factor" => memory_factor,
            "droptol" => droptol,
            "max_retries" => max_retries,
        ),
    )
    return prec, meta
end

"""
    LimitedLDLAbsPD(A::SparseMatrixCSC; memory_factor::Integer, droptol::Real,
                    max_retries::Int=20) -> (Preconditioner, Dict)

Limited-memory LDLᵀ preconditioner with positive definiteness enforced by
taking the absolute value of diagonal entries D after factorization.

Uses standard (potentially indefinite) LDLᵀ factorization. If factorization
fails due to a zero pivot, retries with diagonal shift escalation (α *= 10
each time). After successful factorization, applies |D| to guarantee positive
definiteness.

# Arguments
- `memory_factor::Integer`: fill factor multiplier (target nnz per column = memory_factor × avg degree)
- `droptol::Real`: drop tolerance for incomplete factorization
- `max_retries::Int = 20`: maximum shift-increase attempts before giving up

# Returns
- `Preconditioner`: the preconditioner
- `Dict`: diagnostics with keys `"num_retries"`, `"α_final"`, `"factorized"`, `"method"`, `"neg_diag_count"`
"""
function LimitedLDLAbsPD(A::SparseMatrixCSC;
                         memory_factor::Integer, droptol::Real,
                         max_retries::Int = 20)
    n = size(A, 1)
    nzpercol = ((nnz(A) - n) / 2) / n
    memory_val = ceil(Int, nzpercol * memory_factor)
    P = collect(1:n)

    ldl = LLDL.lldl(A; P = P, memory = memory_val, droptol = droptol,
                     α = 0.0, max_increase_α = max_retries)

    LLDL.factorized(ldl) || error("LimitedLDLAbsPD failed after $(ldl.num_retries) retries, α=$(ldl.α_out)")

    # Count negative diagonals before taking absolute value
    neg_diag_count = count(x -> x < 0, ldl.D)
    ldl.D .= abs.(ldl.D)

    apply_fma = UInt64(ldl.solve_flops)
    generation_fma = UInt64(ldl.flops)
    generation_graph = UInt64(ldl.graph_ops)

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        ldiv!(z, ldl, r)
        return apply_fma
    end

    nnz_L = length(ldl.Lnzvals)
    nnz_lower_A = (nnz(A) - n) ÷ 2

    prec = Preconditioner(apply!, generation_fma, generation_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => generation_fma,
        "generation_graph" => generation_graph,
        "num_retries" => ldl.num_retries,
        "alpha_final" => ldl.α_out,
        "fill_ratio" => nnz_L / nnz_lower_A,
        "neg_diag_count" => neg_diag_count,
        "config" => Dict{String,Any}(
            "memory_factor" => memory_factor,
            "droptol" => droptol,
            "max_retries" => max_retries,
        ),
    )
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# ILU(k) LDLᵀ Preconditioners
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ILUKShiftPD(A::SparseMatrixCSC; ilu_level::Integer,
                max_retries::Int=20) -> (Preconditioner, Dict)

Incomplete LDLᵀ preconditioner with level-of-fill `k` and positive definiteness
enforced via diagonal shifting. Uses the vendored ILUK fork with flop and
graph-op tracking.

When a pivot becomes non-positive during factorization, the library adds a
shift α to the diagonal and retries (α *= 10 each time, starting from
√ε ≈ 1.5e-8). Up to `max_retries` attempts are made internally.

# Arguments
- `ilu_level::Integer`: level of fill (ilu_level ≥ 0)
- `max_retries::Int = 20`: maximum shift-increase attempts before giving up

# Returns
- `Preconditioner`: the preconditioner
- `Dict`: diagnostics with keys `"num_retries"`, `"α_final"`, `"fill_ratio"`
"""
function ILUKShiftPD(A::SparseMatrixCSC;
                     ilu_level::Integer,
                     max_retries::Int = 20)
    n = size(A, 1)

    ldl = ILUK.ldlt_k(A, ilu_level;
                       α = 0.0,
                       ensure_positive = true,
                       max_attempts = max_retries)

    ldl.success || error("ILUKShiftPD failed after $(ldl.num_retries) retries, α=$(ldl.shift)")

    apply_fma = UInt64(ldl.solve_flops)
    generation_fma = UInt64(ldl.flops)
    generation_graph = UInt64(ldl.graph_ops)

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        ILUK.ldl_solve!(z, ldl.L, ldl.D, r)
        return apply_fma
    end

    nnz_lower_A = (nnz(A) - n) ÷ 2

    prec = Preconditioner(apply!, generation_fma, generation_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => generation_fma,
        "generation_graph" => generation_graph,
        "num_retries" => ldl.num_retries,
        "alpha_final" => ldl.shift,
        "fill_ratio" => nnz(ldl.L) / nnz_lower_A,
        "config" => Dict{String,Any}(
            "ilu_level" => ilu_level,
            "max_retries" => max_retries,
        ),
    )
    return prec, meta
end

"""
    ILUKAbsPD(A::SparseMatrixCSC; ilu_level::Integer,
              max_retries::Int=20) -> (Preconditioner, Dict)

Incomplete LDLᵀ preconditioner with level-of-fill `k` and positive definiteness
enforced by taking the absolute value of diagonal entries D after factorization.

Uses standard (potentially indefinite) LDLᵀ factorization. If factorization
fails due to a zero pivot, retries with diagonal shift escalation (α *= 10
each time). After successful factorization, applies |D| to guarantee positive
definiteness.

# Arguments
- `ilu_level::Integer`: level of fill (ilu_level ≥ 0)
- `max_retries::Int = 20`: maximum shift-increase attempts before giving up

# Returns
- `Preconditioner`: the preconditioner
- `Dict`: diagnostics with keys `"num_retries"`, `"α_final"`, `"fill_ratio"`, `"neg_diag_count"`
"""
function ILUKAbsPD(A::SparseMatrixCSC;
                   ilu_level::Integer,
                   max_retries::Int = 20)
    n = size(A, 1)

    ldl = ILUK.ldlt_k(A, ilu_level;
                       α = 0.0,
                       ensure_positive = false,
                       max_attempts = max_retries)

    ldl.success || error("ILUKAbsPD failed after $(ldl.num_retries) retries, α=$(ldl.shift)")

    neg_diag_count = count(x -> x < 0, ldl.D)
    ldl.D .= abs.(ldl.D)

    apply_fma = UInt64(ldl.solve_flops)
    generation_fma = UInt64(ldl.flops)
    generation_graph = UInt64(ldl.graph_ops)

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        ILUK.ldl_solve!(z, ldl.L, ldl.D, r)
        return apply_fma
    end

    nnz_lower_A = (nnz(A) - n) ÷ 2

    prec = Preconditioner(apply!, generation_fma, generation_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => generation_fma,
        "generation_graph" => generation_graph,
        "num_retries" => ldl.num_retries,
        "alpha_final" => ldl.shift,
        "fill_ratio" => nnz(ldl.L) / nnz_lower_A,
        "neg_diag_count" => neg_diag_count,
        "config" => Dict{String,Any}(
            "ilu_level" => ilu_level,
            "max_retries" => max_retries,
        ),
    )
    return prec, meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# PyAMG Preconditioners
#
# Four solver types, each with explicit keyword arguments and Julia-side
# validation. All use the vendored pyamg fork with setup_complexity() for
# work tracking and _MultilevelSolver__solve for zero-allocation apply.
#
# Valid configurations per solver type:
#
# ── PyAMGRuStu (Classical / Ruge-Stuben AMG) ──
#   cycle:         :V, :W, :F, :AMLI
#   num_cycles:    1, 2, ...
#   coarse_solver: :pinv, :lu, :cholesky, :splu, :cg
#   strength:      :symmetric, :classical, :evolution, :distance,
#                  :algebraic_distance, :affinity, :energy_based, nothing
#                  Sub-options: (theta=0.25,), (epsilon=4.0,), (k=2,), etc.
#   CF:            :RS, :PMIS, :PMISc, :CLJP, :CLJPc, :CR
#                  Sub-options: (second_pass=false,), etc.
#   interpolation: :direct, :classical
#   presmoother:   see PYAMG_SMOOTHER_METHODS below
#   postsmoother:  see PYAMG_SMOOTHER_METHODS below
#   max_levels:    default 30
#   max_coarse:    default 10
#
# ── PyAMGSmAgg (Smoothed Aggregation AMG) ──
#   cycle, num_cycles, strength, presmoother, postsmoother: same as RuStu
#   aggregate:     :standard, :lloyd, :balanced_lloyd, :naive, :pairwise, :metis
#   smooth:        :jacobi, :richardson, :energy, nothing
#                  Sub-options: (omega=4/3,), (filter=true,), (krylov=:gmres,)
#   improve_candidates: smoother spec or nothing
#   diagonal_dominance: true/false
#   max_levels:    default 10
#   max_coarse:    default 10
#
# ── PyAMGRootnode (Root-node SA AMG) ──
#   Same as SmAgg except smooth only supports :energy and nothing
#
# ── PyAMGPairwise (Pairwise Aggregation AMG) ──
#   cycle, num_cycles, presmoother, postsmoother: same as RuStu
#   theta:         0.0-1.0 (strength threshold)
#   norm:          :min, :abs
#   matchings:     1, 2, 3, ... (pairwise matchings per level)
#   max_levels:    default 20
#   max_coarse:    default 10
#
# ── Smoother methods (pre/post) ──
#   :gauss_seidel, :block_gauss_seidel, :jacobi, :block_jacobi,
#   :cf_jacobi, :fc_jacobi, :cf_block_jacobi, :fc_block_jacobi,
#   :richardson, :sor, :chebyshev, :gauss_seidel_nr, :gauss_seidel_ne,
#   :jacobi_ne, :cg, :gmres, :cgne, :cgnr, :schwarz,
#   :strength_based_schwarz, nothing
#   Sub-options: (sweep=:forward,), (sweep=:backward,), (sweep=:symmetric,),
#                (iterations=2,), (omega=0.8,), (withrho=true,), (maxiter=5,)
#   Per-level: pass a Vector of specs, e.g.
#     [:gauss_seidel, (:jacobi, (omega=0.8,))]
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── Valid option sets for validation ──────────────────────────────────────────

const PYAMG_CYCLES = (:V, :W, :F, :AMLI, :K)

# Coarse solvers with proper work instrumentation.
# Dense direct: pinv (SVD), lu, cholesky — setup O(n³), apply O(n²)
# Sparse direct: splu — setup O(nnz), apply O(nnz)
# Krylov: cg — no setup cost, apply O(nnz) per iteration (10 default)
#
# NOTE: Relaxation-based coarse solvers (gauss_seidel, jacobi, etc.) are NOT
# included because PyAMG rebuilds the smoother setup on every coarse solve call
# (creates a fresh Level object each time), making them both slow and poorly
# instrumented. GMRES is excluded because our problems are symmetric.
const PYAMG_COARSE_SOLVERS = (:pinv, :lu, :cholesky, :splu, :cg)

const PYAMG_STRENGTH_METHODS = (
    :symmetric, :classical, :evolution, :distance,
    :algebraic_distance, :affinity, :energy_based, nothing,
)

const PYAMG_CF_METHODS = (:RS, :PMIS, :PMISc, :CLJP, :CLJPc, :CR)

const PYAMG_INTERPOLATION_METHODS = (:direct, :classical)

# Note: :balanced_lloyd can fail with "maxsize too small" on graphs with highly
# uneven degree distribution (e.g., power grids with hub nodes). The balanced Lloyd
# clustering algorithm allocates work arrays sized at 12× the average cluster size;
# clusters that grow beyond this limit (due to uneven connectivity) trigger the error.
# This is a limitation of the algorithm, not a bug — it works on more regular graphs.
#
# Note: balanced_lloyd_aggregation (in PyAMG's aggregate.py) constructs its AggOp via
# COO→CSR with numpy int64 row/col arrays, producing int64 indices. The downstream
# amg_core.fit_candidates pybind11 binding only accepts int32 index arrays. We patch
# fit_candidates (in tentative.py) to cast indices to int32 before calling the C extension.
const PYAMG_AGGREGATE_METHODS = (:standard, :lloyd, :balanced_lloyd, :naive, :pairwise, :metis)

# PyAMG uses spaces not underscores in aggregate method names
const _AGGREGATE_PY_NAMES = Dict{Symbol,String}(
    :balanced_lloyd => "balanced lloyd",
)
_aggregate_to_py(s::Symbol) = PythonCall.pystr(get(_AGGREGATE_PY_NAMES, s, String(s)))
_aggregate_to_py(x) = _option_to_py(x)  # tuples and other forms pass through

const PYAMG_SA_SMOOTH_METHODS = (:jacobi, :richardson, :energy, nothing)

const PYAMG_ROOTNODE_SMOOTH_METHODS = (:energy, nothing)

const PYAMG_SMOOTHER_METHODS = (
    :gauss_seidel, :block_gauss_seidel, :jacobi, :block_jacobi,
    :cf_jacobi, :fc_jacobi, :cf_block_jacobi, :fc_block_jacobi,
    :richardson, :sor, :chebyshev,
    :gauss_seidel_nr, :gauss_seidel_ne, :jacobi_ne,
    :cg, :gmres, :cgne, :cgnr,
    :schwarz, :strength_based_schwarz, nothing,
)

# ── Conversion helpers: Julia symbols/named tuples → Python tuples/dicts ─────

"""Convert a Symbol to a Python string, or pass through nothing as Python None."""
function _to_py(s::Symbol)
    return PythonCall.pystr(String(s))
end
function _to_py(::Nothing)
    return PythonCall.Py(nothing)
end

"""
Convert a smoother/option specification from Julia to Python format.

Accepted forms:
- `nothing` → Python None
- `:method` → Python string `"method"`
- `(:method, (key=val, ...))` → Python tuple `("method", {"key": val, ...})`
"""
function _option_to_py(opt::Nothing)
    return PythonCall.Py(nothing)
end
function _option_to_py(opt::Symbol)
    return _to_py(opt)
end
function _option_to_py(opt::Tuple{Symbol, NamedTuple})
    method, kwargs = opt
    d = PythonCall.pydict()
    for (k, v) in pairs(kwargs)
        d[PythonCall.pystr(String(k))] = v isa Symbol ? _to_py(v) : PythonCall.Py(v)
    end
    return PythonCall.pytuple((PythonCall.pystr(String(method)), d))
end

"""
Convert a per-level list of smoother specifications to Python.
Accepts a single spec or a Vector of specs (for per-level control).
"""
function _option_to_py(opt::Vector)
    return PythonCall.pylist([_option_to_py(o) for o in opt])
end

# ── Smoother validation ──────────────────────────────────────────────────────

function _validate_smoother(name::String, spec)
    if spec === nothing
        return
    elseif spec isa Symbol
        spec in PYAMG_SMOOTHER_METHODS || error("Invalid $name method: $spec. Valid: $PYAMG_SMOOTHER_METHODS")
    elseif spec isa Tuple{Symbol, NamedTuple}
        spec[1] in PYAMG_SMOOTHER_METHODS || error("Invalid $name method: $(spec[1]). Valid: $PYAMG_SMOOTHER_METHODS")
    elseif spec isa Vector
        for s in spec
            _validate_smoother(name, s)
        end
    else
        error("Invalid $name format. Use :method, (:method, (key=val,)), or a Vector of these.")
    end
end

# ── Strength/aggregate/smooth validation helpers ─────────────────────────────

function _validate_option(name::String, spec, valid_set)
    if spec === nothing
        nothing in valid_set || error("$name does not accept nothing.")
        return
    elseif spec isa Symbol
        spec in valid_set || error("Invalid $name: $spec. Valid: $valid_set")
    elseif spec isa Tuple{Symbol, NamedTuple}
        spec[1] in valid_set || error("Invalid $name: $(spec[1]). Valid: $valid_set")
    elseif spec isa Vector
        for s in spec
            _validate_option(name, s, valid_set)
        end
    else
        error("Invalid $name format. Use :method, (:method, (key=val,)), or a Vector of these.")
    end
end

# ── Common option parser ─────────────────────────────────────────────────────

"""
    _parse_common_pyamg_opts(; cycle, num_cycles, presmoother, postsmoother,
                              max_levels, max_coarse) -> (Dict, Symbol, Int)

Validates and converts the options shared by all four PyAMG solver types.
Returns `(kwargs_dict, cycle, num_cycles)`.
"""
function _parse_setup_pyamg_opts(;
    coarse_solver::Symbol,
    presmoother,
    postsmoother,
    max_levels::Int,
    max_coarse::Int,
)
    coarse_solver in PYAMG_COARSE_SOLVERS || error("Invalid coarse_solver: $coarse_solver. Valid: $PYAMG_COARSE_SOLVERS")
    max_levels >= 1 || error("max_levels must be >= 1, got $max_levels")
    max_coarse >= 1 || error("max_coarse must be >= 1, got $max_coarse")
    _validate_smoother("presmoother", presmoother)
    _validate_smoother("postsmoother", postsmoother)

    kwargs = Dict{Symbol, Any}(
        :presmoother => _option_to_py(presmoother),
        :postsmoother => _option_to_py(postsmoother),
        :coarse_solver => _to_py(coarse_solver),
        :max_levels => max_levels,
        :max_coarse => max_coarse,
    )

    return kwargs
end

function _parse_common_pyamg_opts(;
    cycle::Symbol,
    num_cycles::Int,
    coarse_solver::Symbol,
    presmoother,
    postsmoother,
    max_levels::Int,
    max_coarse::Int,
)
    cycle in PYAMG_CYCLES || error("Invalid cycle: $cycle. Valid: $PYAMG_CYCLES")
    num_cycles >= 1 || error("num_cycles must be >= 1, got $num_cycles")
    coarse_solver in PYAMG_COARSE_SOLVERS || error("Invalid coarse_solver: $coarse_solver. Valid: $PYAMG_COARSE_SOLVERS")
    max_levels >= 1 || error("max_levels must be >= 1, got $max_levels")
    max_coarse >= 1 || error("max_coarse must be >= 1, got $max_coarse")
    _validate_smoother("presmoother", presmoother)
    _validate_smoother("postsmoother", postsmoother)

    kwargs = Dict{Symbol, Any}(
        :presmoother => _option_to_py(presmoother),
        :postsmoother => _option_to_py(postsmoother),
        :coarse_solver => _to_py(coarse_solver),
        :max_levels => max_levels,
        :max_coarse => max_coarse,
    )

    return kwargs
end

# ── Shared PyAMG builder ─────────────────────────────────────────────────────

"""
    _build_pyamg_hierarchy(A, solver_fn_name, solver_kwargs) -> OptionedPreconditioner

Builds the multigrid hierarchy once and returns an `OptionedPreconditioner` whose
callable creates a `Preconditioner` for any combination of apply-time parameters
(cycle type, num_cycles, smoothers, coarse_solver, K-cycle settings).

The hierarchy, buffers, and setup-time costs are captured in the closure.
Call `op(; kwargs...)` to get a `(Preconditioner, meta)` pair.

Note: smoother and coarse_solver swaps mutate the shared hierarchy in place.
Only one `Preconditioner` from a given hierarchy should be in use at a time.
Omitted options restore the hierarchy's original defaults. Generation costs
include shared hierarchy work plus only the active configuration's setup,
regardless of whether another configuration has already cached that work.
"""
function _build_pyamg_hierarchy(A::SparseMatrixCSC, solver_fn_name::Symbol, solver_kwargs::Dict)
    _sleep_julia_workers!()
    n = size(A, 1)
    sp = PythonCall.pyimport("scipy.sparse")
    pyamg = PythonCall.pyimport("pyamg")
    np = PythonCall.pyimport("numpy")
    smoothing = PythonCall.pyimport("pyamg.relaxation.smoothing")

    # Convert Julia CSC to SciPy CSR (0-indexed; SciPy auto-converts indices to int32)
    A_csr = sp.csr_matrix(
        PythonCall.pytuple((PythonCall.Py(A.nzval), PythonCall.Py(A.rowval .- 1), PythonCall.Py(A.colptr .- 1))),
        shape=PythonCall.pytuple((n, n)),
    )

    # Build multigrid hierarchy
    solver_fn = PythonCall.pygetattr(pyamg, String(solver_fn_name))
    ml = solver_fn(A_csr; solver_kwargs...)

    nnz_finest = PythonCall.pyconvert(Int64, ml.levels[0].A.nnz)
    default_pre = get(solver_kwargs, :presmoother, get(solver_kwargs, :prepostsmoother, PythonCall.Py(nothing)))
    default_post = get(solver_kwargs, :postsmoother, get(solver_kwargs, :prepostsmoother, PythonCall.Py(nothing)))
    default_coarse = ml.coarse_solver
    coarse_solvers = Dict{Symbol,PythonCall.Py}()

    # Level info (fixed per hierarchy)
    num_levels = PythonCall.pyconvert(Int64, PythonCall.pylen(ml.levels))
    coarse_n = PythonCall.pyconvert(Int64, ml.levels[num_levels - 1].A.shape[0])

    # Pre-allocate Julia buffers and create numpy views (zero-copy)
    z_buf = zeros(Float64, n)
    r_buf = zeros(Float64, n)
    z_np = np.asarray(z_buf)
    r_np = np.asarray(r_buf)

    function configure(;
            cycle::Symbol=:V, num_cycles::Int=1,
            k_cg_iters::Int=1, k_early_exit::Real=0.0,
            presmoother=nothing, postsmoother=nothing,
            coarse_solver=nothing)

        _sleep_julia_workers!()
        cycle in PYAMG_CYCLES || error("Invalid cycle: $cycle")
        num_cycles >= 1 || error("num_cycles must be positive")
        cycle == :K && k_cg_iters < 1 && error("k_cg_iters must be positive")
        isfinite(k_early_exit) && k_early_exit >= 0 ||
            throw(ArgumentError("k_early_exit must be finite and nonnegative"))
        _validate_smoother("presmoother", presmoother)
        _validate_smoother("postsmoother", postsmoother)
        # Resolve defaults independently of whichever configuration ran last.
        pre_py = presmoother === nothing ? default_pre : _option_to_py(presmoother)
        post_py = postsmoother === nothing ? default_post : _option_to_py(postsmoother)
        smoothing.change_smoothers(ml, presmoother=pre_py, postsmoother=post_py)

        # Swap coarse solver if requested
        if coarse_solver !== nothing
            coarse_solver in PYAMG_COARSE_SOLVERS || error("Invalid coarse_solver: $coarse_solver")
            pyamg_util = PythonCall.pyimport("pyamg.multilevel")
            ml.coarse_solver = get!(coarse_solvers, coarse_solver) do
                pyamg_util.coarse_grid_solver(PythonCall.pystr(String(coarse_solver)))
            end
        else
            ml.coarse_solver = default_coarse
        end

        parts = ml.setup_work_breakdown()
        generation_breakdown = Dict{String,Any}()
        gen_fma = UInt64(0)
        gen_graph = UInt64(0)
        for name in ("hierarchy", "smoother", "coarse")
            part = parts[name]
            fma = UInt64(round(PythonCall.pyconvert(Float64, part[0])))
            graph = UInt64(round(PythonCall.pyconvert(Float64, part[1])))
            generation_breakdown[name] = Dict("fma" => fma, "graph" => graph)
            gen_fma += fma
            gen_graph += graph
        end

        # Set K-cycle parameters on the solver object
        if cycle == :K
            ml._k_cg_iters = k_cg_iters
            ml._k_early_exit = k_early_exit
        end

        # Compute cycle complexity (reads smoothers dynamically)
        cycle_str = String(cycle)
        py_cycle = PythonCall.pystr(cycle_str)
        cycle_cc = PythonCall.pyconvert(Float64, ml.cycle_complexity(cycle=cycle_str))
        apply_fma_per_cycle = cycle_cc * nnz_finest
        apply_fma = UInt64(round(apply_fma_per_cycle * num_cycles))

        # Build config from setup kwargs + apply kwargs
        config = Dict{String,Any}(String(k) => v for (k, v) in solver_kwargs)
        config["cycle"] = String(cycle)
        config["num_cycles"] = num_cycles
        config["k_cg_iters"] = k_cg_iters
        config["k_early_exit"] = k_early_exit
        if presmoother !== nothing
            config["presmoother"] = presmoother
        end
        if postsmoother !== nothing
            config["postsmoother"] = postsmoother
        end
        if coarse_solver !== nothing
            config["coarse_solver"] = String(coarse_solver)
        end

        meta = Dict{String,Any}(
            "success" => true,
            "apply_fma" => apply_fma,
            # K's full traversal is used for screening only with ratio early
            # exit disabled. Exact-zero branches can still cost less.
            "apply_cost_kind" => cycle == :K ?
                (k_early_exit == 0 ? "fixed_inner_iterations" : "variable") : "fixed",
            "generation_fma" => gen_fma,
            "generation_graph" => gen_graph,
            "generation_cost_breakdown" => generation_breakdown,
            "generation_cost_model" => "configuration_setup_v1",
            "num_levels" => num_levels,
            "coarse_n" => coarse_n,
            "cycle_complexity" => cycle_cc,
            "config" => config,
        )

        if cycle == :K
            function VariableLinearOperator(z::Vector{Float64}, r::Vector{Float64})::Float64
                _sleep_julia_workers!()
                r_buf .= r
                z_buf .= 0.0
                cost = 0.0
                for _ in 1:num_cycles
                    cost += PythonCall.pyconvert(Float64,
                        ml._MultilevelSolver__solve(0, z_np, r_np, py_cycle))
                end
                z .= z_buf
                return cost
            end
            return Preconditioner(VariableLinearOperator, gen_fma, gen_graph), meta
        else
            function LinearOperator(z::Vector{Float64}, r::Vector{Float64})::Float64
                _sleep_julia_workers!()
                r_buf .= r
                z_buf .= 0.0
                for _ in 1:num_cycles
                    ml._MultilevelSolver__solve(0, z_np, r_np, py_cycle)
                end
                z .= z_buf
                return apply_fma
            end
            return Preconditioner(LinearOperator, gen_fma, gen_graph), meta
        end
    end

    return OptionedPreconditioner(configure)
end

"""
    _build_pyamg(A, solver_fn_name, solver_kwargs; cycle, num_cycles, k_cg_iters, k_early_exit)
        -> (Preconditioner, Dict)

Builds the hierarchy and immediately configures it with the given apply-time parameters.
"""
function _build_pyamg(A::SparseMatrixCSC, solver_fn_name::Symbol, solver_kwargs::Dict;
                      cycle::Symbol=:V, num_cycles::Int=1,
                      k_cg_iters::Int=1, k_early_exit::Real=0.0)
    op = _build_pyamg_hierarchy(A, solver_fn_name, solver_kwargs)
    return op(; cycle, num_cycles, k_cg_iters, k_early_exit)
end

# ── Ruge-Stuben ──────────────────────────────────────────────────────────────

"""
    PyAMGRuStu(A; kwargs...) -> Preconditioner

Classical AMG (Ruge-Stuben) preconditioner via PyAMG.
Returns `Preconditioner` for all cycle types. For K-cycles, the `LinearOperator`
returns the actual per-application FMA cost; for other cycles, it returns a fixed cost.

# Keyword Arguments
- `cycle::Symbol = :V` — Cycle type: `:V`, `:W`, `:F`, `:AMLI`, `:K`
- `num_cycles::Int = 1` — Number of multigrid cycles per apply
- `k_cg_iters::Int = 1` — FCG iterations per level for K-cycles (paper's ℓ)
- `k_early_exit::Float64 = 0.0` — K-cycle early exit threshold on ‖r‖/‖r₀‖ (0 = disabled)
- `coarse_solver::Symbol = :pinv` — Coarse level solver: `:pinv`, `:lu`, `:cholesky`, `:splu`, `:cg`
- `strength` — Strength of connection: `:symmetric`, `:classical`, `:evolution`,
  `:distance`, `:algebraic_distance`, `:affinity`, `:energy_based`, `nothing`.
  Use `(:method, (theta=0.25,))` for sub-options.
- `CF` — C/F splitting: `:RS`, `:PMIS`, `:PMISc`, `:CLJP`, `:CLJPc`, `:CR`.
  Use `(:method, (second_pass=false,))` for sub-options.
- `interpolation` — Interpolation: `:direct`, `:classical`
- `presmoother` — Presmoother specification (see smoother format below)
- `postsmoother` — Postsmoother specification
- `max_levels::Int = 30`
- `max_coarse::Int = 10`

# Smoother Format
- `:gauss_seidel` — method with defaults
- `(:gauss_seidel, (sweep=:forward, iterations=2))` — method with options
- `[:gauss_seidel, (:jacobi, (omega=0.8,))]` — per-level specification

Valid smoother methods: `gauss_seidel`, `block_gauss_seidel`, `jacobi`,
`block_jacobi`, `cf_jacobi`, `fc_jacobi`, `cf_block_jacobi`, `fc_block_jacobi`,
`richardson`, `sor`, `chebyshev`, `gauss_seidel_nr`, `gauss_seidel_ne`,
`jacobi_ne`, `cg`, `gmres`, `cgne`, `cgnr`, `schwarz`,
`strength_based_schwarz`, `nothing`.
"""
function PyAMGRuStu(A::SparseMatrixCSC;
    cycle::Symbol = :V,
    num_cycles::Int = 1,
    k_cg_iters::Int = 1,
    k_early_exit::Real = 0.0,
    coarse_solver::Symbol = :pinv,
    strength = (:classical, (theta=0.25,)),
    CF = (:RS, (second_pass=false,)),
    interpolation::Symbol = :classical,
    presmoother = (:gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:gauss_seidel, (sweep=:symmetric,)),
    max_levels::Int = 30,
    max_coarse::Int = 10,
)
    kwargs = _parse_common_pyamg_opts(; cycle, num_cycles, coarse_solver, presmoother, postsmoother, max_levels, max_coarse)

    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("CF", CF, PYAMG_CF_METHODS)
    interpolation in PYAMG_INTERPOLATION_METHODS || error("Invalid interpolation: $interpolation. Valid: $PYAMG_INTERPOLATION_METHODS")

    kwargs[:strength] = _option_to_py(strength)
    kwargs[:CF] = _option_to_py(CF)
    kwargs[:interpolation] = _to_py(interpolation)

    return _build_pyamg(A, :ruge_stuben_solver, kwargs; cycle, num_cycles, k_cg_iters, k_early_exit)
end

function PyAMGRuStu(A::SparseMatrixCSC, ::Type{OptionedPreconditioner};
    coarse_solver::Symbol = :pinv,
    strength = (:classical, (theta=0.25,)),
    CF = (:RS, (second_pass=false,)),
    interpolation::Symbol = :classical,
    presmoother = (:gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:gauss_seidel, (sweep=:symmetric,)),
    max_levels::Int = 30,
    max_coarse::Int = 10,
    keep::Bool = false,
)
    kwargs = _parse_setup_pyamg_opts(; coarse_solver, presmoother, postsmoother, max_levels, max_coarse)

    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("CF", CF, PYAMG_CF_METHODS)
    interpolation in PYAMG_INTERPOLATION_METHODS || error("Invalid interpolation: $interpolation. Valid: $PYAMG_INTERPOLATION_METHODS")

    kwargs[:strength] = _option_to_py(strength)
    kwargs[:CF] = _option_to_py(CF)
    kwargs[:interpolation] = _to_py(interpolation)
    kwargs[:keep] = keep

    return _build_pyamg_hierarchy(A, :ruge_stuben_solver, kwargs)
end

# ── Smoothed Aggregation ─────────────────────────────────────────────────────

"""
    PyAMGSmAgg(A; kwargs...) -> Preconditioner

Smoothed Aggregation AMG preconditioner via PyAMG.
Returns `Preconditioner` for all cycle types. For K-cycles, the `LinearOperator`
returns the actual per-application FMA cost; for other cycles, it returns a fixed cost.

# Keyword Arguments
- `cycle::Symbol = :V` — Cycle type: `:V`, `:W`, `:F`, `:AMLI`, `:K`
- `num_cycles::Int = 1` — Number of multigrid cycles per apply
- `k_cg_iters::Int = 1` — FCG iterations per level for K-cycles (paper's ℓ)
- `k_early_exit::Float64 = 0.0` — K-cycle early exit threshold on ‖r‖/‖r₀‖ (0 = disabled)
- `coarse_solver::Symbol = :pinv` — Coarse level solver: `:pinv`, `:lu`, `:cholesky`, `:splu`, `:cg`
- `strength` — Strength of connection (same options as `PyAMGRuStu`)
- `aggregate` — Aggregation: `:standard`, `:lloyd`, `:balanced_lloyd`, `:naive`,
  `:pairwise`, `:metis`. Use `(:method, (key=val,))` for sub-options.
- `smooth` — Prolongation smoothing: `:jacobi`, `:richardson`, `:energy`, `nothing`.
  Use `(:jacobi, (omega=4/3,))` for sub-options.
- `presmoother` — Presmoother specification (see `PyAMGRuStu`)
- `postsmoother` — Postsmoother specification
- `improve_candidates` — Candidate improvement smoother, or `nothing`
- `max_levels::Int = 10`
- `max_coarse::Int = 10`
- `diagonal_dominance::Bool = false`
"""
function PyAMGSmAgg(A::SparseMatrixCSC;
    cycle::Symbol = :V,
    num_cycles::Int = 1,
    k_cg_iters::Int = 1,
    k_early_exit::Real = 0.0,
    coarse_solver::Symbol = :pinv,
    strength = :symmetric,
    aggregate = :standard,
    smooth = (:jacobi, (omega=4.0/3.0,)),
    presmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    improve_candidates = (:block_gauss_seidel, (sweep=:symmetric, iterations=4)),
    max_levels::Int = 10,
    max_coarse::Int = 10,
    diagonal_dominance::Bool = false,
)
    kwargs = _parse_common_pyamg_opts(; cycle, num_cycles, coarse_solver, presmoother, postsmoother, max_levels, max_coarse)
    kwargs[:symmetry] = _to_py(:symmetric)

    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("aggregate", aggregate, PYAMG_AGGREGATE_METHODS)
    _validate_option("smooth", smooth, PYAMG_SA_SMOOTH_METHODS)
    improve_candidates !== nothing && _validate_smoother("improve_candidates", improve_candidates)

    kwargs[:strength] = _option_to_py(strength)
    kwargs[:aggregate] = _aggregate_to_py(aggregate)
    kwargs[:smooth] = _option_to_py(smooth)
    kwargs[:improve_candidates] = _option_to_py(improve_candidates)
    kwargs[:diagonal_dominance] = diagonal_dominance

    return _build_pyamg(A, :smoothed_aggregation_solver, kwargs; cycle, num_cycles, k_cg_iters, k_early_exit)
end

function PyAMGSmAgg(A::SparseMatrixCSC, ::Type{OptionedPreconditioner};
    coarse_solver::Symbol = :pinv,
    strength = :symmetric,
    aggregate = :standard,
    smooth = (:jacobi, (omega=4.0/3.0,)),
    presmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    improve_candidates = (:block_gauss_seidel, (sweep=:symmetric, iterations=4)),
    max_levels::Int = 10,
    max_coarse::Int = 10,
    diagonal_dominance::Bool = false,
    keep::Bool = false,
)
    kwargs = _parse_setup_pyamg_opts(; coarse_solver, presmoother, postsmoother, max_levels, max_coarse)
    kwargs[:symmetry] = _to_py(:symmetric)

    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("aggregate", aggregate, PYAMG_AGGREGATE_METHODS)
    _validate_option("smooth", smooth, PYAMG_SA_SMOOTH_METHODS)
    improve_candidates !== nothing && _validate_smoother("improve_candidates", improve_candidates)

    kwargs[:strength] = _option_to_py(strength)
    kwargs[:aggregate] = _aggregate_to_py(aggregate)
    kwargs[:smooth] = _option_to_py(smooth)
    kwargs[:improve_candidates] = _option_to_py(improve_candidates)
    kwargs[:diagonal_dominance] = diagonal_dominance
    kwargs[:keep] = keep

    return _build_pyamg_hierarchy(A, :smoothed_aggregation_solver, kwargs)
end

# ── Rootnode ─────────────────────────────────────────────────────────────────

"""
    PyAMGRootnode(A; kwargs...) -> Preconditioner

Root-node Smoothed Aggregation AMG preconditioner via PyAMG.
Returns `Preconditioner` for all cycle types. For K-cycles, the `LinearOperator`
returns the actual per-application FMA cost; for other cycles, it returns a fixed cost.

Same interface as `PyAMGSmAgg` except `smooth` only supports `:energy` and `nothing`.
"""
function PyAMGRootnode(A::SparseMatrixCSC;
    cycle::Symbol = :V,
    num_cycles::Int = 1,
    k_cg_iters::Int = 1,
    k_early_exit::Real = 0.0,
    coarse_solver::Symbol = :pinv,
    strength = :symmetric,
    aggregate = :standard,
    smooth = :energy,
    presmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    improve_candidates = (:block_gauss_seidel, (sweep=:symmetric, iterations=4)),
    max_levels::Int = 10,
    max_coarse::Int = 10,
    diagonal_dominance::Bool = false,
)
    kwargs = _parse_common_pyamg_opts(; cycle, num_cycles, coarse_solver, presmoother, postsmoother, max_levels, max_coarse)
    kwargs[:symmetry] = _to_py(:symmetric)

    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("aggregate", aggregate, PYAMG_AGGREGATE_METHODS)
    _validate_option("smooth", smooth, PYAMG_ROOTNODE_SMOOTH_METHODS)
    improve_candidates !== nothing && _validate_smoother("improve_candidates", improve_candidates)

    kwargs[:strength] = _option_to_py(strength)
    kwargs[:aggregate] = _aggregate_to_py(aggregate)
    kwargs[:smooth] = _option_to_py(smooth)
    kwargs[:improve_candidates] = _option_to_py(improve_candidates)
    kwargs[:diagonal_dominance] = diagonal_dominance

    return _build_pyamg(A, :rootnode_solver, kwargs; cycle, num_cycles, k_cg_iters, k_early_exit)
end

function PyAMGRootnode(A::SparseMatrixCSC, ::Type{OptionedPreconditioner};
    coarse_solver::Symbol = :pinv,
    strength = :symmetric,
    aggregate = :standard,
    smooth = :energy,
    presmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    improve_candidates = (:block_gauss_seidel, (sweep=:symmetric, iterations=4)),
    max_levels::Int = 10,
    max_coarse::Int = 10,
    diagonal_dominance::Bool = false,
    keep::Bool = false,
)
    kwargs = _parse_setup_pyamg_opts(; coarse_solver, presmoother, postsmoother, max_levels, max_coarse)
    kwargs[:symmetry] = _to_py(:symmetric)

    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("aggregate", aggregate, PYAMG_AGGREGATE_METHODS)
    _validate_option("smooth", smooth, PYAMG_ROOTNODE_SMOOTH_METHODS)
    improve_candidates !== nothing && _validate_smoother("improve_candidates", improve_candidates)

    kwargs[:strength] = _option_to_py(strength)
    kwargs[:aggregate] = _aggregate_to_py(aggregate)
    kwargs[:smooth] = _option_to_py(smooth)
    kwargs[:improve_candidates] = _option_to_py(improve_candidates)
    kwargs[:diagonal_dominance] = diagonal_dominance
    kwargs[:keep] = keep

    return _build_pyamg_hierarchy(A, :rootnode_solver, kwargs)
end

# ── Pairwise ─────────────────────────────────────────────────────────────────

"""
    PyAMGPairwise(A; kwargs...) -> Preconditioner

Pairwise Aggregation AMG preconditioner via PyAMG.
Returns `Preconditioner` for all cycle types. For K-cycles, the `LinearOperator`
returns the actual per-application FMA cost; for other cycles, it returns a fixed cost.

# Keyword Arguments
- `cycle::Symbol = :V` — Cycle type: `:V`, `:W`, `:F`, `:AMLI`, `:K`
- `num_cycles::Int = 1` — Number of multigrid cycles per apply
- `k_cg_iters::Int = 1` — FCG iterations per level for K-cycles (paper's ℓ)
- `k_early_exit::Float64 = 0.0` — K-cycle early exit threshold on ‖r‖/‖r₀‖ (0 = disabled)
- `theta::Float64 = 0.25` — Pairwise aggregation strength threshold
- `norm::Symbol = :min` — Norm for pairwise matching: `:min`, `:abs`, `:max`
- `matchings::Int = 2` — Number of pairwise matchings per level
- `presmoother` — Presmoother specification (see `PyAMGRuStu`)
- `postsmoother` — Postsmoother specification
- `max_levels::Int = 20`
- `max_coarse::Int = 10`
"""
function PyAMGPairwise(A::SparseMatrixCSC;
    cycle::Symbol = :V,
    num_cycles::Int = 1,
    k_cg_iters::Int = 1,
    k_early_exit::Real = 0.0,
    coarse_solver::Symbol = :pinv,
    theta::Real = 0.25,
    norm::Symbol = :min,
    matchings::Int = 2,
    presmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    max_levels::Int = 20,
    max_coarse::Int = 10,
)
    kwargs = _parse_common_pyamg_opts(; cycle, num_cycles, coarse_solver, presmoother, postsmoother, max_levels, max_coarse)

    norm in (:min, :abs) || error("Invalid norm: $norm. Valid: :min, :abs")
    matchings >= 1 || error("matchings must be >= 1, got $matchings")

    kwargs[:aggregate] = _aggregate_to_py((:pairwise, (theta=theta, norm=norm, matchings=matchings)))

    return _build_pyamg(A, :pairwise_solver, kwargs; cycle, num_cycles, k_cg_iters, k_early_exit)
end

function PyAMGPairwise(A::SparseMatrixCSC, ::Type{OptionedPreconditioner};
    coarse_solver::Symbol = :pinv,
    theta::Real = 0.25,
    norm::Symbol = :min,
    matchings::Int = 2,
    presmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    postsmoother = (:block_gauss_seidel, (sweep=:symmetric,)),
    max_levels::Int = 20,
    max_coarse::Int = 10,
)
    kwargs = _parse_setup_pyamg_opts(; coarse_solver, presmoother, postsmoother, max_levels, max_coarse)

    norm in (:min, :abs) || error("Invalid norm: $norm. Valid: :min, :abs")
    matchings >= 1 || error("matchings must be >= 1, got $matchings")

    kwargs[:aggregate] = _aggregate_to_py((:pairwise, (theta=theta, norm=norm, matchings=matchings)))

    return _build_pyamg_hierarchy(A, :pairwise_solver, kwargs)
end

# ── Adaptive SA ──────────────────────────────────────────────────────────────

"""
    PyAMGAdaptive(A; kwargs...) -> Preconditioner

Adaptive Smoothed Aggregation AMG preconditioner via PyAMG.
Automatically discovers near-nullspace candidates via relaxation.

# Keyword Arguments
- `cycle::Symbol = :V` — Cycle type: `:V`, `:W`, `:F`, `:AMLI`, `:K`
- `num_cycles::Int = 1` — Number of multigrid cycles per apply
- `k_cg_iters::Int = 1` — FCG iterations per level for K-cycles
- `k_early_exit::Float64 = 0.0` — K-cycle early exit threshold
- `coarse_solver::Symbol = :pinv` — Coarse level solver
- `strength` — Strength of connection (same options as `PyAMGRuStu`)
- `aggregate` — Aggregation method (same options as `PyAMGSmAgg`)
- `smooth` — Prolongation smoothing: `:jacobi`, `:richardson`, `:energy`, `nothing`
- `prepostsmoother` — Smoother used for both pre and post (see `PyAMGRuStu`)
- `num_candidates::Int = 1` — Number of near-nullspace candidates to generate
- `candidate_iters::Int = 5` — Smoothing passes per level in adaptive setup
- `improvement_iters::Int = 0` — Number of candidate improvement passes
- `epsilon::Float64 = 0.1` — Target convergence factor
- `max_levels::Int = 10`
- `max_coarse::Int = 10`
"""
function PyAMGAdaptive(A::SparseMatrixCSC;
    cycle::Symbol = :V,
    num_cycles::Int = 1,
    k_cg_iters::Int = 1,
    k_early_exit::Real = 0.0,
    coarse_solver::Symbol = :pinv,
    strength = :symmetric,
    aggregate = :standard,
    smooth = (:jacobi, NamedTuple()),
    prepostsmoother = (:gauss_seidel, (sweep=:symmetric,)),
    num_candidates::Int = 1,
    candidate_iters::Int = 5,
    improvement_iters::Int = 0,
    epsilon::Real = 0.1,
    max_levels::Int = 10,
    max_coarse::Int = 10,
)
    cycle in PYAMG_CYCLES || error("Invalid cycle: $cycle. Valid: $PYAMG_CYCLES")
    num_cycles >= 1 || error("num_cycles must be >= 1, got $num_cycles")
    coarse_solver in PYAMG_COARSE_SOLVERS || error("Invalid coarse_solver: $coarse_solver. Valid: $PYAMG_COARSE_SOLVERS")
    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("aggregate", aggregate, PYAMG_AGGREGATE_METHODS)
    _validate_option("smooth", smooth, PYAMG_SA_SMOOTH_METHODS)
    _validate_smoother("prepostsmoother", prepostsmoother)

    kwargs = Dict{Symbol, Any}(
        :symmetry => _to_py(:symmetric),
        :pdef => true,
        :coarse_solver => _to_py(coarse_solver),
        :max_levels => max_levels,
        :max_coarse => max_coarse,
        :strength => _option_to_py(strength),
        :aggregate => _option_to_py(aggregate),
        :smooth => _option_to_py(smooth),
        :prepostsmoother => _option_to_py(prepostsmoother),
        :num_candidates => num_candidates,
        :candidate_iters => candidate_iters,
        :improvement_iters => improvement_iters,
        :epsilon => epsilon,
    )

    return _build_pyamg(A, :adaptive_sa_solver, kwargs; cycle, num_cycles, k_cg_iters, k_early_exit)
end

function PyAMGAdaptive(A::SparseMatrixCSC, ::Type{OptionedPreconditioner};
    coarse_solver::Symbol = :pinv,
    strength = :symmetric,
    aggregate = :standard,
    smooth = (:jacobi, NamedTuple()),
    prepostsmoother = (:gauss_seidel, (sweep=:symmetric,)),
    num_candidates::Int = 1,
    candidate_iters::Int = 5,
    improvement_iters::Int = 0,
    epsilon::Real = 0.1,
    max_levels::Int = 10,
    max_coarse::Int = 10,
    keep::Bool = false,
)
    coarse_solver in PYAMG_COARSE_SOLVERS || error("Invalid coarse_solver: $coarse_solver. Valid: $PYAMG_COARSE_SOLVERS")
    _validate_option("strength", strength, PYAMG_STRENGTH_METHODS)
    _validate_option("aggregate", aggregate, PYAMG_AGGREGATE_METHODS)
    _validate_option("smooth", smooth, PYAMG_SA_SMOOTH_METHODS)
    _validate_smoother("prepostsmoother", prepostsmoother)

    kwargs = Dict{Symbol, Any}(
        :symmetry => _to_py(:symmetric),
        :pdef => true,
        :coarse_solver => _to_py(coarse_solver),
        :max_levels => max_levels,
        :max_coarse => max_coarse,
        :strength => _option_to_py(strength),
        :aggregate => _option_to_py(aggregate),
        :smooth => _option_to_py(smooth),
        :prepostsmoother => _option_to_py(prepostsmoother),
        :num_candidates => num_candidates,
        :candidate_iters => candidate_iters,
        :improvement_iters => improvement_iters,
        :epsilon => epsilon,
        :keep => keep,
    )

    return _build_pyamg_hierarchy(A, :adaptive_sa_solver, kwargs)
end

end # module Preconditioners
