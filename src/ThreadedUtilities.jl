module ThreadedUtilities

using SparseArrays
using LinearAlgebra
using Polyester

export threaded_spmv!, threaded_norm, threaded_dot

# Call only after a completed Julia parallel region, from a sequential numerical
# driver. The next @batch wakes these workers; native teams keep their sizes.
function _sleep_julia_workers!()
    if Threads.nthreads() > 1
        Polyester.ThreadingUtilities.sleep_all_tasks()
    end
    return nothing
end

function _sleep_julia_workers_for_blas!()
    BLAS.get_num_threads() > 1 && _sleep_julia_workers!()
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Threaded SpMV (row-parallel, symmetric A)
# ─────────────────────────────────────────────────────────────────

"""
    threaded_spmv!(y, A::SparseMatrixCSC, x)

Compute y = A * x using row-parallel threaded SpMV.
Assumes A is symmetric so CSC === CSR.
"""
function threaded_spmv!(y::AbstractVector{Float64}, A::SparseMatrixCSC{Float64,Int}, x::AbstractVector{Float64})
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    n = size(A, 1)

    @batch per=core for i in 1:n
        @inbounds begin
            s = 0.0
            @simd for p in colptr[i]:(colptr[i+1]-1)
                s = muladd(nzval[p], x[rowval[p]], s)
            end
            y[i] = s
        end
    end

    return y
end

# Accept Adjoint of symmetric SparseMatrixCSC (A' === A for symmetric matrices)
function threaded_spmv!(y::AbstractVector{Float64}, At::Adjoint{Float64, SparseMatrixCSC{Float64,Int}}, x::AbstractVector{Float64})
    return threaded_spmv!(y, parent(At), x)
end

# ─────────────────────────────────────────────────────────────────
# Threaded 2-norm with an exceptional-range scaled fallback.
# ─────────────────────────────────────────────────────────────────

"""
    threaded_norm(x)

Compute the 2-norm of `x` using threaded `@simd` + `muladd` reduction.
`@batch reduction` iterates over thread IDs (not elements) so the inner
`@simd` loop retains vectorization. Zero allocations.
"""
function threaded_norm(x::AbstractVector{Float64})
    nt = Threads.nthreads()
    n = length(x)
    s = 0.0
    @batch per=core reduction=(+, s) for tid in 1:nt
        chunk = (n + nt - 1) ÷ nt
        lo = (tid - 1) * chunk + 1
        hi = min(tid * chunk, n)
        local_s = 0.0
        @inbounds @simd for i in lo:hi
            local_s = muladd(x[i], x[i], local_s)
        end
        s += local_s
    end
    return _norm_from_squared(s, x)
end

# Preserve the ordinary reduction path; rescale only where squaring can lose
# relative accuracy or overflow. This fallback does not enter BLAS.
function _scaled_norm(x::AbstractVector{Float64})
    scale = 0.0
    for xi in x
        scale = max(scale, abs(xi))
    end
    (iszero(scale) || !isfinite(scale)) && return scale
    s = 0.0
    @inbounds @simd for i in eachindex(x)
        y = x[i] / scale
        s = muladd(y, y, s)
    end
    return scale * sqrt(s)
end

@inline function _norm_from_squared(s::Float64, x::AbstractVector{Float64})
    isfinite(s) && s >= floatmin(Float64) * max(1, length(x)) && return sqrt(s)
    return _scaled_norm(x)
end

# A plan belongs to an immutable CSC structure and one thread configuration.
# No row is split: the arithmetic within each row is the legacy kernel.
struct SpMVPlan
    colptr::Vector{Int}
    boundaries::Vector{Int}
    nthreads::Int
end

function _spmv_plan(A::SparseMatrixCSC; force=false)
    n = size(A,1)
    nt = min(n, Threads.nthreads())
    n == 0 && return nothing
    !force && (nt == 1 || nnz(A) < 100_000) && return nothing
    prefix(i) = A.colptr[i] + i - 2
    total = nnz(A) + n
    chunk = cld(n, nt)
    heaviest = maximum(prefix(min(n+1, lo+chunk)) - prefix(lo) for lo in 1:chunk:n)
    !force && heaviest <= 1.25 * (total / nt) && return nothing
    boundaries = Vector{Int}(undef, nt+1)
    boundaries[1], boundaries[end] = 1, n+1
    for part in 1:nt-1
        target = div(total * part, nt)
        lo, hi = 1, n+1
        while lo < hi
            mid = (lo + hi) >> 1
            if prefix(mid) < target
                lo = mid + 1
            else
                hi = mid
            end
        end
        boundaries[part+1] = lo
    end
    return SpMVPlan(A.colptr, boundaries, Threads.nthreads())
end

threaded_spmv!(y, A, x, ::Nothing) = threaded_spmv!(y, A, x)

function threaded_spmv!(y::AbstractVector{Float64}, A::SparseMatrixCSC{Float64,Int},
                        x::AbstractVector{Float64}, plan::SpMVPlan)
    plan.colptr === A.colptr && plan.nthreads == Threads.nthreads() ||
        throw(ArgumentError("SpMV plan requires the original immutable structure and thread configuration"))
    colptr, rowval, nzval = A.colptr, A.rowval, A.nzval
    bounds = plan.boundaries
    @batch per=core for part in 1:length(bounds)-1
        for i in bounds[part]:bounds[part+1]-1
            @inbounds begin
                s = 0.0
                @simd for p in colptr[i]:(colptr[i+1]-1)
                    s = muladd(nzval[p], x[rowval[p]], s)
                end
                y[i] = s
            end
        end
    end
    return y
end

threaded_spmv!(y::AbstractVector{Float64}, At::Adjoint{Float64,SparseMatrixCSC{Float64,Int}},
               x::AbstractVector{Float64}, plan::SpMVPlan) =
    threaded_spmv!(y, parent(At), x, plan)

# Ax is dead as a matrix product before this call. Keep reductions separate so
# their SIMD/reduction trees remain identical to the diagnostic reference.
function _residual_and_error!(r_true, Ax, b, seed, x)
    n = length(x)
    if n < 65536
        @inbounds @simd for i in 1:n
            a = Ax[i]
            r_true[i] = b[i] - a
            Ax[i] = seed[i] - x[i]
        end
    else
        @batch per=core for i in 1:n
            @inbounds begin
                a = Ax[i]
                r_true[i] = b[i] - a
                Ax[i] = seed[i] - x[i]
            end
        end
    end
    return nothing
end

include("solver_kernels.jl")

end # module ThreadedUtilities
