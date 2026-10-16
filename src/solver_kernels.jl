# Solver fusion kernels. No fast-math assumptions; each workspace belongs
# to one sequential solve. Padding separates per-task accumulators.
struct FusedReductionWorkspace
    partials::Matrix{Float64}
end
FusedReductionWorkspace() = FusedReductionWorkspace(zeros(16, Threads.nthreads()))

"""
    threaded_dot(x, y, workspace)

Real Float64 dot product using the solver's Julia thread pool, not BLAS.
The caller owns the workspace; concurrent reductions require separate workspaces.
"""
function threaded_dot(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                      ws::FusedReductionWorkspace)
    axes(x) == axes(y) || throw(DimensionMismatch("dot product axes differ"))
    Base.require_one_based_indexing(x, y)
    n = length(x)
    nt = size(ws.partials, 2)
    chunk = cld(n, nt)
    partials = ws.partials
    @batch per=core for part in 1:nt
        s = 0.0
        @inbounds @simd for i in (part-1)*chunk+1:min(part*chunk,n)
            s = muladd(x[i], y[i], s)
        end
        partials[1,part] = s
    end
    s = 0.0
    @inbounds for part in 1:nt
        s += partials[1,part]
    end
    return s
end

# Branch before constructing Val so inference sees the concrete return type;
# Val(runtime_bool) otherwise boxes the six-scalar result in the solver loop.
@inline function _fused_diagnostics!(r_true, Ax, b, seed, x, r,
        ws::FusedReductionWorkspace, compute_x::Bool)
    if compute_x
        return _fused_diagnostics!(r_true, Ax, b, seed, x, r, ws, Val(true))
    end
    return _fused_diagnostics!(r_true, Ax, b, seed, x, r, ws, Val(false))
end

function _fused_diagnostics!(r_true, Ax, b, seed, x, r,
        ws::FusedReductionWorkspace, ::Val{ComputeX}) where {ComputeX}
    n = length(x)
    nt = size(ws.partials, 2)
    chunk = cld(n, nt)
    partials = ws.partials
    @batch per=core for part in 1:nt
        sr = 0.0; se = 0.0; sx = 0.0; sn = 0.0; xb = 0.0; xAx = 0.0
        @inbounds @simd for i in (part-1)*chunk+1:min(part*chunk,n)
            xi, ai = x[i], Ax[i]
            ri = b[i] - ai
            ei = seed[i] - xi
            r_true[i] = ri
            Ax[i] = ei
            sr = muladd(ri, ri, sr)
            se = muladd(ei, ei, se)
            if ComputeX
                sx = muladd(xi, xi, sx)
            end
            if r !== nothing
                sn = muladd(r[i], r[i], sn)
            end
            xb = muladd(xi, b[i], xb)
            xAx = muladd(xi, ai, xAx)
        end
        partials[1,part] = sr; partials[2,part] = se
        partials[3,part] = sx; partials[4,part] = sn
        partials[5,part] = xb; partials[6,part] = xAx
    end
    sr = 0.0; se = 0.0; sx = 0.0; sn = 0.0; xb = 0.0; xAx = 0.0
    @inbounds for part in 1:nt
        sr += partials[1,part]; se += partials[2,part]
        sx += partials[3,part]; sn += partials[4,part]
        xb += partials[5,part]; xAx += partials[6,part]
    end
    return (norm_true=_norm_from_squared(sr, r_true), norm_error=_norm_from_squared(se, Ax),
            norm_x=ComputeX ? _norm_from_squared(sx, x) : 0.0,
            norm_native=r === nothing ? 0.0 : _norm_from_squared(sn, r), xb=xb, xAx=xAx)
end

function _fused_update_norm!(x, alpha, p, ws::FusedReductionWorkspace,
                             r=nothing, Ap=nothing, base=nothing, candidate=x)
    n = length(x)
    nt = size(ws.partials, 2)
    chunk = cld(n, nt)
    partials = ws.partials
    @batch per=core for part in 1:nt
        sx = 0.0
        @inbounds @simd for i in (part-1)*chunk+1:min(part*chunk,n)
            xi = muladd(alpha, p[i], x[i])
            x[i] = xi
            global_xi = base === nothing ? xi : base[i] + xi
            if base !== nothing
                candidate[i] = global_xi
            end
            if r !== nothing
                r[i] = muladd(-alpha, Ap[i], r[i])
            end
            sx = muladd(global_xi, global_xi, sx)
        end
        partials[1,part] = sx
    end
    sx = 0.0
    @inbounds for part in 1:nt
        sx += partials[1,part]
    end
    return _norm_from_squared(sx, candidate)
end

# Retained only for direct-kernel A/B validation; production uses _spmv_plan.
struct ExperimentalSplitSpMVPlan
    colptr::Vector{Int}
    starts::Vector{Int}
    stops::Vector{Int}
    first_rows::Vector{Int}
    last_rows::Vector{Int}
    partials::Matrix{Float64}
    nthreads::Int
end

function _experimental_split_spmv_plan(A::SparseMatrixCSC{Float64,Int})
    nt = min(Threads.nthreads(), nnz(A))
    nt <= 1 && return nothing
    starts = [fld((part-1)*nnz(A), nt)+1 for part in 1:nt]
    stops = [fld(part*nnz(A), nt) for part in 1:nt]
    first_rows = [searchsortedlast(A.colptr, p) for p in starts]
    last_rows = [searchsortedlast(A.colptr, p) for p in stops]
    return ExperimentalSplitSpMVPlan(A.colptr, starts, stops, first_rows,
        last_rows, zeros(16,nt), Threads.nthreads())
end

function threaded_spmv!(y::AbstractVector{Float64}, A::SparseMatrixCSC{Float64,Int},
                        x::AbstractVector{Float64}, plan::ExperimentalSplitSpMVPlan)
    plan.colptr === A.colptr && plan.nthreads == Threads.nthreads() ||
        throw(ArgumentError("split SpMV plan requires its original structure and thread configuration"))
    # Covers empty rows and initializes the boundary-row accumulations. This
    # extra O(n) write is deliberately included in experimental timing.
    fill!(y, 0.0)
    ptr, rows, vals = A.colptr, A.rowval, A.nzval
    starts, stops = plan.starts, plan.stops
    first_rows, last_rows, partials = plan.first_rows, plan.last_rows, plan.partials
    @batch per=core for part in eachindex(starts)
        first_row, last_row = first_rows[part], last_rows[part]
        for row in first_row:last_row
            s = 0.0
            @inbounds @simd for p in max(ptr[row],starts[part]):min(ptr[row+1]-1,stops[part])
                s = muladd(vals[p],x[rows[p]],s)
            end
            if row == first_row
                partials[1,part] = s
            elseif row == last_row
                partials[2,part] = s
            else
                y[row] = s
            end
        end
    end
    # Workers never write boundary rows. Combine their fragments only after
    # joining, in segment order, including when one row spans many segments.
    @inbounds for part in eachindex(plan.starts)
        y[plan.first_rows[part]] += plan.partials[1,part]
        if plan.first_rows[part] != plan.last_rows[part]
            y[plan.last_rows[part]] += plan.partials[2,part]
        end
    end
    return y
end

threaded_spmv!(y::AbstractVector{Float64}, At::Adjoint{Float64,SparseMatrixCSC{Float64,Int}},
               x::AbstractVector{Float64}, plan::ExperimentalSplitSpMVPlan) =
    threaded_spmv!(y, parent(At), x, plan)
