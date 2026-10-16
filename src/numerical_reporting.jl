# Public wrappers preserve the numerical routines' return values and keyword defaults.
function estimate_eigbounds(A::SparseMatrixCSC; progress_callback = NOOP_PHASE_CALLBACK,
                            progress_every_iters::Int = 100,
                            progress_min_seconds::Real = 60.0, kwargs...)
    run_scoped_phase(progress_callback, "eigenvalue_estimation"; n=size(A, 1)) do child
        p = NumericalProgress(child; every_iters=progress_every_iters, min_seconds=progress_min_seconds)
        result = _estimate_eigbounds(A; progress=p, kwargs...)
        if progress_due!(p, result[4]; final=true)
            emit_numerical_progress!(p; iteration=result[4], upper_estimate=result[2],
                upper_ritz_residual=result[6], estimate_iteration=result[4],
                total_fma=result[3], final=true)
        end
        result
    end
end

@doc (@doc _estimate_eigbounds) estimate_eigbounds
@doc (@doc _lanczos_quadrature) lanczos_quadrature

function lanczos_quadrature(A::SparseMatrixCSC, b::Vector{Float64};
                            progress_callback = NOOP_PHASE_CALLBACK,
                            progress_every_iters::Int = 100,
                            progress_min_seconds::Real = 60.0, kwargs...)
    run_scoped_phase(progress_callback, "rhs_quadrature"; n=size(A, 1)) do child
        p = NumericalProgress(child; every_iters=progress_every_iters, min_seconds=progress_min_seconds)
        result = _lanczos_quadrature(A, b; progress=p, kwargs...)
        if progress_due!(p, result[7]; final=true)
            emit_numerical_progress!(p; iteration=result[7], quadrature_value=result[1],
                total_fma=result[6], final=true)
        end
        result
    end
end
