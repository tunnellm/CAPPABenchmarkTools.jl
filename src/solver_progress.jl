# Progress reads existing diagnostic rows; it never computes or certifies residuals.
function _report_solver_progress!(p, data, row, cols, norm_b, maxiters, work_limit,
                                  total_fma, check_iters, check_count, restarts;
                                  final=false, status=nothing, policy=nothing)
    iteration = Int(data[row, cols.iteration])
    crossing = policy !== nothing && policy.crossing_report_pending
    due = progress_due!(p, iteration; final=final)
    if crossing && phase_reporting_enabled(p.callback)
        p.last_report = p.clock()
        policy.crossing_report_pending = false
    elseif !due
        return nothing
    end
    now = Float64(time_ns()) / 1e9
    policy_fields = if policy === nothing
        (;)
    else
        dt = now - policy.last_report_at
        rate = dt > 0 ? (total_fma - policy.last_report_fma) / dt : nothing
        policy.last_report_at = now
        policy.last_report_fma = total_fma
        cap = policy.continuation_work_limit
        effective_cap = policy.work_limit > 0 ?
            (cap === nothing ? policy.work_limit : min(cap, policy.work_limit)) : cap
        (solver_mode=string(policy.mode), native_candidate_iter=policy.native_candidate_iter,
         native_candidate_total_fma=policy.native_candidate_total_fma,
         continuation_work_limit=cap, effective_work_limit=effective_cap,
         work_fraction=effective_cap === nothing ? nothing : total_fma / effective_cap,
         elapsed_seconds=now-policy.started_at, interval_fma_per_second=rate,
         policy_event=crossing ? "native_candidate" : final ? "terminal" : "progress")
    end
    checked = check_count > 0 && check_iters[check_count] == iteration
    emit_numerical_progress!(p; iteration=iteration, maxiters=maxiters,
        native_relative_residual=data[row, cols.native] / norm_b,
        true_relative_residual=data[row, cols.true_rr],
        relative_normwise_2=data[row, cols.normwise],
        true_residual_role=checked ? "certification" : "diagnostic",
        certification_charged=checked, certification_checks=check_count,
        correction_restarts=restarts,
        krylov_block=cols.block === nothing ? nothing : Int(data[row, cols.block]),
        total_fma=total_fma, work_limit=work_limit > 0 ? work_limit : nothing,
        status=status, final=final, policy_fields..., _diagnostic_progress(data,row)...)
    return nothing
end

function pcg(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                 progress_callback=NOOP_PHASE_CALLBACK, progress_every_iters::Int=100,
                 progress_min_seconds::Real=60.0, kwargs...)
    run_scoped_phase(progress_callback, "solver_kernel"; solver="pcg", n=size(A, 1)) do child
        cadence = get(kwargs, :solver_mode, :certified) == :bounded_continuation ? 1 : progress_every_iters
        progress = NumericalProgress(child; every_iters=cadence,
                                     min_seconds=progress_min_seconds)
        result = all(iszero, b) ?
            _zero_rhs_result(PCG_COLUMN_NAMES, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                             progress=progress, kwargs...) :
            _pcg_with_progress(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                              progress=progress, kwargs...)
        # Initial breakdown may return before any diagnostic row exists.
        if !progress.final_reported && progress_due!(progress, result.iterations; final=true)
            emit_numerical_progress!(progress; iteration=result.iterations,
                total_fma=result.total_fma, status=string(result.status), final=true)
        end
        result
    end
end

# Keep the existing public solver documentation on the reporting wrappers.
@doc (@doc _pcg_with_progress) pcg
@doc (@doc _lanczos_pcg_with_progress) lanczos_pcg
@doc (@doc _minres_with_progress) minres
@doc (@doc _pcr_with_progress) pcr
@doc (@doc _fpcg1_with_progress) fpcg1
@doc (@doc _fpcg2_with_progress) fpcg2

function lanczos_pcg(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                 progress_callback=NOOP_PHASE_CALLBACK, progress_every_iters::Int=100,
                 progress_min_seconds::Real=60.0, kwargs...)
    run_scoped_phase(progress_callback, "solver_kernel"; solver="lanczos_pcg", n=size(A, 1)) do child
        cadence = get(kwargs, :solver_mode, :certified) == :bounded_continuation ? 1 : progress_every_iters
        progress = NumericalProgress(child; every_iters=cadence,
                                     min_seconds=progress_min_seconds)
        result = all(iszero, b) ?
            _zero_rhs_result(LPCG_COLUMN_NAMES, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                             progress=progress, kwargs...) :
            _lanczos_pcg_with_progress(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                              progress=progress, kwargs...)
        # Initial breakdown may return before any diagnostic row exists.
        if !progress.final_reported && progress_due!(progress, result.iterations; final=true)
            emit_numerical_progress!(progress; iteration=result.iterations,
                total_fma=result.total_fma, status=string(result.status), final=true)
        end
        result
    end
end

function minres(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                 progress_callback=NOOP_PHASE_CALLBACK, progress_every_iters::Int=100,
                 progress_min_seconds::Real=60.0, kwargs...)
    run_scoped_phase(progress_callback, "solver_kernel"; solver="minres", n=size(A, 1)) do child
        cadence = get(kwargs, :solver_mode, :certified) == :bounded_continuation ? 1 : progress_every_iters
        progress = NumericalProgress(child; every_iters=cadence,
                                     min_seconds=progress_min_seconds)
        result = all(iszero, b) ?
            _zero_rhs_result(MINRES_COLUMN_NAMES, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                             progress=progress, kwargs...) :
            _minres_with_progress(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                              progress=progress, kwargs...)
        # Initial breakdown may return before any diagnostic row exists.
        if !progress.final_reported && progress_due!(progress, result.iterations; final=true)
            emit_numerical_progress!(progress; iteration=result.iterations,
                total_fma=result.total_fma, status=string(result.status), final=true)
        end
        result
    end
end

function pcr(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                 progress_callback=NOOP_PHASE_CALLBACK, progress_every_iters::Int=100,
                 progress_min_seconds::Real=60.0, kwargs...)
    run_scoped_phase(progress_callback, "solver_kernel"; solver="pcr", n=size(A, 1)) do child
        cadence = get(kwargs, :solver_mode, :certified) == :bounded_continuation ? 1 : progress_every_iters
        progress = NumericalProgress(child; every_iters=cadence,
                                     min_seconds=progress_min_seconds)
        result = all(iszero, b) ?
            _zero_rhs_result(PCR_COLUMN_NAMES, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                             progress=progress, kwargs...) :
            _pcr_with_progress(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                              progress=progress, kwargs...)
        # Initial breakdown may return before any diagnostic row exists.
        if !progress.final_reported && progress_due!(progress, result.iterations; final=true)
            emit_numerical_progress!(progress; iteration=result.iterations,
                total_fma=result.total_fma, status=string(result.status), final=true)
        end
        result
    end
end

function fpcg1(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                 progress_callback=NOOP_PHASE_CALLBACK, progress_every_iters::Int=100,
                 progress_min_seconds::Real=60.0, kwargs...)
    run_scoped_phase(progress_callback, "solver_kernel"; solver="fpcg1", n=size(A, 1)) do child
        cadence = get(kwargs, :solver_mode, :certified) == :bounded_continuation ? 1 : progress_every_iters
        progress = NumericalProgress(child; every_iters=cadence,
                                     min_seconds=progress_min_seconds)
        result = all(iszero, b) ?
            _zero_rhs_result(FPCG_COLUMN_NAMES, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                             progress=progress, kwargs...) :
            _fpcg1_with_progress(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                              progress=progress, kwargs...)
        # Initial breakdown may return before any diagnostic row exists.
        if !progress.final_reported && progress_due!(progress, result.iterations; final=true)
            emit_numerical_progress!(progress; iteration=result.iterations,
                total_fma=result.total_fma, status=string(result.status), final=true)
        end
        result
    end
end

function fpcg2(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                 progress_callback=NOOP_PHASE_CALLBACK, progress_every_iters::Int=100,
                 progress_min_seconds::Real=60.0, kwargs...)
    run_scoped_phase(progress_callback, "solver_kernel"; solver="fpcg2", n=size(A, 1)) do child
        cadence = get(kwargs, :solver_mode, :certified) == :bounded_continuation ? 1 : progress_every_iters
        progress = NumericalProgress(child; every_iters=cadence,
                                     min_seconds=progress_min_seconds)
        result = all(iszero, b) ?
            _zero_rhs_result(FPCG_COLUMN_NAMES, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                             progress=progress, kwargs...) :
            _fpcg2_with_progress(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
                              progress=progress, kwargs...)
        # Initial breakdown may return before any diagnostic row exists.
        if !progress.final_reported && progress_due!(progress, result.iterations; final=true)
            emit_numerical_progress!(progress; iteration=result.iterations,
                total_fma=result.total_fma, status=string(result.status), final=true)
        end
        result
    end
end
