# Called only when every RHS entry is exactly zero, never from a norm that
# could have underflowed. No preconditioner or Krylov recurrence is needed.
function _zero_rhs_result(columns, A::SparseMatrixCSC, b::Vector{Float64},
        seed::Vector{Float64}, spectral_norm::Float64, prec::Preconditioner,
        maxiters::Int64, raw_out::String, conv_out::String;
        work_limit::Int64=-1, convergence_tolerance::Float64=minimum_tolerance,
        correction_restart_normwise_tolerance::Float64=correction_restart_normwise_tolerance_default,
        write_solution::Union{Nothing,String}=nothing,
        _experimental_final_iterate::Union{Nothing,Vector{Float64}}=nothing,
        _diagnostic_block_rows::Int=DIAGNOSTIC_BLOCK_ROWS,
        progress=NumericalProgress(NOOP_PHASE_CALLBACK),
        solver_mode::Symbol=:certified, continuation_work_factor::Float64=2.0)
    n = length(b)
    size(A) == (n,n) && length(seed) == n || throw(DimensionMismatch("solver input dimensions"))
    maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
    isfinite(convergence_tolerance) && convergence_tolerance >= 0 ||
        throw(ArgumentError("convergence_tolerance must be finite and nonnegative"))
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    Threads.atomic_xchg!(KILL_FLAG, false)
    records = Dict{String,Any}[]
    _record_certification_check!(records, 0, "initial_solution", 0.0, 0.0, 0.0, 0.0, 0.0;
        certified=true, action="accept_convergence", source="zero_rhs_identity", charged=false)

    run_phase(progress.callback, "result_write") do
        data = DiagnosticBlockWriter(raw_out, 1, columns;
            block_rows=_diagnostic_block_rows, callback=progress.callback)
        # Recurrence scalars/spectral estimates do not exist without a step.
        for col in eachindex(columns)
            data[1,col] = NaN
        end
        for col in 1:7
            data[1,col] = 0.0
        end
        data[1,PCG_COL_ERROR] = all(iszero, seed) ? 0.0 : 1.0
        for name in ("norm_r", "norm_r_true", "energy", "krylov_block", "diag_abs_flag")
            col = findfirst(==(name), columns)
            col === nothing || (data[1,col] = 0.0)
        end
        used = _finish_diagnostic_data!(data, 1)
        open(replace(raw_out, r"\.[^/]*$" => "_columns.json"), "w") do io
            JSON3.write(io, columns)
        end
        _write_convergence(conv_out * "_relative_residual.csv", used, PCG_COL_RELATIVE_RESIDUAL)
        _write_convergence(conv_out * "_relative_normwise_2.csv", used, PCG_COL_RELATIVE_NORMWISE_2)
        _write_certification_convergence(conv_out * "_certification.csv", used,
            PCG_COL_NORM_R, 0.0, records, Int[]; n=n, base_fma=nnz(A),
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor,
            work_limit=work_limit,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance)
        write_solution === nothing || _write_solution_vec(write_solution, zeros(n))
    end
    if progress_due!(progress, 0; final=true)
        emit_numerical_progress!(progress; iteration=0, maxiters=maxiters,
            native_relative_residual=0.0, true_relative_residual=0.0,
            relative_normwise_2=0.0, true_residual_role="certification",
            certification_check_source="zero_rhs_identity", certification_charged=false,
            certification_checks=0, correction_restarts=0, krylov_block=0, total_fma=0.0,
            last_committed_iteration=0, buffered_rows=0, solver_mode=string(solver_mode),
            status=string(CONVERGED), final=true)
    end
    metadata = _solver_policy_metadata(policy; status=CONVERGED,
        convergence_tolerance=convergence_tolerance,
        correction_restart_normwise_tolerance=correction_restart_normwise_tolerance)
    metadata["initial_solution"] = "exact_zero_rhs"
    return SolverResult(CONVERGED, 0, 0.0, 0.0, 0.0,
        0, Dict{String,Any}[], 0, Dict{String,Any}[], 0, 0, 0,
        Int[], Int[], 0, nothing, nothing, nothing, records, metadata)
end

# Initial preconditioner failure still has a well-defined x0 and durable row.
function _initial_breakdown_result(data, b, seed, cum_prec, cum_vector, policy,
        conv_out, raw_out, progress, convergence_tolerance, restart_tolerance)
    total = Float64(cum_prec + cum_vector)
    status = policy.work_limit > 0 && total > policy.work_limit ? WORK_LIMIT : BREAKDOWN
    norm_b = threaded_norm(b)
    for col in eachindex(data.columns)
        data[1,col] = NaN
    end
    data[1,1] = 0.0
    data[1,2] = 1.0
    data[1,3] = 1.0
    data[1,4] = all(iszero, seed) ? NaN : 1.0
    data[1,5] = 0.0
    data[1,6] = Float64(cum_prec)
    data[1,7] = cum_vector
    for name in ("norm_r", "norm_r_true", "energy", "krylov_block", "diag_abs_flag")
        col = findfirst(==(name), data.columns)
        col === nothing || (data[1,col] = startswith(name, "norm_r") ? norm_b : 0.0)
    end
    run_phase(progress.callback, "result_write") do
        used = _finish_diagnostic_data!(data, 1)
        open(replace(raw_out, r"\.[^/]*$" => "_columns.json"), "w") do io
            JSON3.write(io, data.columns)
        end
        _write_convergence(conv_out * "_relative_residual.csv", used, PCG_COL_RELATIVE_RESIDUAL)
        _write_convergence(conv_out * "_relative_normwise_2.csv", used, PCG_COL_RELATIVE_NORMWISE_2)
        _write_certification_convergence(conv_out * "_certification.csv", used,
            PCG_COL_NORM_R, norm_b, Dict{String,Any}[], Int[];
            solver_policy=:scalar, n=length(b), solver_mode=policy.mode,
            continuation_work_factor=policy.continuation_work_factor,
            work_limit=Int64(policy.work_limit), correction_restart_normwise_tolerance=restart_tolerance)
    end
    result = SolverResult(status, 0, 0.0, total, 1.0)
    merge!(result.solver_policy, _solver_policy_metadata(policy; status,
        convergence_tolerance, correction_restart_normwise_tolerance=restart_tolerance))
    return result
end
