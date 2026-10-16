# Admission is per configuration/solver, not per hierarchy or optioned batch.
# Producers must explicitly identify their cost semantics; an untyped estimate
# is not sufficient evidence to skip a solve.
function _application_cost_guard_result(prec_meta, work_limit;
        solver_mode=:certified, continuation_work_factor=2.0,
        convergence_tolerance=minimum_tolerance,
        progress_callback=NOOP_PHASE_CALLBACK)
    prec_meta === nothing && return nothing
    isfinite(work_limit) && work_limit > 0 || return nothing
    kind = get(prec_meta, "apply_cost_kind", nothing)
    kind in ("fixed", "fixed_inner_iterations") || return nothing
    cost = get(prec_meta, "apply_fma", nothing)
    cost isa Real && !(cost isa Bool) && isfinite(cost) && cost >= 0 || return nothing
    cost > work_limit || return nothing

    guard = Dict{String,Any}(
        "schema_version" => 1,
        "reason" => "single_application_exceeds_full_work_limit",
        "apply_fma" => cost,
        "apply_cost_kind" => kind,
        "work_limit" => work_limit,
        "threshold_factor" => 1.0,
        "application_executed" => false,
        "exact_zero_shortcuts_not_modeled" => kind == "fixed_inner_iterations",
    )
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    return run_phase(progress_callback, "solver_admission";
                     status=string(SKIPPED_COST_GUARD), application_cost_guard=guard) do
        result = SolverResult(SKIPPED_COST_GUARD, 0, 0.0, 0.0, NaN)
        merge!(result.solver_policy, _solver_policy_metadata(policy;
            status=SKIPPED_COST_GUARD, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance_default))
        result.solver_policy["application_cost_guard"] = guard
        result
    end
end

function _application_cost_metadata!(meta, result)
    result.status == SKIPPED_COST_GUARD || return meta
    meta["phase"] = "solver_admission"
    meta["solver_executed"] = false
    meta["diagnostic_artifacts_current"] = false
    meta["application_cost_guard"] = result.solver_policy["application_cost_guard"]
    return meta
end
