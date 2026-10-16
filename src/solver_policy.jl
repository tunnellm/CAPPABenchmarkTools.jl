# Policy bookkeeping never examines diagnostic residuals. Native crossings
# arm the survey cap; certification remains the only residual-based stop.
mutable struct SolverPolicyState
    mode::Symbol
    continuation_work_factor::Float64
    work_limit::Float64
    native_candidate_iter::Union{Nothing,Int}
    native_candidate_total_fma::Union{Nothing,Float64}
    continuation_work_limit::Union{Nothing,Float64}
    crossing_report_pending::Bool
    started_at::Float64
    last_report_at::Float64
    last_report_fma::Float64
end

function SolverPolicyState(mode::Symbol, factor::Real, work_limit::Real)
    mode in (:certified, :bounded_continuation) ||
        throw(ArgumentError("solver_mode must be :certified or :bounded_continuation"))
    isfinite(factor) && factor > 1 ||
        throw(ArgumentError("continuation_work_factor must be finite and greater than one"))
    now = Float64(time_ns()) / 1e9
    return SolverPolicyState(mode, Float64(factor), Float64(work_limit),
        nothing, nothing, nothing, false, now, now, 0.0)
end

function _observe_native_candidate!(policy::SolverPolicyState, native_candidate, iter, total_fma)
    if native_candidate && policy.native_candidate_iter === nothing
        policy.native_candidate_iter = iter
        policy.native_candidate_total_fma = total_fma
        if policy.mode == :bounded_continuation
            policy.continuation_work_limit = policy.continuation_work_factor * total_fma
        end
        policy.crossing_report_pending = true
    end
    return nothing
end

function _policy_stop_status(policy::SolverPolicyState, total_fma, certified)
    # Attribute an overshoot to the smaller budget. Passing exactly at a
    # budget is allowed, matching the existing overall-budget convention.
    cap = policy.continuation_work_limit
    if policy.work_limit > 0 && (cap === nothing || policy.work_limit <= cap) &&
            total_fma > policy.work_limit
        return WORK_LIMIT
    end
    cap !== nothing && total_fma > cap && return CONTINUATION_LIMIT
    policy.work_limit > 0 && total_fma > policy.work_limit && return WORK_LIMIT
    certified && return CONVERGED
    cap !== nothing && total_fma >= cap && return CONTINUATION_LIMIT
    return nothing
end

function _solver_policy_metadata(policy::SolverPolicyState; status=nothing,
                                convergence_tolerance=nothing, correction_restart_normwise_tolerance=nothing)
    return Dict{String,Any}(
        "schema_version" => 1,
        "convergence_tolerance" => convergence_tolerance,
        "correction_restart_normwise_tolerance" => correction_restart_normwise_tolerance,
        "effective_correction_restart_gate" => convergence_tolerance === nothing ||
            correction_restart_normwise_tolerance === nothing ? nothing :
            min(convergence_tolerance,correction_restart_normwise_tolerance),
        "max_correction_restarts" => policy.mode == :certified ? 1 : 0,
        "solver_mode" => string(policy.mode),
        "continuation_work_factor" => policy.mode == :bounded_continuation ? policy.continuation_work_factor : nothing,
        "native_candidate_total_fma" => policy.native_candidate_total_fma,
        "continuation_work_limit" => policy.continuation_work_limit,
        "work_limit" => policy.work_limit > 0 ? policy.work_limit : nothing,
        "stop_reason" => status === nothing ? nothing : string(status),
    )
end

# Default runs retain their existing cache identities, especially Control.
function _solver_policy_config(kwargs::Dict{String,Any}, mode::Symbol, factor::Real)
    SolverPolicyState(mode, factor, -1) # validate before creating output paths
    mode == :certified && return kwargs
    haskey(kwargs, "__solver_policy") && throw(ArgumentError("reserved config key __solver_policy"))
    return merge(kwargs, Dict{String,Any}("__solver_policy" =>
        (mode=string(mode), continuation_work_factor=Float64(factor))))
end
