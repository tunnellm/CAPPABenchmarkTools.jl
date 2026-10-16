module TestWrapper

using SparseArrays
using LinearAlgebra
using Statistics
using JSON3
using SHA

using ..Utilities:
    ComponentDescriptor, describe_components, _execution_metadata!,
    prepare_system,
    Logger, open_logger, log!, close_logger!,
    ResourceMonitor, start_monitor, stop_monitor!, write_resources,
    NOOP_PHASE_CALLBACK, run_phase, run_scoped_phase, phase_callback_with_context
using ..PreconditionerTypes: Preconditioner, OptionedPreconditioner
using ..Preconditioners: _preconditioner_identity_kwargs
using ..Solvers: _diagnostic_storage_metadata!, pcg, minres, pcr, lanczos_pcg, fpcg1, fpcg2, SolverResult, SolverStatus,
    CONVERGED, MAX_ITERS, WORK_LIMIT, NAN_DETECTED, BREAKDOWN, STAGNATED, KILLED,
    CONTINUATION_LIMIT, SKIPPED_COST_GUARD, _solver_policy_config,
    _application_cost_guard_result, _application_cost_metadata!
using ..Solvers: _diagnostic_attempt_complete

export run_matrix

const MIN_COMPONENT_DIMENSION = 50

# ─── Hashing ─────────────────────────────────────────────────────────

"""
    _config_hash(kwargs::Dict{String,Any}) -> String

Compute an 8-character hex hash from the sorted preconditioner kwargs.
"""
function _config_hash(kwargs::Dict{String,Any})
    # Deterministic JSON serialization: sorted keys
    sorted_keys = sort(collect(keys(kwargs)))
    parts = [string(k, "=", repr(kwargs[k])) for k in sorted_keys]
    joined = join(parts, ",")
    return bytes2hex(sha256(joined))[1:8]
end

# ─── Control cost lookup ─────────────────────────────────────────────

"""
    _load_control_cost(results_location, name, solver, ci) -> Union{Float64, Nothing}

Look up Control's total_fma for a specific component and solver.
Control is ordering-independent, so results are always stored under "natural".
For Control, total_fma = base + vector (prec apply cost is 0).
Returns nothing if Control results don't exist or didn't converge.
"""
function _control_metadata_path(results_location::String, name::String, solver::Symbol, ci::Int)
    control_hash = _config_hash(Dict{String,Any}())
    return joinpath(results_location, name, "natural", "Control", control_hash,
                         string(solver), "c$ci", "metadata.json")
end

function _load_control_cost(results_location::String, name::String,
                            solver::Symbol, ci::Int)
    meta_path = _control_metadata_path(results_location,name,solver,ci)
    if !isfile(meta_path)
        return nothing
    end
    meta = _load_metadata(meta_path)
    if meta === nothing
        return nothing
    end
    status = get(meta, "status", "")
    if status != "CONVERGED"
        return nothing
    end
    _diagnostic_attempt_complete(meta, dirname(meta_path)) || return nothing
    total = get(meta, "total_fma", nothing)
    return total === nothing ? nothing : Float64(total)
end

function _require_survey_baselines(results_location, name, component, solvers, factor)
    isfinite(factor) && factor > 0 ||
        throw(ArgumentError("survey work_limit_factor must be finite and positive"))
    for solver in solvers
        cost = _load_control_cost(results_location, name, solver, component)
        cost !== nothing && isfinite(cost) && cost > 0 ||
            error("bounded continuation requires an existing converged Control baseline for $name c$component $solver")
        1 <= factor * cost <= typemax(Int64) ||
            error("survey Control work budget is outside the supported range")
    end
    return nothing
end

# ─── run_matrix ──────────────────────────────────────────────────────

"""
    run_matrix(name, preconditioner; kwargs...)

Run iterative solvers on all components of a matrix with a given preconditioner.

# Arguments
- `name::String`: Matrix name (looked up via `describe_components`).
- `preconditioner::Function`: Preconditioner constructor, called as
  `preconditioner(A; preconditioner_kwargs...)`. Must return `(Preconditioner, Dict)`.

# Keyword Arguments
- `preconditioner_kwargs::Dict{String,Any}`: Arguments passed to the preconditioner
  constructor. Also hashed to produce the configuration directory and written to `config.json`.
- `solvers::Vector{Symbol}`: Solvers to run. Default `[:pcg, :minres]`.
- `reordering::String`: Matrix reordering. Default `"natural"`.
- `wants_sdd::Bool`: If true, preserve SDD structure when possible. Default `false`.
- `force::Bool`: Re-run even if previous results exist. Default `false`.
- `work_limit_factor::Float64`: Stop if cumulative work exceeds this multiple of
  Control's cost on the same component. Default `5.0`. Ignored if Control results
  don't exist.
- `results_location::String`: Base output directory. Default `"./results/"`.
- `base_dir::String`: Base directory for matrix data. Default `"./matrices/"`.
# Output Layout
```
{results_location}/{name}/{reordering}/{preconditioner_name}/{hash}/
    config.json
    {solver}/
        c{i}/
            raw_data/manifest.json
            raw_data/attempt_*/000001.npy.zstd
            convergence_data_*.csv
            metadata.json
            resources_generation.csv
            resources_solver.csv
            log.jsonl
```
"""
function run_matrix(
    name::String,
    preconditioner::Function;
    preconditioner_kwargs::Dict{String,Any} = Dict{String,Any}(),
    solvers::Vector{Symbol} = [:pcg, :minres],
    reordering::String = "natural",
    wants_sdd::Bool = false,
    force::Bool = false,
    work_limit_factor::Float64 = 5.0,
    results_location::String = "./results/",
    base_dir::String = "./matrices/",
    phase_callback = NOOP_PHASE_CALLBACK,
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    prec_name = string(preconditioner)
    solver_mode == :bounded_continuation && prec_name == "Control" &&
        throw(ArgumentError("bounded continuation uses existing Control baselines; do not rerun Control"))
    preconditioner_kwargs = _preconditioner_identity_kwargs(preconditioner, preconditioner_kwargs)
    policy_config = _solver_policy_config(preconditioner_kwargs, solver_mode, continuation_work_factor)
    hash = _config_hash(policy_config)

    # Convert string keys to symbols for splatting
    sym_kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in preconditioner_kwargs)

    # Write config.json
    config_dir = joinpath(results_location, name, reordering, prec_name, hash)
    mkpath(config_dir)
    config_path = joinpath(config_dir, "config.json")
    if !isfile(config_path)
        open(config_path, "w") do f
            JSON3.write(f, policy_config)
        end
    end

    # Load all components
    prepare_progress_callback =
        phase_callback_with_context(phase_callback; matrix = name)
    descs = run_scoped_phase(phase_callback, "prepare"; matrix = name, ordering=reordering) do child_progress
        describe_components(
            name;
            base_dir = base_dir,
            reordering = reordering,
            delete_original = true,
            prepare_progress_callback = child_progress,
        )
    end

    for desc in descs
        ci = desc.component_index
        if desc.dimension <= MIN_COMPONENT_DIMENSION
            println("Skipping $name c$ci (n=$(desc.dimension) ≤ $MIN_COMPONENT_DIMENSION)")
            continue
        end
        n_maxiters = 100 * desc.dimension

        # Check which solvers need to run for this component
        solvers_to_run = Symbol[]
        for s in solvers
            solver_dir = joinpath(config_dir, string(s), "c$ci")
            if !force && _should_skip(solver_dir)
                continue
            end
            push!(solvers_to_run, s)
        end

        if isempty(solvers_to_run)
            println("All solvers skipped for $prec_name on $name c$ci")
            continue
        end

        if solver_mode == :bounded_continuation
            needed = solvers_to_run
            _require_survey_baselines(results_location, name, desc.component_index, needed, work_limit_factor)
        end

        # Resolve the norm once in natural ordering before any permutation.
        spectral_info = Ref{Any}()
        # Prepare the system
        A, seed, b = run_phase(phase_callback, "system_prepare"; matrix=name, component=desc.component_index, ordering=reordering,
            result_directories=[abspath(joinpath(config_dir,string(s),"c$ci")) for s in solvers_to_run]) do
            prepare_system(desc; wants_sdd=wants_sdd, spectral_info, progress_callback=phase_callback)
        end
        n = size(A, 1)
        spectral_norm = Float64(spectral_info[]["estimate"])

        # Build preconditioner (with resource monitoring)
        local prec_result
        local prec_meta::Dict{String,Any}
        local gen_samples
        gen_mon = start_monitor(paths=[joinpath(config_dir,string(s),"c$ci","resources_generation.csv")
            for s in solvers_to_run], context=Dict("phase"=>"preconditioner_generation"))
        try
            prec_result = run_phase(
                phase_callback, "preconditioner_generation";
                matrix = name, component = ci, preconditioner = prec_name,
                result_directories = [abspath(joinpath(config_dir, string(s), "c$ci")) for s in solvers_to_run],
            ) do
                preconditioner(A; sym_kwargs...)
            end
            gen_samples = stop_monitor!(gen_mon)
        catch e
            gen_samples = stop_monitor!(gen_mon)
            is_interrupt = e isa InterruptException
            errmsg = sprint(showerror, e, catch_backtrace())
            if is_interrupt
                println("Killed during preconditioner generation for $prec_name on $name c$ci")
            else
                println("Error building preconditioner $prec_name for $name c$ci: $e")
            end
            for s in solvers_to_run
                solver_dir = joinpath(config_dir, string(s), "c$ci")
                mkpath(solver_dir)
                write_resources(joinpath(solver_dir, "resources_generation.csv"), gen_samples)
                logger = open_logger(joinpath(solver_dir, "log.jsonl"))
                if is_interrupt
                    log!(logger, :error, "preconditioner_killed", "Killed during generation";
                         preconditioner=prec_name, matrix=name, component=ci)
                    _write_killed_metadata(solver_dir, "preconditioner_generation")
                else
                    log!(logger, :error, "preconditioner_error", errmsg;
                         preconditioner=prec_name, matrix=name, component=ci)
                    _write_error_metadata(solver_dir, errmsg)
                end
                close_logger!(logger)
            end
            continue
        end

        # Unpack preconditioner result
        prec, prec_meta = prec_result

        # Run each solver
        for s in solvers_to_run
            solver_dir = joinpath(config_dir, string(s), "c$ci")
            mkpath(solver_dir)

            raw_out = joinpath(solver_dir, "raw_data.npy.zstd")
            conv_out = joinpath(solver_dir, "convergence_data")

            # Write generation resources
            write_resources(joinpath(solver_dir, "resources_generation.csv"), gen_samples)

            # Compute work limit from Control cost
            work_limit = Int64(-1)
            control_cost = _load_control_cost(results_location, name, s, ci)
            if control_cost !== nothing
                work_limit = round(Int64, work_limit_factor * control_cost)
            end

            logger = open_logger(joinpath(solver_dir, "log.jsonl"))
            log!(logger, :info, "solver_start", "Running $s with $prec_name on $name c$ci";
                 solver=string(s), preconditioner=prec_name, matrix=name, component=ci,
                 n=n, nnz=nnz(A), spectral_norm=spectral_norm,
                 apply_fma=get(prec_meta, "apply_fma", "unknown"),
                 generation_fma=prec.generation_fma,
                 generation_graph=prec.generation_graph)

            solver_mon = start_monitor(paths=[joinpath(solver_dir,"resources_solver.csv")],
                context=Dict("phase"=>"solver", "work_limit_provenance"=>Dict(
                    "work_limit"=>work_limit, "factor"=>work_limit_factor,
                    "control_total_fma"=>control_cost,
                    "control_metadata"=>abspath(_control_metadata_path(results_location,name,s,ci)))))
            try
                result = run_scoped_phase(
                    phase_callback, "solver";
                    matrix = name, component = ci, preconditioner = prec_name,
                    solver = string(s), config_hash = hash,
                    result_directories = [abspath(solver_dir)],
                ) do child_progress
                    _run_solver(s, A, b, seed, spectral_norm, prec, n_maxiters,
                                raw_out, conv_out; work_limit=work_limit, prec_meta=prec_meta, progress_callback=child_progress,
                                    solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
                end
                solver_samples = stop_monitor!(solver_mon)
                run_phase(phase_callback, "result_metadata_write"; matrix=name, solver=string(s), result_directories=[abspath(solver_dir)]) do
                    write_resources(joinpath(solver_dir, "resources_solver.csv"), solver_samples)
                    _write_metadata(solver_dir, result, n, Int64(nnz(A)), prec_meta; spectral_norm_info=spectral_info[])
                end
                msg = _status_message(name, prec_name, s, result)
                log!(logger, :info, "solver_done", msg;
                     status=string(result.status), iterations=result.iterations)
                println(msg)
            catch e
                solver_samples = stop_monitor!(solver_mon)
                write_resources(joinpath(solver_dir, "resources_solver.csv"), solver_samples)
                is_interrupt = e isa InterruptException
                errmsg = sprint(showerror, e, catch_backtrace())
                if is_interrupt
                    println("Killed during $s for $prec_name on $name c$ci")
                    log!(logger, :error, "solver_killed", "Killed during $s";
                         solver=string(s), preconditioner=prec_name, matrix=name, component=ci)
                    _write_killed_metadata(solver_dir, "solver")
                else
                    println("Error in $s for $prec_name on $name c$ci: $e")
                    log!(logger, :error, "solver_error", errmsg;
                         solver=string(s), preconditioner=prec_name, matrix=name, component=ci)
                    _write_error_metadata(solver_dir, errmsg)
                end
            end
            close_logger!(logger)
        end
    end

    return nothing
end

# ─── run_matrix for OptionedPreconditioner ───────────────────────────

"""
    run_matrix(name, hierarchy_builder, configs; setup_kwargs=..., ...)

Run iterative solvers on all components of a matrix using an `OptionedPreconditioner`.
The hierarchy is built once per component; each config in `configs` reconfigures
apply-time parameters (cycle, smoothers, etc.) without rebuilding.

# Arguments
- `name::String`: Matrix name (looked up via `describe_components`).
- `hierarchy_builder::Function`: Called as `hierarchy_builder(A, OptionedPreconditioner; setup_kwargs...)`
  and must return an `OptionedPreconditioner`.
- `configs::Vector{<:NamedTuple}`: Each entry has:
  - `kwargs::Dict{String,Any}` — apply-time arguments passed to `op(; ...)`.
  - `solvers::Vector{Symbol}` — solvers to run for this config.

# Keyword Arguments
- `setup_kwargs::Dict{String,Any}`: Setup-time arguments passed to `hierarchy_builder`.
  Included in every config's hash.
- `reordering::String`: Matrix reordering. Default `"natural"`.
- `wants_sdd::Bool`: If true, preserve SDD structure when possible. Default `false`.
- `force::Bool`: Re-run even if previous results exist. Default `false`.
- `work_limit_factor::Float64`: Stop if cumulative work exceeds this multiple of
  Control's cost. Default `5.0`.
- `results_location::String`: Base output directory. Default `"./results/"`.
- `base_dir::String`: Base directory for matrix data. Default `"./matrices/"`.

# Output Layout
Same as the single-config `run_matrix`, with one directory per config hash:
```
{results_location}/{name}/{reordering}/{prec_name}/{hash}/{solver}/c{i}/...
```
"""
function run_matrix(
    name::String,
    hierarchy_builder::Function,
    configs::Vector{<:NamedTuple};
    setup_kwargs::Dict{String,Any} = Dict{String,Any}(),
    reordering::String = "natural",
    wants_sdd::Bool = false,
    force::Bool = false,
    work_limit_factor::Float64 = 5.0,
    results_location::String = "./results/",
    base_dir::String = "./matrices/",
    phase_callback = NOOP_PHASE_CALLBACK,
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    prec_name = string(hierarchy_builder)
    sym_setup_kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in setup_kwargs)

    # Pre-compute per-config: full kwargs (for hash), apply kwargs (for configure), hash, config_dir
    config_info = map(configs) do cfg
        full_kwargs = _solver_policy_config(merge(setup_kwargs, cfg.kwargs), solver_mode, continuation_work_factor)
        hash = _config_hash(full_kwargs)
        sym_apply_kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in cfg.kwargs)
        config_dir = joinpath(results_location, name, reordering, prec_name, hash)
        (full_kwargs=full_kwargs, sym_apply_kwargs=sym_apply_kwargs, hash=hash,
         config_dir=config_dir, solvers=cfg.solvers)
    end

    # Write config.json for each config
    for ci in config_info
        mkpath(ci.config_dir)
        config_path = joinpath(ci.config_dir, "config.json")
        if !isfile(config_path)
            open(config_path, "w") do f
                JSON3.write(f, ci.full_kwargs)
            end
        end
    end

    # Load all components
    prepare_progress_callback =
        phase_callback_with_context(phase_callback; matrix = name)
    descs = run_scoped_phase(phase_callback, "prepare"; matrix = name, ordering=reordering) do child_progress
        describe_components(
            name;
            base_dir = base_dir,
            reordering = reordering,
            delete_original = true,
            prepare_progress_callback = child_progress,
        )
    end

    for desc in descs
        comp_idx = desc.component_index
        if desc.dimension <= MIN_COMPONENT_DIMENSION
            println("Skipping $name c$comp_idx (n=$(desc.dimension) ≤ $MIN_COMPONENT_DIMENSION)")
            continue
        end
        n_maxiters = 100 * desc.dimension

        # Determine which configs have work to do for this component
        configs_to_run = filter(config_info) do ci
            any(ci.solvers) do s
                solver_dir = joinpath(ci.config_dir, string(s), "c$comp_idx")
                force || !_should_skip(solver_dir)
            end
        end

        if isempty(configs_to_run)
            println("All configs skipped for $prec_name on $name c$comp_idx")
            continue
        end

        if solver_mode == :bounded_continuation
            needed = unique(vcat([c.solvers for c in configs_to_run]...))
            _require_survey_baselines(results_location, name, desc.component_index, needed, work_limit_factor)
        end

        spectral_info = Ref{Any}()
        # Prepare the system
        A, seed, b = run_phase(phase_callback, "system_prepare"; matrix=name, component=desc.component_index, ordering=reordering,
            result_directories=[abspath(joinpath(c.config_dir,string(s),"c$comp_idx"))
                for c in configs_to_run for s in c.solvers
                if force || !_should_skip(joinpath(c.config_dir,string(s),"c$comp_idx"))]) do
            prepare_system(desc; wants_sdd=wants_sdd, spectral_info, progress_callback=phase_callback)
        end
        n = size(A, 1)
        spectral_norm = Float64(spectral_info[]["estimate"])

        # Build hierarchy once for this component (with resource monitoring)
        local op::OptionedPreconditioner
        local gen_samples
        gen_mon = start_monitor(paths=[joinpath(c.config_dir,string(s),"c$comp_idx","resources_generation.csv")
            for c in configs_to_run for s in c.solvers
            if force || !_should_skip(joinpath(c.config_dir,string(s),"c$comp_idx"))],
            context=Dict("phase"=>"preconditioner_generation"))
        try
            op = run_phase(
                phase_callback, "preconditioner_generation";
                matrix = name, component = comp_idx, preconditioner = prec_name,
                result_directories = [abspath(joinpath(c.config_dir, string(s), "c$comp_idx"))
                    for c in configs_to_run for s in c.solvers
                    if force || !_should_skip(joinpath(c.config_dir, string(s), "c$comp_idx"))],
            ) do
                hierarchy_builder(A, OptionedPreconditioner; sym_setup_kwargs...)
            end
            gen_samples = stop_monitor!(gen_mon)
        catch e
            gen_samples = stop_monitor!(gen_mon)
            is_interrupt = e isa InterruptException
            errmsg = sprint(showerror, e, catch_backtrace())
            if is_interrupt
                println("Killed during hierarchy build for $prec_name on $name c$comp_idx")
            else
                println("Error building hierarchy $prec_name for $name c$comp_idx: $e")
            end
            for ci in configs_to_run
                for s in ci.solvers
                    solver_dir = joinpath(ci.config_dir, string(s), "c$comp_idx")
                    mkpath(solver_dir)
                    write_resources(joinpath(solver_dir, "resources_generation.csv"), gen_samples)
                    logger = open_logger(joinpath(solver_dir, "log.jsonl"))
                    if is_interrupt
                        log!(logger, :error, "preconditioner_killed", "Killed during hierarchy build";
                             preconditioner=prec_name, matrix=name, component=comp_idx)
                        _write_killed_metadata(solver_dir, "preconditioner_generation")
                    else
                        log!(logger, :error, "preconditioner_error", errmsg;
                             preconditioner=prec_name, matrix=name, component=comp_idx)
                        _write_error_metadata(solver_dir, errmsg)
                    end
                    close_logger!(logger)
                end
            end
            continue
        end

        # Run each config against this component's hierarchy
        for ci in configs_to_run
            # Configure apply-time parameters
            local prec::Preconditioner
            local prec_meta::Dict{String,Any}
            try
                prec, prec_meta = run_phase(
                    phase_callback, "preconditioner_configure";
                    matrix = name, component = comp_idx, preconditioner = prec_name,
                    config_hash = ci.hash,
                    result_directories = [abspath(joinpath(ci.config_dir, string(s), "c$comp_idx"))
                        for s in ci.solvers if force || !_should_skip(joinpath(ci.config_dir, string(s), "c$comp_idx"))],
                ) do
                    op(; ci.sym_apply_kwargs...)
                end
            catch e
                is_interrupt = e isa InterruptException
                errmsg = sprint(showerror, e, catch_backtrace())
                if is_interrupt
                    println("Killed during configure for $prec_name ($(ci.hash)) on $name c$comp_idx")
                else
                    println("Error configuring $prec_name ($(ci.hash)) for $name c$comp_idx: $e")
                end
                for s in ci.solvers
                    solver_dir = joinpath(ci.config_dir, string(s), "c$comp_idx")
                    mkpath(solver_dir)
                    write_resources(joinpath(solver_dir, "resources_generation.csv"), gen_samples)
                    logger = open_logger(joinpath(solver_dir, "log.jsonl"))
                    if is_interrupt
                        log!(logger, :error, "configure_killed", "Killed during configure";
                             preconditioner=prec_name, matrix=name, component=comp_idx)
                        _write_killed_metadata(solver_dir, "preconditioner_configure")
                    else
                        log!(logger, :error, "configure_error", errmsg;
                             preconditioner=prec_name, matrix=name, component=comp_idx)
                        _write_error_metadata(solver_dir, errmsg)
                    end
                    close_logger!(logger)
                end
                continue
            end

            # Run solvers for this config
            for s in ci.solvers
                solver_dir = joinpath(ci.config_dir, string(s), "c$comp_idx")
                if !force && _should_skip(solver_dir)
                    continue
                end
                mkpath(solver_dir)

                raw_out = joinpath(solver_dir, "raw_data.npy.zstd")
                conv_out = joinpath(solver_dir, "convergence_data")

                write_resources(joinpath(solver_dir, "resources_generation.csv"), gen_samples)

                work_limit = Int64(-1)
                control_cost = _load_control_cost(results_location, name, s, comp_idx)
                if control_cost !== nothing
                    work_limit = round(Int64, work_limit_factor * control_cost)
                end

                logger = open_logger(joinpath(solver_dir, "log.jsonl"))
                log!(logger, :info, "solver_start", "Running $s with $prec_name ($(ci.hash)) on $name c$comp_idx";
                     solver=string(s), preconditioner=prec_name, matrix=name, component=comp_idx,
                     n=n, nnz=nnz(A), spectral_norm=spectral_norm,
                     apply_fma=get(prec_meta, "apply_fma", "unknown"),
                     generation_fma=prec.generation_fma,
                     generation_graph=prec.generation_graph)

                solver_mon = start_monitor(paths=[joinpath(solver_dir,"resources_solver.csv")],
                    context=Dict("phase"=>"solver", "work_limit_provenance"=>Dict(
                        "work_limit"=>work_limit, "factor"=>work_limit_factor,
                        "control_total_fma"=>control_cost,
                        "control_metadata"=>abspath(_control_metadata_path(results_location,name,s,comp_idx)))))
                try
                    result = run_scoped_phase(
                        phase_callback, "solver";
                        matrix = name, component = comp_idx, preconditioner = prec_name,
                        solver = string(s), config_hash = ci.hash,
                        result_directories = [abspath(solver_dir)],
                    ) do child_progress
                        _run_solver(s, A, b, seed, spectral_norm, prec, n_maxiters,
                                    raw_out, conv_out; work_limit=work_limit, prec_meta=prec_meta, progress_callback=child_progress,
                                    solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
                    end
                    solver_samples = stop_monitor!(solver_mon)
                    run_phase(phase_callback, "result_metadata_write"; matrix=name, solver=string(s), result_directories=[abspath(solver_dir)]) do
                        write_resources(joinpath(solver_dir, "resources_solver.csv"), solver_samples)
                        _write_metadata(solver_dir, result, n, Int64(nnz(A)), prec_meta; spectral_norm_info=spectral_info[])
                    end
                    msg = _status_message(name, prec_name, s, result)
                    log!(logger, :info, "solver_done", msg;
                         status=string(result.status), iterations=result.iterations)
                    println(msg)
                catch e
                    solver_samples = stop_monitor!(solver_mon)
                    write_resources(joinpath(solver_dir, "resources_solver.csv"), solver_samples)
                    is_interrupt = e isa InterruptException
                    errmsg = sprint(showerror, e, catch_backtrace())
                    if is_interrupt
                        println("Killed during $s for $prec_name ($(ci.hash)) on $name c$comp_idx")
                        log!(logger, :error, "solver_killed", "Killed during $s";
                             solver=string(s), preconditioner=prec_name, matrix=name, component=comp_idx)
                        _write_killed_metadata(solver_dir, "solver")
                    else
                        println("Error in $s for $prec_name ($(ci.hash)) on $name c$comp_idx: $e")
                        log!(logger, :error, "solver_error", errmsg;
                             solver=string(s), preconditioner=prec_name, matrix=name, component=comp_idx)
                        _write_error_metadata(solver_dir, errmsg)
                    end
                end
                close_logger!(logger)
            end
        end
    end

    return nothing
end

"""
    _should_skip(solver_dir) -> Bool

Check if a previous run exists and should be skipped.
KILLED runs are re-tried; all other terminal statuses are skipped.
"""
function _should_skip(solver_dir::String)
    meta_path = joinpath(solver_dir, "metadata.json")
    if !isfile(meta_path)
        return false
    end
    meta = _load_metadata(meta_path)
    if meta === nothing
        return false
    end
    status = get(meta, "status", "")
    if status == "KILLED"
        return false
    end
    get(meta, "scheduler_failure_stub", false) && return false
    if status in ("CONVERGED", "MAX_ITERS", "WORK_LIMIT", "NAN_DETECTED", "BREAKDOWN", "STAGNATED", "CONTINUATION_LIMIT", "SKIPPED_COST_GUARD", "ERROR")
        return status in ("ERROR", "SKIPPED_COST_GUARD") || _diagnostic_attempt_complete(meta, solver_dir)
    end
    return false
end

"""
    _load_metadata(path) -> Dict or nothing

Load a metadata.json file, returning nothing on parse failure.
"""
function _load_metadata(path::String)
    try
        return open(path, "r") do f
            JSON3.read(f, Dict{String, Any})
        end
    catch
        return nothing
    end
end


"""
    _write_metadata(solver_dir, result, n, nnz_A, prec_meta)

Write structured metadata.json for a successful solver run.
Merges preconditioner metadata into the output under a "preconditioner" key.
"""
_json_safe(x::Float64) = isfinite(x) ? x : nothing
_json_safe(x) = x
_json_safe_dict(d::Dict) = Dict{String,Any}(k => _json_safe_value(v) for (k, v) in d)
_json_safe_value(x::Float64) = _json_safe(x)
_json_safe_value(d::Dict) = _json_safe_dict(d)
_json_safe_value(x) = x

function _write_metadata(solver_dir::String, result::SolverResult, n::Int, nnz_A::Int64,
                         prec_meta::Dict{String,Any}; spectral_norm_info=nothing)
    meta = Dict{String, Any}(
        "timestamp" => time(),
        "status" => string(result.status),
        "iterations" => result.iterations,
        "wall_time_seconds" => result.wall_time_seconds,
        "total_fma" => _json_safe(result.total_fma),
        "final_relative_residual" => _json_safe(result.final_relative_residual),
        "residual_replacements" => result.residual_replacements,
        "residual_replacement_records" => result.residual_replacement_records,
        "correction_restarts" => result.correction_restarts,
        "correction_restart_records" => result.correction_restart_records,
        "true_residual_diagnostic_stream_available" => result.physical_true_residual_evals > 0,
        "physical_true_residual_evals" => result.physical_true_residual_evals,
        "diagnostic_true_residual_evals" => result.diagnostic_true_residual_evals,
        "certification_true_residual_evals" => result.certification_true_residual_evals,
        "certification_check_iters" => result.certification_check_iters,
        "certification_spmv_charge" => result.certification_true_residual_evals,
        "diagnostic_spmv_charge" => result.diagnostic_true_residual_evals,
        "true_residual_used_for_decision_iters" => result.true_residual_used_for_decision_iters,
        "false_native_crossings" => result.false_native_crossings,
        "native_candidate_iter" => result.native_candidate_iter,
        "native_candidate_true_relres" => _json_safe(result.native_candidate_true_relres),
        "native_candidate_normwise" => _json_safe(result.native_candidate_normwise),
        "certification_records" => result.certification_records,
        "solver_policy" => result.solver_policy,
        "n" => n,
        "nnz" => nnz_A,
        "preconditioner" => _json_safe_dict(prec_meta),
    )
    _application_cost_metadata!(meta, result)
    spectral_norm_info === nothing || (meta["spectral_norm"] = spectral_norm_info)
    result.status != SKIPPED_COST_GUARD && _diagnostic_storage_metadata!(meta, solver_dir)
    _execution_metadata!(meta, solver_dir)
    open(joinpath(solver_dir, "metadata.json"), "w") do f
        JSON3.write(f, meta)
    end
end

"""
    _write_error_metadata(solver_dir, errmsg)

Write metadata.json for a failed run (preconditioner build or solver error).
"""
function _write_error_metadata(solver_dir::String, errmsg::String)
    meta = Dict{String, Any}(
        "timestamp" => time(),
        "status" => "ERROR",
        "error" => errmsg,
    )
    _execution_metadata!(meta, solver_dir)
    open(joinpath(solver_dir, "metadata.json"), "w") do f
        JSON3.write(f, meta)
    end
end

"""
    _write_killed_metadata(solver_dir, phase)

Write metadata.json for a run interrupted by a kill signal.
`phase` names the interrupted prepare, generation, configure, restore, or solver
phase.
"""
function _write_killed_metadata(solver_dir::String, phase::String)
    meta = Dict{String, Any}(
        "timestamp" => time(),
        "status" => "KILLED",
        "phase" => phase,
    )
    _execution_metadata!(meta, solver_dir)
    open(joinpath(solver_dir, "metadata.json"), "w") do f
        JSON3.write(f, meta)
    end
end


function _run_solver(
    solver::Symbol, A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out;
    work_limit::Int64 = Int64(-1),
    progress_callback=NOOP_PHASE_CALLBACK,
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
    prec_meta=nothing,
)
    solver in (:pcg, :minres, :pcr, :lanczos_pcg, :fpcg1, :fpcg2) ||
        error("Unknown solver: $solver")
    skipped = _application_cost_guard_result(prec_meta, work_limit;
        solver_mode, continuation_work_factor, progress_callback)
    skipped !== nothing && return skipped
    if solver == :pcg
        return pcg(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out; work_limit=work_limit, progress_callback=progress_callback,
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
    elseif solver == :minres
        return minres(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out; work_limit=work_limit, progress_callback=progress_callback,
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
    elseif solver == :pcr
        return pcr(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out; work_limit=work_limit, progress_callback=progress_callback,
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
    elseif solver == :lanczos_pcg
        return lanczos_pcg(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out; work_limit=work_limit, progress_callback=progress_callback,
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
    elseif solver == :fpcg1
        return fpcg1(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out; work_limit=work_limit, progress_callback=progress_callback,
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
    elseif solver == :fpcg2
        return fpcg2(A, b, seed, spectral_norm, prec, maxiters, raw_out, conv_out; work_limit=work_limit, progress_callback=progress_callback,
            solver_mode=solver_mode, continuation_work_factor=continuation_work_factor)
    else
        error("Unknown solver: $solver")
    end
end

function _status_message(matrix_name, prec_name, solver, result::SolverResult)
    s = string(solver)
    t = round(result.wall_time_seconds; digits=2)
    if result.status == SKIPPED_COST_GUARD
        return "$matrix_name with $prec_name skipped $s: one preconditioner application exceeds the full work budget."
    elseif result.status == CONVERGED
        return "$matrix_name with $prec_name met $s convergence criteria in $(result.iterations) iterations ($(t)s)."
    elseif result.status == MAX_ITERS
        return "$matrix_name with $prec_name halted in $s due to iteration limit ($(result.iterations)) ($(t)s)."
    elseif result.status == WORK_LIMIT
        return "$matrix_name with $prec_name halted in $s due to work limit ($(t)s)."
    elseif result.status == NAN_DETECTED
        return "$matrix_name with $prec_name halted in $s due to NaN ($(t)s)."
    elseif result.status == BREAKDOWN
        return "$matrix_name with $prec_name halted in $s due to breakdown ($(t)s)."
    elseif result.status == STAGNATED
        return "$matrix_name with $prec_name halted in $s after audited residual stagnation ($(t)s)."
    elseif result.status == KILLED
        return "$matrix_name with $prec_name killed in $s after $(result.iterations) iterations ($(t)s)."
    else
        return "$matrix_name with $prec_name finished $s with status $(result.status) in $(result.iterations) iterations ($(t)s)."
    end
end

end # module TestWrapper
