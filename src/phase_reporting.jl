# Callbacks are explicit: independent tasks/runs never share an active phase stack.
const NOOP_PHASE_CALLBACK = _ -> nothing
const _phase_sequence = Threads.Atomic{UInt64}(0)

function _phase_resource_sample()
    rss, _, user, sys = _read_proc_stat()
    gc = Base.gc_num()
    (rss=rss, cpu=user + sys, allocated=Base.gc_total_bytes(gc),
     gc_ns=gc.total_time, collections=gc.pause, full_collections=gc.full_sweep)
end

function _phase_resource_summary(first, last, peak, samples, seconds)
    Dict{String,Any}(
        "scope" => "process_inclusive", "wall_seconds" => seconds,
        "cpu_seconds" => max(0.0, last.cpu-first.cpu),
        "sampled_peak_rss_bytes" => peak, "rss_samples" => samples,
        "sampling_interval_seconds" => 1.0,
        "allocated_bytes" => max(0, last.allocated-first.allocated),
        "gc_seconds" => max(0, last.gc_ns-first.gc_ns)/1e9,
        "gc_collections" => max(0, last.collections-first.collections),
        "gc_full_collections" => max(0, last.full_collections-first.full_collections))
end

struct PhaseContext{F}
    callback::F
    context::Dict{String,Any}
end
(c::PhaseContext)(event) = c.callback(merge(copy(c.context), event))

struct PhaseScope{F}
    callback::F
    id::String
    phase::String
    context::Dict{String,Any}
end

function (scope::PhaseScope)(event)
    out = merge(copy(scope.context), event)
    if !haskey(out, "operation_id")
        out["operation_id"] = scope.id
        out["phase"] = scope.phase
    elseif out["operation_id"] != scope.id
        get!(out, "parent_operation_id", scope.id)
    end
    scope.callback(out)
end

phase_reporting_enabled(callback) = callback !== NOOP_PHASE_CALLBACK
phase_reporting_enabled(callback::Union{PhaseContext,PhaseScope}) = phase_reporting_enabled(callback.callback)

function phase_callback_with_context(callback; context...)
    phase_reporting_enabled(callback) || return NOOP_PHASE_CALLBACK
    PhaseContext(callback, Dict{String,Any}(String(k) => v for (k, v) in context))
end

function begin_reported_phase(callback, phase::AbstractString; context...)
    phase_reporting_enabled(callback) || return nothing
    id = string(getpid(), ":", time_ns(), ":", Threads.atomic_add!(_phase_sequence, UInt64(1)))
    base = Dict{String,Any}(String(k) => v for (k, v) in context)
    merge!(base, Dict("operation_id" => id, "phase" => String(phase),
                     "phase_started_at" => time(), "schema_version" => 2))
    started = time_ns()
    callback(merge(copy(base), Dict("event" => "phase_start", "timestamp" => time())))
    scope = PhaseScope(callback, id, String(phase), base)
    initial = _phase_resource_sample()
    peak = Ref(initial.rss)
    samples = Ref(1)
    timer = Timer(1.0; interval=1.0) do _
        sample = _phase_resource_sample()
        peak[] = max(peak[], sample.rss)
        samples[] += 1
    end
    return (callback=callback, scope=scope, base=base, started=started,
            initial=initial, peak=peak, samples=samples, timer=timer)
end

function finish_reported_phase(operation; error=nothing)
    operation === nothing && return nothing
    close(operation.timer)
    last = _phase_resource_sample()
    event = merge(copy(operation.base), Dict("event" => error === nothing ? "phase_done" : "phase_error",
        "timestamp" => time(), "elapsed_seconds" => Float64(time_ns() - operation.started) / 1e9))
    error === nothing || (event["error"] = sprint(showerror, error))
    event["resource_summary"] = _phase_resource_summary(operation.initial, last,
        max(operation.peak[], last.rss), operation.samples[] + 1, event["elapsed_seconds"])
    operation.callback(event)
    return nothing
end

"""
    run_scoped_phase(f, callback, phase; context...)

Run `f(child_callback)` with balanced lifecycle events and explicit parent identity.
Use the child callback for nested operations and progress. Durations are monotonic;
timestamps are Unix seconds for correlation with worker/monitor logs.
"""
function run_scoped_phase(f::F, callback, phase::AbstractString; context...) where F
    operation = begin_reported_phase(callback, phase; context...)
    operation === nothing && return f(NOOP_PHASE_CALLBACK)
    result = try
        f(operation.scope)
    catch e
        finish_reported_phase(operation; error=e)
        rethrow()
    end
    finish_reported_phase(operation)
    return result
end

"Run `f()` with lifecycle events. Use `run_scoped_phase` when nesting operations."
run_phase(f::F, callback, phase::AbstractString; context...) where F =
    run_scoped_phase(_ -> f(), callback, phase; context...)

mutable struct NumericalProgress{F,C}
    callback::F
    every_iters::Int
    min_seconds::Float64
    clock::C
    last_report::Float64
    final_reported::Bool
end

function NumericalProgress(callback; every_iters::Int = 100, min_seconds::Real = 60.0,
                           clock = () -> Float64(time_ns()) / 1e9)
    every_iters > 0 || throw(ArgumentError("progress_every_iters must be positive"))
    isfinite(min_seconds) && min_seconds >= 0 ||
        throw(ArgumentError("progress_min_seconds must be finite and nonnegative"))
    NumericalProgress(callback, every_iters, Float64(min_seconds), clock,
                      phase_reporting_enabled(callback) ? clock() : 0.0, false)
end

function progress_due!(p::NumericalProgress, iteration::Int; final::Bool = false)
    phase_reporting_enabled(p.callback) || return false
    final || (iteration > 0 && iteration % p.every_iters == 0) || return false
    now = p.clock()
    final || now - p.last_report >= p.min_seconds || return false
    p.last_report = now
    final && (p.final_reported = true)
    return true
end

function emit_numerical_progress!(p::NumericalProgress; values...)
    event = Dict{String,Any}("event" => "phase_progress", "progress_kind" => "numerical",
                            "timestamp" => time(), "schema_version" => 2)
    invalid = String[]
    for (key, value) in values
        name = String(key)
        if value isa AbstractFloat && !isfinite(value)
            event[name] = nothing
            push!(invalid, name)
        else
            event[name] = value
        end
    end
    event["invalid_fields"] = invalid
    p.callback(event)
    return nothing
end
