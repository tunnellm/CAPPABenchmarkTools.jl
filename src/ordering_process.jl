# Ordering native calls cannot be safely cancelled inside the preparation process.
function _ordering_was_skipped(metadata::AbstractDict, ordering::String)
    ordering = lowercase(ordering)
    ordering != "natural" &&
        haskey(get(metadata, "reordering_skips", Dict()), ordering) &&
        !haskey(get(metadata, "reorderings", Dict()), ordering)
end

function _stop_ordering_process!(process; grace_seconds::Real = 5.0)
    if process_running(process)
        kill(process, Base.SIGTERM)
        timedwait(() -> process_exited(process), grace_seconds; pollint = 0.05)
        process_running(process) && kill(process, Base.SIGKILL)
    end
    wait(process)
    return nothing
end

function _run_ordering_command(command, timeout_seconds::Real)
    isfinite(timeout_seconds) && timeout_seconds > 0 ||
        throw(ArgumentError("ordering timeout must be finite and positive"))
    started = time_ns()
    process = run(command; wait = false)
    timed_out = false
    try
        while process_running(process)
            elapsed = (time_ns() - started) / 1e9
            if elapsed >= timeout_seconds
                timed_out = true
                _stop_ordering_process!(process)
                break
            end
            sleep(min(0.1, timeout_seconds - elapsed))
        end
        wait(process)
        timed_out || success(process) || error("Ordering worker failed (exit code $(process.exitcode))")
    finally
        process_running(process) && _stop_ordering_process!(process)
    end
    return (timed_out = timed_out, elapsed_seconds = (time_ns() - started) / 1e9)
end

function _validate_ordering_permutation(perm::Vector{Int64}, n::Int)
    length(perm) == n || error("Ordering worker returned $(length(perm)) entries; expected $n")
    seen = falses(n)
    for value in perm
        1 <= value <= n || error("Ordering worker returned out-of-range index $value")
        seen[value] && error("Ordering worker returned duplicate index $value")
        seen[value] = true
    end
    return perm
end

function _bounded_ordering(comp_dir::String, files::AbstractDict, n::Int,
                           ordering::String, reorder_dir::String, ci::Int;
                           timeout_seconds::Real = ORDERING_TIMEOUT_SECONDS_DEFAULT)
    ordering in ("rcm", "amd", "metis") || throw(ArgumentError("Unknown ordering: $ordering"))
    # Preserve the caller's resolved environment, including stacked projects.
    # A developed package checkout need not have its own Manifest.toml.
    project = Base.active_project()
    project_args = project === nothing ? `` : `--project=$(dirname(project))`
    load_path = join(Base.load_path(), Sys.iswindows() ? ';' : ':')
    worker = joinpath(@__DIR__, "ordering_worker.jl")
    log_name = "$(ordering)_c$(ci).worker.log"
    mktempdir(reorder_dir; prefix = ".ordering_") do tmp
        output = joinpath(tmp, "permutation.npy.zstd")
        colptr = abspath(joinpath(comp_dir, files["colptr"]))
        rowval = abspath(joinpath(comp_dir, files["rowval"]))
        cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --threads=1 $project_args $worker $ordering $n $colptr $rowval $output`,
                     "JULIA_LOAD_PATH" => load_path)
        outcome = open(joinpath(reorder_dir, log_name), "w") do io
            try
                _run_ordering_command(pipeline(cmd; stdout = io, stderr = io), timeout_seconds)
            catch e
                e isa InterruptException && rethrow()
                error("$ordering worker failed; see $(joinpath(reorder_dir, log_name)): $(sprint(showerror, e))")
            end
        end
        perm = outcome.timed_out ? nothing :
            _validate_ordering_permutation(read_npy_zstd_1d_i64(output), n)
        return (skipped = outcome.timed_out, permutation = perm,
                elapsed_seconds = outcome.elapsed_seconds, worker_log = log_name)
    end
end

function _ordering_permutation(A::SparseMatrixCSC, ordering::String)
    ordering == "rcm" && return SymRCM.symrcm(A)
    ordering == "amd" && return AMD.symamd(A)
    ordering == "metis" && return Metis.permutation(A)[1]
    throw(ArgumentError("Unknown ordering: $ordering"))
end
