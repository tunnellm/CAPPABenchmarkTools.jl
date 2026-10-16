# Spectral radius, not largest algebraic eigenvalue, supplies the symmetric
# operator 2-norm. Cache in natural ordering independently of solve ordering.
const SPECTRAL_NORM_POLICY = (version=1, target="radius", m=30, max_restarts=20, tolerance=1e-3, seed=42)

function _cached_spectral_norm(load_natural, cache_dir, source_identity;
                               progress_callback=NOOP_PHASE_CALLBACK)
    # Serialize misses across preconditioner jobs. The lock is released by the
    # OS on process exit, including cancellation; no stale lock recovery needed.
    lock_io = try
        mkpath(cache_dir)
        open(joinpath(cache_dir, "estimate.lock"), "a+")
    catch e
        e isa InterruptException && rethrow()
        @warn "Spectral norm cache is not writable; estimating without a shared lock" cache_dir exception=e
        nothing
    end
    lock_io === nothing && return _cached_spectral_norm_unlocked(load_natural, cache_dir, source_identity; progress_callback)
    try
        while ccall(:flock, Cint, (Cint,Cint), fd(lock_io), 2 | 4) != 0
            Base.Libc.errno() in (Base.Libc.EAGAIN, Base.Libc.EINTR) ||
                error("Cannot lock spectral norm cache: errno $(Base.Libc.errno())")
            sleep(0.1)
        end
        return _cached_spectral_norm_unlocked(load_natural, cache_dir, source_identity; progress_callback)
    finally
        close(lock_io)
    end
end

function _cached_spectral_norm_unlocked(load_natural, cache_dir, source_identity;
                                        progress_callback=NOOP_PHASE_CALLBACK)
    identity = Dict("source"=>source_identity, "policy"=>Dict(string(k)=>v for (k,v) in pairs(SPECTRAL_NORM_POLICY)))
    key = bytes2hex(SHA.sha256(JSON3.write(identity)))
    path = joinpath(cache_dir, key * ".json")
    if isfile(path)
        cached = try
            entry = JSON3.read(read(path,String), Dict{String,Any})
            entry["identity"] == identity || error("spectral norm cache identity mismatch")
            body = entry["payload"]::String
            bytes2hex(SHA.sha256(body)) == entry["sha256"] || error("spectral norm cache checksum mismatch")
            value = JSON3.read(body, Dict{String,Any})
            isfinite(value["estimate"]) && value["estimate"] >= 0 || error("invalid spectral norm estimate")
            value
        catch e
            e isa InterruptException && rethrow()
            @warn "Invalid spectral norm cache; recomputing" path exception=e
            nothing
        end
        cached === nothing || return cached
    end
    value = run_phase(progress_callback, "spectral_norm_estimation"; cache_path=path) do
        A = load_natural()
        lo, hi, fma, iters, rlo, rhi, glo, ghi = estimate_eigbounds(A;
            target=:radius, m=SPECTRAL_NORM_POLICY.m,
            max_restarts=SPECTRAL_NORM_POLICY.max_restarts,
            upper_tol=SPECTRAL_NORM_POLICY.tolerance, seed=SPECTRAL_NORM_POLICY.seed,
            progress_callback)
        negative = abs(lo) > abs(hi)
        estimate = max(abs(lo), abs(hi))
        residual, gap = negative ? (rlo,glo) : (rhi,ghi)
        isfinite(estimate) && estimate >= 0 || error("nonfinite spectral norm estimate")
        converged = residual <= SPECTRAL_NORM_POLICY.tolerance * estimate && residual < gap
        Dict{String,Any}("estimate"=>estimate, "kind"=>"largest_magnitude_ritz",
            "status"=>converged ? "ritz_converged" : "budget_exhausted",
            "ritz_residual"=>_json_safe(residual), "iterations"=>iters,
            "estimation_fma"=>fma, "canonical_ordering"=>"natural", "cache_key"=>key,
            "policy"=>identity["policy"], "source"=>source_identity)
    end
    # Publication failure does not discard a usable estimate, e.g. read-only caches.
    try
        body = JSON3.write(value)
        _atomic_write_json(path, Dict("identity"=>identity, "payload"=>body,
            "sha256"=>bytes2hex(SHA.sha256(body))))
    catch e
        e isa InterruptException && rethrow()
        @warn "Cannot cache spectral norm estimate" path exception=e
    end
    return value
end

function component_spectral_norm(desc::ComponentDescriptor, natural::SparseMatrixCSC;
                                  wants_sdd=false, progress_callback=NOOP_PHASE_CALLBACK)
    scaled = !(wants_sdd && desc.unscaled_is_sdd && !desc.scaled_is_sdd)
    source = Dict("matrix_sha256"=>_component_prepare_fingerprint(natural), "scaled"=>scaled)
    cache_dir = joinpath(desc.base_dir,desc.name,"components","analysis_cache","spectral_norm","c$(desc.component_index)")
    return _cached_spectral_norm(cache_dir, source; progress_callback) do
        A = _copy_csc_values(natural)
        scaled && scale_symmetric!(A)
        A
    end
end
