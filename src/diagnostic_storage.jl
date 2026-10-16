# Raw storage blocks are independent of Krylov/tridiagonal restart blocks.
const DIAGNOSTIC_BLOCK_ROWS = 500_000

function _diagnostic_directory(path::AbstractString)
    endswith(path, ".npy.zstd") ? chop(path; tail=9) : path * ".blocks"
end

function _diagnostic_manifest(path::AbstractString)
    isdir(path) && return joinpath(path, "manifest.json")
    basename(path) == "manifest.json" && return String(path)
    return joinpath(_diagnostic_directory(path), "manifest.json")
end

function _diagnostic_storage_metadata!(meta, solver_dir)
    path = joinpath(solver_dir, "raw_data", "manifest.json")
    if isfile(path)
        manifest = JSON3.read(read(path, String))
        meta["raw_data_storage"] = Dict("version"=>Int(manifest["version"]),
            "manifest"=>"raw_data/manifest.json", "attempt"=>get(manifest,"attempt",nothing))
        meta["raw_data_schema"] = get(manifest, "schema", Dict("version"=>1))
        artifacts = Dict{String,Any}[]
        for block in 0:Int(get(meta,"correction_restarts",0)), suffix in ("_tridiag","_recurrence")
            artifact = _diagnostic_manifest(_coefficient_path(
                joinpath(solver_dir,"raw_data.npy.zstd"),suffix,block))
            isfile(artifact) || continue
            contents = JSON3.read(read(artifact,String))
            schema = get(contents,"schema",Dict())
            # A retry may leave older unreferenced files in the directory.
            attempt = get(manifest,"attempt",nothing)
            attempt !== nothing && get(schema,"source_attempt",nothing) == attempt || continue
            push!(artifacts,Dict("manifest"=>relpath(artifact,solver_dir),
                "krylov_block"=>block,"schema"=>schema,"rows"=>contents["rows"]))
        end
        meta["coefficient_storage"] = artifacts
    end
    return meta
end

function _atomic_diagnostic_json(path, value)
    tmp, io = mktemp(dirname(path))
    try
        JSON3.write(io, value)
        close(io)
        Base.Filesystem.rename(tmp, path)
    finally
        isopen(io) && close(io)
        isfile(tmp) && rm(tmp)
    end
end

# Legacy completed streams have no attempt marker. New writers publish the
# marker before replacing a manifest, invalidating old terminal metadata even
# if a process dies before its first row or during CSV/metadata finalization.
function _diagnostic_attempt_complete(meta, solver_dir)
    root = joinpath(solver_dir, "raw_data")
    marker = joinpath(root, "active_attempt.json")
    manifest = joinpath(root, "manifest.json")
    !isfile(marker) && !isfile(manifest) && return true
    try
        stream = JSON3.read(read(manifest, String))
        get(stream, "finalized", true) || return false
        isfile(marker) || return true
        active = JSON3.read(read(marker, String))["attempt"]
        recorded = get(get(meta, "raw_data_storage", Dict()), "attempt", nothing)
        return active == get(stream, "attempt", nothing) == recorded
    catch
        return false
    end
end

function _diagnostic_child(root, file)
    file = String(file)
    (isempty(file) || isabspath(file) || any(==(".."), splitpath(file))) &&
        error("invalid diagnostic artifact path: $file")
    path = joinpath(root, file)
    isfile(path) || error("missing diagnostic block: $path")
    resolved = realpath(path)
    relative = relpath(resolved, realpath(root))
    any(==(".."), splitpath(relative)) && error("diagnostic artifact escapes directory: $file")
    return resolved
end

mutable struct DiagnosticBlockReader <: AbstractMatrix{Float64}
    paths::Vector{String}
    ends::Vector{Int}
    ncols::Int
    cached_index::Int
    cached::Matrix{Float64}
end

Base.size(data::DiagnosticBlockReader) = (isempty(data.ends) ? 0 : last(data.ends), data.ncols)

function _diagnostic_block!(data::DiagnosticBlockReader, index::Int)
    if data.cached_index != index
        start = index == 1 ? 1 : data.ends[index-1] + 1
        expected = (data.ends[index] - start + 1, data.ncols)
        block = _read_npy_zstd_file(data.paths[index]; expected_size=expected)
        data.cached = block
        data.cached_index = index
    end
    return data.cached
end

function Base.getindex(data::DiagnosticBlockReader, row::Int, col::Int)
    @boundscheck checkbounds(data, row, col)
    index = searchsortedfirst(data.ends, row)
    block = _diagnostic_block!(data, index)
    offset = index == 1 ? 0 : data.ends[index-1]
    return block[row-offset, col]
end

"""
    open_diagnostic_data(path; require_finalized=false)

Open a legacy NPY file (v1), a v1 manifest with a `file` entry, or a v2
numbered-block manifest. `path` may be the manifest, its directory, or the
legacy logical filename. A sibling manifest takes precedence over an old
file. V2 returns a matrix-like snapshot with a one-block read cache; it never
concatenates the history. Only manifest-listed blocks are visible. Unfinalized
snapshots expose the durable prefix, not proof that the solver completed.
"""
function open_diagnostic_data(path::AbstractString; require_finalized::Bool=false)
    manifest_path = _diagnostic_manifest(path)
    if !isfile(manifest_path)
        (isdir(path) || basename(path) == "manifest.json") &&
            error("missing diagnostic manifest: $manifest_path")
        return _read_npy_zstd_file(String(path))
    end
    meta = JSON3.read(read(manifest_path, String))
    version = Int(meta["version"])
    root = dirname(manifest_path)
    if version == 1
        return _read_npy_zstd_file(_diagnostic_child(root, meta["file"]))
    end
    version == 2 || error("unsupported diagnostic storage version: $version")
    meta["format"] == "cappa_diagnostics" || error("invalid diagnostic format")
    meta["dtype"] == "<f8" || error("unsupported diagnostic dtype")
    require_finalized && !meta["finalized"] && error("diagnostic stream is not finalized")
    ncols = Int(meta["ncols"])
    ncols > 0 || error("invalid diagnostic column count")
    length(meta["columns"]) == ncols || error("diagnostic column schema mismatch")
    paths = String[]
    ends = Int[]
    total = 0
    for block in meta["blocks"]
        Int(block["first_row"]) == total + 1 || error("noncontiguous diagnostic blocks")
        rows = Int(block["rows"])
        rows > 0 || error("empty diagnostic block")
        total = Base.checked_add(total, rows)
        push!(paths, _diagnostic_child(root, block["file"]))
        push!(ends, total)
    end
    length(unique(paths)) == length(paths) || error("duplicate diagnostic blocks")
    total == Int(meta["rows"]) || error("diagnostic manifest row count mismatch")
    return DiagnosticBlockReader(paths, ends, ncols, 0, zeros(0, 0))
end

"""
    read_npy_zstd(path)

Materialize a v1 file or v2 diagnostic stream. Use `open_diagnostic_data` for
bounded-memory analysis. The filename API also resolves its sibling v2 directory.
"""
function read_npy_zstd(path::AbstractString)
    data = open_diagnostic_data(path)
    data isa DiagnosticBlockReader || return data
    result = Matrix{Float64}(undef, size(data))
    start = 1
    for index in eachindex(data.paths)
        block = _diagnostic_block!(data, index)
        copyto!(view(result, start:data.ends[index], :), block)
        start = data.ends[index] + 1
    end
    return result
end

mutable struct DiagnosticBlockWriter <: AbstractMatrix{Float64}
    directory::String
    attempt::String
    buffer::Matrix{Float64}
    previous_row::Vector{Float64}
    columns::Vector{String}
    limit::Int
    committed::Int
    blocks::Vector{Dict{String,Any}}
    callback::Any
    flush_seconds::Float64
    compressed_bytes::Int
    schema::Dict{String,Any}
end

function DiagnosticBlockWriter(path, limit::Int, columns; block_rows::Int=DIAGNOSTIC_BLOCK_ROWS,
                               callback=NOOP_PHASE_CALLBACK,
                               schema=Dict{String,Any}("version"=>2,
                                   "alpha_semantics"=>"signed",
                                   "reductions"=>"julia_threaded"))
    limit > 0 && block_rows > 0 || throw(ArgumentError("diagnostic row limits must be positive"))
    directory = _diagnostic_directory(path)
    mkpath(directory)
    attempt = mktempdir(directory; prefix="attempt_", cleanup=false)
    names = String.(columns)
    schema = Dict{String,Any}(schema)
    if get(schema,"kind",nothing) === nothing
        schema["kind"] = "solver_diagnostics"
        schema["step_scalar_alignment"] = "ortho_coeff_1" in names ?
            "next_direction_before_any_restart_setup; see recurrence alpha_used for actual update" :
            "current_update; row_zero_is_initial_state"
    end
    data = DiagnosticBlockWriter(directory, attempt,
        zeros(min(limit, block_rows), length(names)), zeros(length(names)),
        names, limit, 0, Dict{String,Any}[], callback, 0.0, 0, Dict{String,Any}(schema))
    _atomic_diagnostic_json(joinpath(directory, "active_attempt.json"),
        Dict("attempt"=>basename(attempt)))
    _publish_diagnostic_manifest(data, false)
    return data
end

Base.size(data::DiagnosticBlockWriter) = (data.limit, length(data.columns))

function Base.getindex(data::DiagnosticBlockWriter, row::Int, col::Int)
    @boundscheck checkbounds(data, row, col)
    # Progress reporting and early breakdown can refer to the last flushed row.
    row == data.committed && return data.previous_row[col]
    return data.buffer[row-data.committed, col]
end

function Base.setindex!(data::DiagnosticBlockWriter, value, row::Int, col::Int)
    @boundscheck checkbounds(data, row, col)
    data.buffer[row-data.committed, col] = value
    return value
end

function _publish_diagnostic_manifest(data::DiagnosticBlockWriter, finalized::Bool)
    _atomic_diagnostic_json(joinpath(data.directory, "manifest.json"), Dict(
        "version"=>2, "format"=>"cappa_diagnostics", "dtype"=>"<f8",
        "ncols"=>length(data.columns), "columns"=>data.columns,
        "rows"=>data.committed, "finalized"=>finalized, "blocks"=>data.blocks,
        "schema"=>data.schema, "attempt"=>basename(data.attempt)))
end

_committed_iteration(data::DiagnosticBlockWriter) =
    get(data.schema,"first_iteration",0) + data.committed - 1

function _flush_diagnostic_block!(data::DiagnosticBlockWriter, rows::Int)
    0 < rows <= size(data.buffer, 1) || throw(ArgumentError("invalid diagnostic flush length"))
    file = lpad(string(length(data.blocks)+1), 6, '0') * ".npy.zstd"
    path = joinpath(data.attempt, file)
    operation = begin_reported_phase(data.callback, "diagnostic_flush";
        block_index=length(data.blocks)+1, rows=rows,
        manifest_path=abspath(joinpath(data.directory,"manifest.json")),
        stream_kind=get(data.schema,"kind","solver_diagnostics"),
        krylov_block=get(data.schema,"krylov_block",nothing),
        last_committed_iteration=_committed_iteration(data), buffered_rows=rows)
    started = time_ns()
    try
        _write_diagnostic_block!(data, rows, path)
        bytes = filesize(path)
        data.compressed_bytes += bytes
        data.flush_seconds += Float64(time_ns()-started)/1e9
        if operation !== nothing
            merge!(operation.base, Dict("compressed_bytes"=>bytes,
                "last_committed_iteration"=>_committed_iteration(data), "buffered_rows"=>0,
                "total_flush_seconds"=>data.flush_seconds,
                "total_compressed_bytes"=>data.compressed_bytes))
        end
    catch e
        finish_reported_phase(operation; error=e)
        rethrow()
    end
    finish_reported_phase(operation)
    return nothing
end

function _write_diagnostic_block!(data::DiagnosticBlockWriter, rows::Int, path::String)
    tmp, io = mktemp(data.attempt)
    close(io)
    try
        write_npy_zstd(tmp, view(data.buffer, 1:rows, :))
        Base.Filesystem.rename(tmp, path)
    finally
        isfile(tmp) && rm(tmp)
    end
    copyto!(data.previous_row, view(data.buffer, rows, :))
    push!(data.blocks, Dict("file"=>relpath(path, data.directory),
        "first_row"=>data.committed+1, "rows"=>rows))
    data.committed += rows
    # Publication follows compressor close and atomic block rename. Unlisted
    # files from an interrupted flush cannot be mistaken for committed rows.
    _publish_diagnostic_manifest(data, false)
    return nothing
end

_diagnostic_progress(data::DiagnosticBlockWriter, row) = (
    last_committed_iteration=data.committed-1, buffered_rows=max(0,row-data.committed),
    diagnostic_flush_seconds=data.flush_seconds, diagnostic_compressed_bytes=data.compressed_bytes)

function _prepare_diagnostic_row!(data::DiagnosticBlockWriter, required::Int, limit::Int)
    limit == data.limit && 1 <= required <= limit || throw(ArgumentError("invalid diagnostic row"))
    if required > data.committed + size(data.buffer, 1)
        required == data.committed + size(data.buffer, 1) + 1 || error("diagnostic rows must be sequential")
        _flush_diagnostic_block!(data, size(data.buffer, 1))
        fill!(data.buffer, 0.0)
    end
    return data
end

function _finish_diagnostic_data!(data::DiagnosticBlockWriter, nrows::Int; callback=data.callback)
    data.callback = callback
    0 <= nrows-data.committed <= size(data.buffer, 1) || error("invalid final diagnostic row")
    nrows > data.committed && _flush_diagnostic_block!(data, nrows-data.committed)
    _publish_diagnostic_manifest(data, true)
    data.buffer = zeros(0, 0)
    return open_diagnostic_data(data.directory; require_finalized=true)
end
