const _provenance_repositories = Dict{String,Any}()
const _provenance_lock = ReentrantLock()

function _repository_observation(path::String)
    lock(_provenance_lock) do
        get!(_provenance_repositories, abspath(path)) do
            try
                head = strip(read(`git -C $path rev-parse HEAD`, String))
                dirty = !isempty(read(`git -C $path status --porcelain --untracked-files=normal`, String))
                Dict("path"=>abspath(path), "commit"=>head, "dirty"=>dirty,
                    "observed_at"=>time(), "source"=>"checkout_observation")
            catch e
                Dict("path"=>abspath(path), "commit"=>nothing, "dirty"=>nothing,
                    "error"=>sprint(showerror,e))
            end
        end
    end
end

function _thread_affinity_observation()
    result = Dict{String,Any}()
    isdir("/proc/self/task") || return result
    for tid in readdir("/proc/self/task")
        try
            line = match(r"(?m)^Cpus_allowed_list:\s*(.+)$",read("/proc/self/task/$tid/status",String))
            line === nothing || (result[tid] = strip(line[1]))
        catch
            # Threads can exit during the snapshot.
        end
    end
    result
end

function _loaded_pyamg_observation()
    for (_,mod) in Base.loaded_modules
        nameof(mod) == :PythonCall || continue
        try
            pyamg = mod.pyimport("sys").modules.get("pyamg")
            mod.pyis(pyamg,nothing) && return nothing
            return Dict("version"=>mod.pyconvert(String,pyamg.__version__),
                "file"=>mod.pyconvert(String,pyamg.__file__))
        catch e
            return Dict("error"=>sprint(showerror,e))
        end
    end
    nothing
end

"Observational snapshot; does not import PyAMG or change thread settings."
function execution_provenance()
    scheduler = get(ENV,"CAPPA_SCHEDULER_SOURCE","")
    Dict{String,Any}(
        "schema_version"=>1, "observed_at"=>time(), "pid"=>getpid(),
        "hostname"=>gethostname(), "julia_version"=>string(VERSION),
        "julia_threads"=>Threads.nthreads(), "blas_threads"=>BLAS.get_num_threads(),
        "os_thread_affinity"=>_thread_affinity_observation(),
        "thread_environment"=>Dict(k=>get(ENV,k,nothing) for k in
            ("JULIA_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS",
             "OPENBLAS_THREAD_TIMEOUT","OMP_WAIT_POLICY","GOMP_SPINCOUNT","KMP_BLOCKTIME")),
        "blas_config"=>string(BLAS.get_config()),
        "memory_enforcement"=>haskey(ENV,"CAPPA_MEMORY_ENFORCEMENT") ?
            JSON3.read(ENV["CAPPA_MEMORY_ENFORCEMENT"], Dict{String,Any}) : nothing,
        "benchmark"=>_repository_observation(dirname(@__DIR__)),
        "scheduler"=>isempty(scheduler) ? nothing : _repository_observation(scheduler),
        "pyamg"=>_loaded_pyamg_observation(),
        "native_bundle"=>get(ENV,"CAPPA_NATIVE_BUNDLE",nothing),
        "native_manifest_sha256"=>isempty(NativeLibraries.manifest_sha256[]) ? nothing : NativeLibraries.manifest_sha256[],
        "sysimage_bundle"=>get(ENV,"CAPPA_SYSIMAGE_BUNDLE",nothing),
        "job_id"=>get(ENV,"CAPPA_JOB_ID",nothing),
        "run_uuid"=>get(ENV,"CAPPA_RUN_UUID",nothing),
        "raw_storage_version"=>2, "phase_schema_version"=>2)
end

function _write_execution_provenance!(dir::String, context)
    mkpath(dir)
    meta = execution_provenance()
    artifact = "execution_$(getpid())_$(time_ns()).json"
    meta["artifact"] = artifact
    meta["context"] = context
    open(joinpath(dir,artifact),"w") do io
        JSON3.write(io,meta)
    end
    tmp,io=mktemp(dir)
    try
        JSON3.write(io,meta); close(io)
        Base.Filesystem.rename(tmp,joinpath(dir,"execution.json"))
    finally
        isopen(io) && close(io)
        isfile(tmp) && rm(tmp)
    end
    meta
end

function _execution_metadata!(meta,dir)
    path=joinpath(dir,"execution.json")
    if isfile(path)
        snapshot=JSON3.read(read(path,String),Dict{String,Any})
        meta["execution_provenance"] = String(snapshot["artifact"])
        context=snapshot["context"]
        haskey(context,"work_limit_provenance") &&
            (meta["work_limit_provenance"] = context["work_limit_provenance"])
    end
    meta
end
