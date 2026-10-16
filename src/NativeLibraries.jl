module NativeLibraries

using Libdl, SHA
import JSON3

const paths = Dict{String,String}()
const manifest_sha256 = Ref("")

function __init__()
    empty!(paths)
    manifest_sha256[] = ""
    root = get(ENV, "CAPPA_NATIVE_BUNDLE", "")
    isempty(root) && return
    root = abspath(root)
    manifest = JSON3.read(read(joinpath(root, "native.json"), String))
    manifest_sha256[] = bytes2hex(open(sha256, joinpath(root,"native.json")))
    manifest.schema_version == 1 || error("Unsupported native bundle schema")
    manifest.hostname == gethostname() || error("Native bundle belongs to another host")
    manifest.machine == Sys.MACHINE || error("Native bundle architecture mismatch")
    python = joinpath(root, "python", "bin", "python")
    get(ENV, "JULIA_CONDAPKG_BACKEND", "") == "Null" || error("Native bundles require the Null CondaPkg backend")
    abspath(get(ENV, "JULIA_PYTHONCALL_EXE", "")) == python || error("Native bundle Python mismatch")
    for (name, entry) in pairs(manifest.libraries)
        path = joinpath(root, "lib", String(name))
        isfile(path) || error("Missing bundled native library: $path; rebuild offline")
        bytes2hex(open(sha256, path)) == entry.sha256 || error("Native library checksum mismatch: $path")
        paths[String(name)] = path
    end
end

# Function-valued ccall library specifications are resolved in the running process,
# not from a path frozen into a Julia precompile cache or sysimage.
library(default::String) = isempty(get(ENV, "CAPPA_NATIVE_BUNDLE", "")) ? default : paths[basename(default)]

allow_build() = isempty(get(ENV,"CAPPA_NATIVE_BUNDLE","")) || error("Native builds are disabled in bundle mode")

function ensure(default::String, builder::Function)
    path = library(default)
    if !isfile(path)
        isempty(get(ENV, "CAPPA_NATIVE_BUNDLE", "")) || error("Missing bundled library: $path; builds are disabled")
        builder()
    end
    handle = dlopen(path)
    dlclose(handle)
    nothing
end

end
