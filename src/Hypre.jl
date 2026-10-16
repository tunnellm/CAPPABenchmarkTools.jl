module Hypre

using ..NativeLibraries

using SparseArrays
using Libdl
using ..PreconditionerTypes: Preconditioner, OptionedPreconditioner
using ..ThreadedUtilities: _sleep_julia_workers!

include("hypre/types.jl")
include("hypre/build.jl")
include("hypre/ccall_core.jl")
include("hypre/ccall_boomeramg.jl")
include("hypre/ccall_fsai.jl")
include("hypre/ccall_cheby.jl")
include("hypre/ccall_parasails.jl")
include("hypre/ccall_schwarz.jl")
include("hypre/convert.jl")
include("hypre/preconditioners.jl")


function __init__()
    ensure_built()
end

end # module Hypre
