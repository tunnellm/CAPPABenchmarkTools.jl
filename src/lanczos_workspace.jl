# Owned by one sequential analysis walk; never shared by concurrent calls.
mutable struct LanczosWorkspace
    basis::Matrix{Float64}
    w::Vector{Float64}
    h::Vector{Float64}
    y_min::Vector{Float64}
    y_max::Vector{Float64}
end
LanczosWorkspace() = LanczosWorkspace(Matrix{Float64}(undef,0,0), Float64[], Float64[], Float64[], Float64[])

function _release_lanczos_workspace!(ws::LanczosWorkspace)
    ws.basis = Matrix{Float64}(undef,0,0)
    ws.w = Float64[]; ws.h = Float64[]
    ws.y_min = Float64[]; ws.y_max = Float64[]
    return ws
end

function _lanczos_workspace(workspace, n, k; both=false, restart=false)
    ws = workspace === nothing ? LanczosWorkspace() : workspace
    if size(ws.basis,1) != n || size(ws.basis,2) < k
        ws.basis = Matrix{Float64}(undef,n,k)
    end
    resize!(ws.w,n); fill!(ws.w,0.0)
    resize!(ws.h,k); fill!(ws.h,0.0)
    resize!(ws.y_max,restart ? n : 0); fill!(ws.y_max,0.0)
    resize!(ws.y_min,both ? n : 0); fill!(ws.y_min,0.0)
    # Each active basis column is assigned before it enters a BLAS view.
    return ws
end
