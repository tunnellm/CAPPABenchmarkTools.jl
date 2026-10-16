# ---------------------------------------------------------------------------
# Schwarz Preconditioner
# ---------------------------------------------------------------------------

# Core
function HYPRE_SchwarzCreate(solver::Ref{HYPRESolver})
    hypre_check(ccall((:HYPRE_SchwarzCreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (Ptr{HYPRESolver},), solver))
end

function HYPRE_SchwarzDestroy(solver::HYPRESolver)
    hypre_check(ccall((:HYPRE_SchwarzDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver,), solver))
end

function HYPRE_SchwarzSetup(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_SchwarzSetup, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_SchwarzSolve(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_SchwarzSolve, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

# Set functions
function HYPRE_SchwarzSetVariant(solver::HYPRESolver, variant::Integer)
    hypre_check(ccall((:HYPRE_SchwarzSetVariant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(variant)))
end

function HYPRE_SchwarzSetOverlap(solver::HYPRESolver, overlap::Integer)
    hypre_check(ccall((:HYPRE_SchwarzSetOverlap, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(overlap)))
end

function HYPRE_SchwarzSetDomainType(solver::HYPRESolver, domain_type::Integer)
    hypre_check(ccall((:HYPRE_SchwarzSetDomainType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(domain_type)))
end

function HYPRE_SchwarzSetNumFunctions(solver::HYPRESolver, num_functions::Integer)
    hypre_check(ccall((:HYPRE_SchwarzSetNumFunctions, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_functions)))
end

function HYPRE_SchwarzSetRelaxWeight(solver::HYPRESolver, relax_weight::Real)
    hypre_check(ccall((:HYPRE_SchwarzSetRelaxWeight, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(relax_weight)))
end

# Get work-counting functions
function HYPRE_SchwarzGetSetupFlops(solver::HYPRESolver, setup_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_SchwarzGetSetupFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_flops))
end

function HYPRE_SchwarzGetSetupGraphOps(solver::HYPRESolver, setup_graph_ops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_SchwarzGetSetupGraphOps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_graph_ops))
end

function HYPRE_SchwarzGetApplyFlops(solver::HYPRESolver, apply_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_SchwarzGetApplyFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, apply_flops))
end

function HYPRE_SchwarzGetMaxDomainOverlap(solver::HYPRESolver, max_overlap::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_SchwarzGetMaxDomainOverlap, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, max_overlap))
end
