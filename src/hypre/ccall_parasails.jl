# ---------------------------------------------------------------------------
# ParaSails Preconditioner
# ---------------------------------------------------------------------------

# Core
function HYPRE_ParaSailsCreate(comm::HYPREMPIComm, solver::Ref{HYPRESolver})
    hypre_check(ccall((:HYPRE_ParaSailsCreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREMPIComm, Ptr{HYPRESolver}), comm, solver))
end

function HYPRE_ParaSailsDestroy(solver::HYPRESolver)
    hypre_check(ccall((:HYPRE_ParaSailsDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver,), solver))
end

function HYPRE_ParaSailsSetup(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_ParaSailsSetup, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_ParaSailsSolve(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_ParaSailsSolve, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

# Set functions
function HYPRE_ParaSailsSetParams(solver::HYPRESolver, thresh::Real, nlevels::Integer)
    hypre_check(ccall((:HYPRE_ParaSailsSetParams, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal, HYPREInt),
                      solver, Cdouble(thresh), Cint(nlevels)))
end

function HYPRE_ParaSailsSetFilter(solver::HYPRESolver, filter::Real)
    hypre_check(ccall((:HYPRE_ParaSailsSetFilter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(filter)))
end

function HYPRE_ParaSailsSetSym(solver::HYPRESolver, sym::Integer)
    hypre_check(ccall((:HYPRE_ParaSailsSetSym, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(sym)))
end

function HYPRE_ParaSailsSetLoadbal(solver::HYPRESolver, loadbal::Real)
    hypre_check(ccall((:HYPRE_ParaSailsSetLoadbal, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(loadbal)))
end

function HYPRE_ParaSailsSetReuse(solver::HYPRESolver, reuse::Integer)
    hypre_check(ccall((:HYPRE_ParaSailsSetReuse, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(reuse)))
end

function HYPRE_ParaSailsSetLogging(solver::HYPRESolver, logging::Integer)
    hypre_check(ccall((:HYPRE_ParaSailsSetLogging, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(logging)))
end

function HYPRE_ParaSailsBuildIJMatrix(solver::HYPRESolver, pij_A::Ref{HYPREIJMatrix})
    hypre_check(ccall((:HYPRE_ParaSailsBuildIJMatrix, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREIJMatrix}), solver, pij_A))
end

function HYPRE_ParaSailsSetThresh(solver::HYPRESolver, thresh::Real)
    hypre_check(ccall((:HYPRE_ParaSailsSetThresh, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(thresh)))
end

function HYPRE_ParaSailsGetThresh(solver::HYPRESolver, thresh::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParaSailsGetThresh, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, thresh))
end

function HYPRE_ParaSailsSetNlevels(solver::HYPRESolver, nlevels::Integer)
    hypre_check(ccall((:HYPRE_ParaSailsSetNlevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(nlevels)))
end

function HYPRE_ParaSailsGetNlevels(solver::HYPRESolver, nlevels::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_ParaSailsGetNlevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, nlevels))
end

function HYPRE_ParaSailsGetFilter(solver::HYPRESolver, filter::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParaSailsGetFilter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, filter))
end

function HYPRE_ParaSailsGetSym(solver::HYPRESolver, sym::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_ParaSailsGetSym, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, sym))
end

function HYPRE_ParaSailsGetLoadbal(solver::HYPRESolver, loadbal::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParaSailsGetLoadbal, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, loadbal))
end

function HYPRE_ParaSailsGetReuse(solver::HYPRESolver, reuse::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_ParaSailsGetReuse, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, reuse))
end

function HYPRE_ParaSailsGetLogging(solver::HYPRESolver, logging::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_ParaSailsGetLogging, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, logging))
end

# Get work-counting functions
function HYPRE_ParaSailsGetSetupFlops(solver::HYPRESolver, setup_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParaSailsGetSetupFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_flops))
end

function HYPRE_ParaSailsGetSetupGraphOps(solver::HYPRESolver, setup_graph_ops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParaSailsGetSetupGraphOps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_graph_ops))
end

function HYPRE_ParaSailsGetApplyFlops(solver::HYPRESolver, apply_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParaSailsGetApplyFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, apply_flops))
end
