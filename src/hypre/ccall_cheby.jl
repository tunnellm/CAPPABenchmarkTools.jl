# ---------------------------------------------------------------------------
# Chebyshev Solver
# ---------------------------------------------------------------------------

# Core
function HYPRE_ParChebyCreate(solver::Ref{HYPRESolver})
    hypre_check(ccall((:HYPRE_ParChebyCreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (Ptr{HYPRESolver},), solver))
end

function HYPRE_ParChebyDestroy(solver::HYPRESolver)
    hypre_check(ccall((:HYPRE_ParChebyDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver,), solver))
end

function HYPRE_ParChebySetup(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_ParChebySetup, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_ParChebySolve(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_ParChebySolve, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

# Set functions
function HYPRE_ParChebySetMaxIterations(solver::HYPRESolver, max_iterations::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetMaxIterations, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_iterations)))
end

function HYPRE_ParChebySetTolerance(solver::HYPRESolver, tolerance::Real)
    hypre_check(ccall((:HYPRE_ParChebySetTolerance, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(tolerance)))
end

function HYPRE_ParChebySetPrintLevel(solver::HYPRESolver, print_level::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetPrintLevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(print_level)))
end

function HYPRE_ParChebySetLogging(solver::HYPRESolver, logging::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetLogging, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(logging)))
end

function HYPRE_ParChebySetOrder(solver::HYPRESolver, order::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetOrder, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(order)))
end

function HYPRE_ParChebySetVariant(solver::HYPRESolver, variant::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetVariant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(variant)))
end

function HYPRE_ParChebySetScale(solver::HYPRESolver, scale::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetScale, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(scale)))
end

function HYPRE_ParChebySetEigRatio(solver::HYPRESolver, eig_ratio::Real)
    hypre_check(ccall((:HYPRE_ParChebySetEigRatio, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(eig_ratio)))
end

function HYPRE_ParChebySetEigEst(solver::HYPRESolver, eig_est::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetEigEst, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(eig_est)))
end

function HYPRE_ParChebySetMinMaxEigEst(solver::HYPRESolver, min_eig_est::Real, max_eig_est::Real)
    hypre_check(ccall((:HYPRE_ParChebySetMinMaxEigEst, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal, HYPREReal),
                      solver, Cdouble(min_eig_est), Cdouble(max_eig_est)))
end

function HYPRE_ParChebyGetMinMaxEigEst(solver::HYPRESolver, min_eig_est::Ref{HYPREReal}, max_eig_est::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParChebyGetMinMaxEigEst, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}, Ptr{HYPREReal}),
                      solver, min_eig_est, max_eig_est))
end

function HYPRE_ParChebySetZeroGuess(solver::HYPRESolver, zero_guess::Integer)
    hypre_check(ccall((:HYPRE_ParChebySetZeroGuess, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(zero_guess)))
end

# Get functions
function HYPRE_ParChebyGetSetupFlops(solver::HYPRESolver, setup_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParChebyGetSetupFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_flops))
end

function HYPRE_ParChebyGetSetupGraphOps(solver::HYPRESolver, setup_graph_ops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParChebyGetSetupGraphOps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_graph_ops))
end

function HYPRE_ParChebyGetApplyFlops(solver::HYPRESolver, apply_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_ParChebyGetApplyFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, apply_flops))
end
