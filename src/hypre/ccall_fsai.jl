# ---------------------------------------------------------------------------
# FSAI Solver
# ---------------------------------------------------------------------------

# Core
function HYPRE_FSAICreate(solver::Ref{HYPRESolver})
    hypre_check(ccall((:HYPRE_FSAICreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (Ptr{HYPRESolver},), solver))
end

function HYPRE_FSAIDestroy(solver::HYPRESolver)
    hypre_check(ccall((:HYPRE_FSAIDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver,), solver))
end

function HYPRE_FSAISetup(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_FSAISetup, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_FSAISolve(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_FSAISolve, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

# Set functions
function HYPRE_FSAISetAlgoType(solver::HYPRESolver, algo_type::Integer)
    hypre_check(ccall((:HYPRE_FSAISetAlgoType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(algo_type)))
end

function HYPRE_FSAISetLocalSolveType(solver::HYPRESolver, local_solve_type::Integer)
    hypre_check(ccall((:HYPRE_FSAISetLocalSolveType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(local_solve_type)))
end

function HYPRE_FSAISetMaxSteps(solver::HYPRESolver, max_steps::Integer)
    hypre_check(ccall((:HYPRE_FSAISetMaxSteps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_steps)))
end

function HYPRE_FSAISetMaxStepSize(solver::HYPRESolver, max_step_size::Integer)
    hypre_check(ccall((:HYPRE_FSAISetMaxStepSize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_step_size)))
end

function HYPRE_FSAISetMaxNnzRow(solver::HYPRESolver, max_nnz_row::Integer)
    hypre_check(ccall((:HYPRE_FSAISetMaxNnzRow, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_nnz_row)))
end

function HYPRE_FSAISetNumLevels(solver::HYPRESolver, num_levels::Integer)
    hypre_check(ccall((:HYPRE_FSAISetNumLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_levels)))
end

function HYPRE_FSAISetThreshold(solver::HYPRESolver, threshold::Real)
    hypre_check(ccall((:HYPRE_FSAISetThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(threshold)))
end

function HYPRE_FSAISetKapTolerance(solver::HYPRESolver, kap_tolerance::Real)
    hypre_check(ccall((:HYPRE_FSAISetKapTolerance, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(kap_tolerance)))
end

function HYPRE_FSAISetOmega(solver::HYPRESolver, omega::Real)
    hypre_check(ccall((:HYPRE_FSAISetOmega, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(omega)))
end

function HYPRE_FSAISetMaxIterations(solver::HYPRESolver, max_iterations::Integer)
    hypre_check(ccall((:HYPRE_FSAISetMaxIterations, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_iterations)))
end

function HYPRE_FSAISetEigMaxIters(solver::HYPRESolver, eig_max_iters::Integer)
    hypre_check(ccall((:HYPRE_FSAISetEigMaxIters, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(eig_max_iters)))
end

function HYPRE_FSAISetTolerance(solver::HYPRESolver, tolerance::Real)
    hypre_check(ccall((:HYPRE_FSAISetTolerance, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(tolerance)))
end

function HYPRE_FSAISetPrintLevel(solver::HYPRESolver, print_level::Integer)
    hypre_check(ccall((:HYPRE_FSAISetPrintLevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(print_level)))
end

function HYPRE_FSAISetZeroGuess(solver::HYPRESolver, zero_guess::Integer)
    hypre_check(ccall((:HYPRE_FSAISetZeroGuess, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(zero_guess)))
end

# Get functions
function HYPRE_FSAIGetSetupFlops(solver::HYPRESolver, setup_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_FSAIGetSetupFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_flops))
end

function HYPRE_FSAIGetSetupGraphOps(solver::HYPRESolver, setup_graph_ops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_FSAIGetSetupGraphOps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_graph_ops))
end

function HYPRE_FSAIGetApplyFlops(solver::HYPRESolver, apply_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_FSAIGetApplyFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, apply_flops))
end
