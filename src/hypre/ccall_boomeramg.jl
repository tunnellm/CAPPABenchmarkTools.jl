# ==============================================================================
# BoomerAMG Solver — Complete ccall wrappers
# ==============================================================================

# ==============================================================================
# Core functions
# ==============================================================================

function HYPRE_BoomerAMGCreate(solver::Ref{HYPRESolver})
    hypre_check(ccall((:HYPRE_BoomerAMGCreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (Ptr{HYPRESolver},), solver))
end

function HYPRE_BoomerAMGDestroy(solver::HYPRESolver)
    hypre_check(ccall((:HYPRE_BoomerAMGDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver,), solver))
end

function HYPRE_BoomerAMGSetup(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_BoomerAMGSetup, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_BoomerAMGSolve(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_BoomerAMGSolve, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_BoomerAMGSolveT(solver::HYPRESolver, A::HYPREParCSRMatrix, b::HYPREParVector, x::HYPREParVector)
    hypre_check(ccall((:HYPRE_BoomerAMGSolveT, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREParCSRMatrix, HYPREParVector, HYPREParVector),
                      solver, A, b, x))
end

function HYPRE_BoomerAMGSetOldDefault(solver::HYPRESolver)
    hypre_check(ccall((:HYPRE_BoomerAMGSetOldDefault, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver,), solver))
end

# ==============================================================================
# Result getters
# ==============================================================================

function HYPRE_BoomerAMGGetResidual(solver::HYPRESolver, residual::Ref{HYPREParVector})
    hypre_check(ccall((:HYPRE_BoomerAMGGetResidual, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREParVector}), solver, residual))
end

function HYPRE_BoomerAMGGetNumIterations(solver::HYPRESolver, num_iterations::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetNumIterations, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, num_iterations))
end

function HYPRE_BoomerAMGGetCumNumIterations(solver::HYPRESolver, cum_num_iterations::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetCumNumIterations, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, cum_num_iterations))
end

function HYPRE_BoomerAMGGetCumNnzAP(solver::HYPRESolver, cum_nnz_AP::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetCumNnzAP, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, cum_nnz_AP))
end

function HYPRE_BoomerAMGSetCumNnzAP(solver::HYPRESolver, cum_nnz_AP::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCumNnzAP, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(cum_nnz_AP)))
end

function HYPRE_BoomerAMGGetSetupFlops(solver::HYPRESolver, setup_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSetupFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_flops))
end

function HYPRE_BoomerAMGGetSolveFlops(solver::HYPRESolver, solve_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSolveFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, solve_flops))
end

function HYPRE_BoomerAMGGetSetupGraphOps(solver::HYPRESolver, setup_graph_ops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSetupGraphOps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, setup_graph_ops))
end

function HYPRE_BoomerAMGGetApplyFlops(solver::HYPRESolver, apply_flops::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetApplyFlops, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, apply_flops))
end

function HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver::HYPRESolver, rel_resid_norm::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetFinalRelativeResidualNorm, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, rel_resid_norm))
end

# ==============================================================================
# System PDEs
# ==============================================================================

function HYPRE_BoomerAMGSetNumFunctions(solver::HYPRESolver, num_functions::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNumFunctions, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_functions)))
end

function HYPRE_BoomerAMGGetNumFunctions(solver::HYPRESolver, num_functions::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetNumFunctions, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, num_functions))
end

function HYPRE_BoomerAMGSetFilterFunctions(solver::HYPRESolver, filter_functions::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFilterFunctions, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(filter_functions)))
end

function HYPRE_BoomerAMGGetFilterFunctions(solver::HYPRESolver, filter_functions::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetFilterFunctions, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, filter_functions))
end

function HYPRE_BoomerAMGSetDofFunc(solver::HYPRESolver, dof_func::Ptr{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetDofFunc, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, dof_func))
end

# ==============================================================================
# Convergence
# ==============================================================================

function HYPRE_BoomerAMGSetConvergeType(solver::HYPRESolver, type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetConvergeType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(type)))
end

function HYPRE_BoomerAMGGetConvergeType(solver::HYPRESolver, type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetConvergeType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, type))
end

function HYPRE_BoomerAMGSetTol(solver::HYPRESolver, tol::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(tol)))
end

function HYPRE_BoomerAMGGetTol(solver::HYPRESolver, tol::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, tol))
end

function HYPRE_BoomerAMGSetMaxIter(solver::HYPRESolver, max_iter::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMaxIter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_iter)))
end

function HYPRE_BoomerAMGGetMaxIter(solver::HYPRESolver, max_iter::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMaxIter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, max_iter))
end

function HYPRE_BoomerAMGSetMinIter(solver::HYPRESolver, min_iter::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMinIter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(min_iter)))
end

# ==============================================================================
# Grid sizes and levels
# ==============================================================================

function HYPRE_BoomerAMGSetMaxCoarseSize(solver::HYPRESolver, max_coarse_size::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMaxCoarseSize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_coarse_size)))
end

function HYPRE_BoomerAMGGetMaxCoarseSize(solver::HYPRESolver, max_coarse_size::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMaxCoarseSize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, max_coarse_size))
end

function HYPRE_BoomerAMGSetMinCoarseSize(solver::HYPRESolver, min_coarse_size::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMinCoarseSize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(min_coarse_size)))
end

function HYPRE_BoomerAMGGetMinCoarseSize(solver::HYPRESolver, min_coarse_size::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMinCoarseSize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, min_coarse_size))
end

function HYPRE_BoomerAMGSetMaxLevels(solver::HYPRESolver, max_levels::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMaxLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_levels)))
end

function HYPRE_BoomerAMGGetMaxLevels(solver::HYPRESolver, max_levels::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMaxLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, max_levels))
end

# ==============================================================================
# Coarsening
# ==============================================================================

function HYPRE_BoomerAMGSetCoarsenCutFactor(solver::HYPRESolver, coarsen_cut_factor::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCoarsenCutFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(coarsen_cut_factor)))
end

function HYPRE_BoomerAMGGetCoarsenCutFactor(solver::HYPRESolver, coarsen_cut_factor::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetCoarsenCutFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, coarsen_cut_factor))
end

function HYPRE_BoomerAMGSetStrongThreshold(solver::HYPRESolver, strong_threshold::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetStrongThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(strong_threshold)))
end

function HYPRE_BoomerAMGGetStrongThreshold(solver::HYPRESolver, strong_threshold::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetStrongThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, strong_threshold))
end

function HYPRE_BoomerAMGSetStrongThresholdR(solver::HYPRESolver, strong_threshold::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetStrongThresholdR, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(strong_threshold)))
end

function HYPRE_BoomerAMGGetStrongThresholdR(solver::HYPRESolver, strong_threshold::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetStrongThresholdR, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, strong_threshold))
end

function HYPRE_BoomerAMGSetFilterThresholdR(solver::HYPRESolver, filter_threshold::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFilterThresholdR, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(filter_threshold)))
end

function HYPRE_BoomerAMGGetFilterThresholdR(solver::HYPRESolver, filter_threshold::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetFilterThresholdR, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, filter_threshold))
end

function HYPRE_BoomerAMGSetSCommPkgSwitch(solver::HYPRESolver, S_commpkg_switch::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSCommPkgSwitch, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(S_commpkg_switch)))
end

function HYPRE_BoomerAMGSetMaxRowSum(solver::HYPRESolver, max_row_sum::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMaxRowSum, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(max_row_sum)))
end

function HYPRE_BoomerAMGGetMaxRowSum(solver::HYPRESolver, max_row_sum::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMaxRowSum, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, max_row_sum))
end

function HYPRE_BoomerAMGSetCoarsenType(solver::HYPRESolver, coarsen_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCoarsenType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(coarsen_type)))
end

function HYPRE_BoomerAMGGetCoarsenType(solver::HYPRESolver, coarsen_type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetCoarsenType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, coarsen_type))
end

function HYPRE_BoomerAMGSetMeasureType(solver::HYPRESolver, measure_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMeasureType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(measure_type)))
end

function HYPRE_BoomerAMGGetMeasureType(solver::HYPRESolver, measure_type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMeasureType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, measure_type))
end

function HYPRE_BoomerAMGSetSetupType(solver::HYPRESolver, setup_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSetupType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(setup_type)))
end

# ==============================================================================
# Non-Galerkin
# ==============================================================================

function HYPRE_BoomerAMGSetNonGalerkinTol(solver::HYPRESolver, nongalerkin_tol::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNonGalerkinTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(nongalerkin_tol)))
end

function HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver::HYPRESolver, nongalerkin_tol::Real, level::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetLevelNonGalerkinTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal, HYPREInt), solver, Cdouble(nongalerkin_tol), Cint(level)))
end

function HYPRE_BoomerAMGSetNonGalerkTol(solver::HYPRESolver, nongalerk_num_tol::Integer, nongalerk_tol::Ptr{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGSetNonGalerkTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, Ptr{HYPREReal}), solver, Cint(nongalerk_num_tol), nongalerk_tol))
end

# ==============================================================================
# Aggressive coarsening
# ==============================================================================

function HYPRE_BoomerAMGSetAggNumLevels(solver::HYPRESolver, agg_num_levels::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAggNumLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(agg_num_levels)))
end

function HYPRE_BoomerAMGSetNumPaths(solver::HYPRESolver, num_paths::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNumPaths, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_paths)))
end

function HYPRE_BoomerAMGSetCGCIts(solver::HYPRESolver, its::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCGCIts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(its)))
end

# ==============================================================================
# Nodal systems
# ==============================================================================

function HYPRE_BoomerAMGSetNodal(solver::HYPRESolver, nodal::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNodal, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(nodal)))
end

function HYPRE_BoomerAMGSetNodalDiag(solver::HYPRESolver, nodal_diag::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNodalDiag, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(nodal_diag)))
end

function HYPRE_BoomerAMGSetNodalLevels(solver::HYPRESolver, nodal_levels::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNodalLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(nodal_levels)))
end

function HYPRE_BoomerAMGSetKeepSameSign(solver::HYPRESolver, keep_same_sign::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetKeepSameSign, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(keep_same_sign)))
end

# ==============================================================================
# Interpolation
# ==============================================================================

function HYPRE_BoomerAMGSetInterpType(solver::HYPRESolver, interp_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetInterpType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(interp_type)))
end

function HYPRE_BoomerAMGSetInterpRefine(solver::HYPRESolver, num_refine::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetInterpRefine, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_refine)))
end

function HYPRE_BoomerAMGSetTruncFactor(solver::HYPRESolver, trunc_factor::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetTruncFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(trunc_factor)))
end

function HYPRE_BoomerAMGGetTruncFactor(solver::HYPRESolver, trunc_factor::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetTruncFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, trunc_factor))
end

function HYPRE_BoomerAMGSetPMaxElmts(solver::HYPRESolver, P_max_elmts::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetPMaxElmts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(P_max_elmts)))
end

function HYPRE_BoomerAMGGetPMaxElmts(solver::HYPRESolver, P_max_elmts::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetPMaxElmts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, P_max_elmts))
end

function HYPRE_BoomerAMGSetSepWeight(solver::HYPRESolver, sep_weight::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSepWeight, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(sep_weight)))
end

# ==============================================================================
# Aggressive coarsening interpolation
# ==============================================================================

function HYPRE_BoomerAMGSetAggInterpType(solver::HYPRESolver, agg_interp_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAggInterpType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(agg_interp_type)))
end

function HYPRE_BoomerAMGSetAggTruncFactor(solver::HYPRESolver, agg_trunc_factor::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAggTruncFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(agg_trunc_factor)))
end

function HYPRE_BoomerAMGSetAggP12TruncFactor(solver::HYPRESolver, agg_P12_trunc_factor::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAggP12TruncFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(agg_P12_trunc_factor)))
end

function HYPRE_BoomerAMGSetAggPMaxElmts(solver::HYPRESolver, agg_P_max_elmts::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAggPMaxElmts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(agg_P_max_elmts)))
end

function HYPRE_BoomerAMGSetAggP12MaxElmts(solver::HYPRESolver, agg_P12_max_elmts::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAggP12MaxElmts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(agg_P12_max_elmts)))
end

# ==============================================================================
# Interp vectors (GMG)
# ==============================================================================

function HYPRE_BoomerAMGSetInterpVectors(solver::HYPRESolver, num_vectors::Integer, interp_vectors::Ptr{HYPREParVector})
    hypre_check(ccall((:HYPRE_BoomerAMGSetInterpVectors, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, Ptr{HYPREParVector}),
                      solver, Cint(num_vectors), interp_vectors))
end

function HYPRE_BoomerAMGSetInterpVecVariant(solver::HYPRESolver, var::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetInterpVecVariant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(var)))
end

function HYPRE_BoomerAMGSetSmoothInterpVectors(solver::HYPRESolver, smooth_interp_vectors::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSmoothInterpVectors, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(smooth_interp_vectors)))
end

function HYPRE_BoomerAMGSetInterpVecQMax(solver::HYPRESolver, q_max::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetInterpVecQMax, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(q_max)))
end

function HYPRE_BoomerAMGSetInterpVecAbsQTrunc(solver::HYPRESolver, q_trunc::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetInterpVecAbsQTrunc, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(q_trunc)))
end

# ==============================================================================
# GSMG
# ==============================================================================

function HYPRE_BoomerAMGSetGSMG(solver::HYPRESolver, gsmg::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetGSMG, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(gsmg)))
end

function HYPRE_BoomerAMGSetNumSamples(solver::HYPRESolver, num_samples::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNumSamples, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_samples)))
end

# ==============================================================================
# Cycle type
# ==============================================================================

function HYPRE_BoomerAMGSetCycleType(solver::HYPRESolver, cycle_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCycleType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(cycle_type)))
end

function HYPRE_BoomerAMGGetCycleType(solver::HYPRESolver, cycle_type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetCycleType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, cycle_type))
end

function HYPRE_BoomerAMGSetFCycle(solver::HYPRESolver, fcycle::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFCycle, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(fcycle)))
end

function HYPRE_BoomerAMGGetFCycle(solver::HYPRESolver, fcycle::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetFCycle, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, fcycle))
end

# ==============================================================================
# Additive methods
# ==============================================================================

function HYPRE_BoomerAMGSetAdditive(solver::HYPRESolver, addlvl::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAdditive, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(addlvl)))
end

function HYPRE_BoomerAMGGetAdditive(solver::HYPRESolver, additive::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetAdditive, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, additive))
end

function HYPRE_BoomerAMGSetMultAdditive(solver::HYPRESolver, addlvl::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMultAdditive, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(addlvl)))
end

function HYPRE_BoomerAMGGetMultAdditive(solver::HYPRESolver, mult_additive::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetMultAdditive, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, mult_additive))
end

function HYPRE_BoomerAMGSetSimple(solver::HYPRESolver, addlvl::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSimple, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(addlvl)))
end

function HYPRE_BoomerAMGGetSimple(solver::HYPRESolver, simple::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSimple, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, simple))
end

function HYPRE_BoomerAMGSetAddLastLvl(solver::HYPRESolver, add_last_lvl::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAddLastLvl, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(add_last_lvl)))
end

function HYPRE_BoomerAMGSetMultAddTruncFactor(solver::HYPRESolver, add_trunc_factor::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMultAddTruncFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(add_trunc_factor)))
end

function HYPRE_BoomerAMGSetMultAddPMaxElmts(solver::HYPRESolver, add_P_max_elmts::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMultAddPMaxElmts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(add_P_max_elmts)))
end

function HYPRE_BoomerAMGSetAddRelaxType(solver::HYPRESolver, add_rlx_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAddRelaxType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(add_rlx_type)))
end

function HYPRE_BoomerAMGSetAddPMaxElmts(solver::HYPRESolver, add_P_max_elmts::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAddPMaxElmts, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(add_P_max_elmts)))
end

function HYPRE_BoomerAMGSetAddTruncFactor(solver::HYPRESolver, add_trunc_factor::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAddTruncFactor, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(add_trunc_factor)))
end

function HYPRE_BoomerAMGSetAddRelaxWt(solver::HYPRESolver, add_rlx_wt::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetAddRelaxWt, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(add_rlx_wt)))
end

# ==============================================================================
# Sequential / redundant
# ==============================================================================

function HYPRE_BoomerAMGSetSeqThreshold(solver::HYPRESolver, seq_threshold::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSeqThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(seq_threshold)))
end

function HYPRE_BoomerAMGGetSeqThreshold(solver::HYPRESolver, seq_threshold::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSeqThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, seq_threshold))
end

function HYPRE_BoomerAMGSetRedundant(solver::HYPRESolver, redundant::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetRedundant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(redundant)))
end

function HYPRE_BoomerAMGGetRedundant(solver::HYPRESolver, redundant::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetRedundant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, redundant))
end

# ==============================================================================
# Smoothing / relaxation
# ==============================================================================

function HYPRE_BoomerAMGSetNumGridSweeps(solver::HYPRESolver, num_grid_sweeps::Ptr{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetNumGridSweeps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, num_grid_sweeps))
end

function HYPRE_BoomerAMGSetNumSweeps(solver::HYPRESolver, num_sweeps::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNumSweeps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_sweeps)))
end

function HYPRE_BoomerAMGSetCycleNumSweeps(solver::HYPRESolver, num_sweeps::Integer, k::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCycleNumSweeps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, HYPREInt), solver, Cint(num_sweeps), Cint(k)))
end

function HYPRE_BoomerAMGGetCycleNumSweeps(solver::HYPRESolver, num_sweeps::Ref{HYPREInt}, k::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGGetCycleNumSweeps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}, HYPREInt), solver, num_sweeps, Cint(k)))
end

function HYPRE_BoomerAMGSetGridRelaxType(solver::HYPRESolver, grid_relax_type::Ptr{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetGridRelaxType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, grid_relax_type))
end

function HYPRE_BoomerAMGSetRelaxType(solver::HYPRESolver, relax_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetRelaxType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(relax_type)))
end

function HYPRE_BoomerAMGSetCycleRelaxType(solver::HYPRESolver, relax_type::Integer, k::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCycleRelaxType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, HYPREInt), solver, Cint(relax_type), Cint(k)))
end

function HYPRE_BoomerAMGGetCycleRelaxType(solver::HYPRESolver, relax_type::Ref{HYPREInt}, k::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGGetCycleRelaxType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}, HYPREInt), solver, relax_type, Cint(k)))
end

function HYPRE_BoomerAMGSetRelaxOrder(solver::HYPRESolver, relax_order::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetRelaxOrder, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(relax_order)))
end

function HYPRE_BoomerAMGSetGridRelaxPoints(solver::HYPRESolver, grid_relax_points::Ptr{Ptr{HYPREInt}})
    hypre_check(ccall((:HYPRE_BoomerAMGSetGridRelaxPoints, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{Ptr{HYPREInt}}), solver, grid_relax_points))
end

function HYPRE_BoomerAMGSetRelaxWeight(solver::HYPRESolver, relax_weight::Ptr{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGSetRelaxWeight, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, relax_weight))
end

function HYPRE_BoomerAMGSetRelaxWt(solver::HYPRESolver, relax_weight::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetRelaxWt, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(relax_weight)))
end

function HYPRE_BoomerAMGSetLevelRelaxWt(solver::HYPRESolver, relax_weight::Real, level::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetLevelRelaxWt, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal, HYPREInt), solver, Cdouble(relax_weight), Cint(level)))
end

function HYPRE_BoomerAMGSetOmega(solver::HYPRESolver, omega::Ptr{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGSetOmega, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, omega))
end

function HYPRE_BoomerAMGSetOuterWt(solver::HYPRESolver, omega::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetOuterWt, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(omega)))
end

function HYPRE_BoomerAMGSetLevelOuterWt(solver::HYPRESolver, omega::Real, level::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetLevelOuterWt, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal, HYPREInt), solver, Cdouble(omega), Cint(level)))
end

# ==============================================================================
# Chebyshev smoother settings (within AMG)
# ==============================================================================

function HYPRE_BoomerAMGSetChebyOrder(solver::HYPRESolver, order::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetChebyOrder, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(order)))
end

function HYPRE_BoomerAMGSetChebyFraction(solver::HYPRESolver, ratio::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetChebyFraction, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(ratio)))
end

function HYPRE_BoomerAMGSetChebyScale(solver::HYPRESolver, scale::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetChebyScale, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(scale)))
end

function HYPRE_BoomerAMGSetChebyVariant(solver::HYPRESolver, variant::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetChebyVariant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(variant)))
end

function HYPRE_BoomerAMGSetChebyEigEst(solver::HYPRESolver, eig_est::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetChebyEigEst, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(eig_est)))
end

# ==============================================================================
# Smooth type (complex smoothers)
# ==============================================================================

function HYPRE_BoomerAMGSetSmoothType(solver::HYPRESolver, smooth_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSmoothType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(smooth_type)))
end

function HYPRE_BoomerAMGGetSmoothType(solver::HYPRESolver, smooth_type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSmoothType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, smooth_type))
end

function HYPRE_BoomerAMGSetSmoothNumLevels(solver::HYPRESolver, smooth_num_levels::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSmoothNumLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(smooth_num_levels)))
end

function HYPRE_BoomerAMGGetSmoothNumLevels(solver::HYPRESolver, smooth_num_levels::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSmoothNumLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, smooth_num_levels))
end

function HYPRE_BoomerAMGSetSmoothNumSweeps(solver::HYPRESolver, smooth_num_sweeps::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSmoothNumSweeps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(smooth_num_sweeps)))
end

function HYPRE_BoomerAMGGetSmoothNumSweeps(solver::HYPRESolver, smooth_num_sweeps::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSmoothNumSweeps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, smooth_num_sweeps))
end

# ==============================================================================
# Schwarz smoother settings
# ==============================================================================

function HYPRE_BoomerAMGSetVariant(solver::HYPRESolver, variant::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetVariant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(variant)))
end

function HYPRE_BoomerAMGGetVariant(solver::HYPRESolver, variant::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetVariant, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, variant))
end

function HYPRE_BoomerAMGSetOverlap(solver::HYPRESolver, overlap::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetOverlap, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(overlap)))
end

function HYPRE_BoomerAMGGetOverlap(solver::HYPRESolver, overlap::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetOverlap, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, overlap))
end

function HYPRE_BoomerAMGSetDomainType(solver::HYPRESolver, domain_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetDomainType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(domain_type)))
end

function HYPRE_BoomerAMGGetDomainType(solver::HYPRESolver, domain_type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetDomainType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, domain_type))
end

function HYPRE_BoomerAMGSetSchwarzRlxWeight(solver::HYPRESolver, schwarz_rlx_weight::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSchwarzRlxWeight, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(schwarz_rlx_weight)))
end

function HYPRE_BoomerAMGGetSchwarzRlxWeight(solver::HYPRESolver, schwarz_rlx_weight::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetSchwarzRlxWeight, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, schwarz_rlx_weight))
end

function HYPRE_BoomerAMGSetSchwarzUseNonSymm(solver::HYPRESolver, use_nonsymm::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSchwarzUseNonSymm, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(use_nonsymm)))
end

# ==============================================================================
# ParaSails smoother settings (within AMG)
# ==============================================================================

function HYPRE_BoomerAMGSetSym(solver::HYPRESolver, sym::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSym, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(sym)))
end

function HYPRE_BoomerAMGSetLevel(solver::HYPRESolver, level::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetLevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(level)))
end

function HYPRE_BoomerAMGSetThreshold(solver::HYPRESolver, threshold::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(threshold)))
end

function HYPRE_BoomerAMGSetFilter(solver::HYPRESolver, filter::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFilter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(filter)))
end

# ==============================================================================
# Pilut smoother settings
# ==============================================================================

function HYPRE_BoomerAMGSetDropTol(solver::HYPRESolver, drop_tol::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetDropTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(drop_tol)))
end

function HYPRE_BoomerAMGSetMaxNzPerRow(solver::HYPRESolver, max_nz_per_row::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetMaxNzPerRow, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_nz_per_row)))
end

# ==============================================================================
# Euclid smoother settings
# ==============================================================================

function HYPRE_BoomerAMGSetEuclidFile(solver::HYPRESolver, euclidfile::Cstring)
    hypre_check(ccall((:HYPRE_BoomerAMGSetEuclidFile, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Cstring), solver, euclidfile))
end

function HYPRE_BoomerAMGSetEuLevel(solver::HYPRESolver, eu_level::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetEuLevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(eu_level)))
end

function HYPRE_BoomerAMGSetEuSparseA(solver::HYPRESolver, eu_sparse_A::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetEuSparseA, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(eu_sparse_A)))
end

function HYPRE_BoomerAMGSetEuBJ(solver::HYPRESolver, eu_bj::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetEuBJ, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(eu_bj)))
end

# ==============================================================================
# ILU smoother settings
# ==============================================================================

function HYPRE_BoomerAMGSetILUType(solver::HYPRESolver, ilu_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_type)))
end

function HYPRE_BoomerAMGSetILULevel(solver::HYPRESolver, ilu_lfil::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILULevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_lfil)))
end

function HYPRE_BoomerAMGSetILUMaxRowNnz(solver::HYPRESolver, ilu_max_row_nnz::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUMaxRowNnz, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_max_row_nnz)))
end

function HYPRE_BoomerAMGSetILUMaxIter(solver::HYPRESolver, ilu_max_iter::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUMaxIter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_max_iter)))
end

function HYPRE_BoomerAMGSetILUDroptol(solver::HYPRESolver, ilu_droptol::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUDroptol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(ilu_droptol)))
end

function HYPRE_BoomerAMGSetILUTriSolve(solver::HYPRESolver, ilu_tri_solve::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUTriSolve, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_tri_solve)))
end

function HYPRE_BoomerAMGSetILULowerJacobiIters(solver::HYPRESolver, ilu_lower_jacobi_iters::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILULowerJacobiIters, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_lower_jacobi_iters)))
end

function HYPRE_BoomerAMGSetILUUpperJacobiIters(solver::HYPRESolver, ilu_upper_jacobi_iters::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUUpperJacobiIters, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_upper_jacobi_iters)))
end

function HYPRE_BoomerAMGSetILULocalReordering(solver::HYPRESolver, ilu_reordering_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILULocalReordering, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_reordering_type)))
end

function HYPRE_BoomerAMGSetILUIterSetupType(solver::HYPRESolver, ilu_iter_setup_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUIterSetupType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_iter_setup_type)))
end

function HYPRE_BoomerAMGSetILUIterSetupOption(solver::HYPRESolver, ilu_iter_setup_option::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUIterSetupOption, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_iter_setup_option)))
end

function HYPRE_BoomerAMGSetILUIterSetupMaxIter(solver::HYPRESolver, ilu_iter_setup_max_iter::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUIterSetupMaxIter, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(ilu_iter_setup_max_iter)))
end

function HYPRE_BoomerAMGSetILUIterSetupTolerance(solver::HYPRESolver, ilu_iter_setup_tolerance::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetILUIterSetupTolerance, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(ilu_iter_setup_tolerance)))
end

# ==============================================================================
# FSAI smoother settings (within AMG)
# ==============================================================================

function HYPRE_BoomerAMGSetFSAIAlgoType(solver::HYPRESolver, algo_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIAlgoType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(algo_type)))
end

function HYPRE_BoomerAMGSetFSAILocalSolveType(solver::HYPRESolver, local_solve_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAILocalSolveType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(local_solve_type)))
end

function HYPRE_BoomerAMGSetFSAIMaxSteps(solver::HYPRESolver, max_steps::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIMaxSteps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_steps)))
end

function HYPRE_BoomerAMGSetFSAIMaxStepSize(solver::HYPRESolver, max_step_size::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIMaxStepSize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_step_size)))
end

function HYPRE_BoomerAMGSetFSAIMaxNnzRow(solver::HYPRESolver, max_nnz_row::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIMaxNnzRow, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(max_nnz_row)))
end

function HYPRE_BoomerAMGSetFSAINumLevels(solver::HYPRESolver, num_levels::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAINumLevels, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_levels)))
end

function HYPRE_BoomerAMGSetFSAIThreshold(solver::HYPRESolver, threshold::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(threshold)))
end

function HYPRE_BoomerAMGSetFSAIEigMaxIters(solver::HYPRESolver, eig_max_iters::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIEigMaxIters, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(eig_max_iters)))
end

function HYPRE_BoomerAMGSetFSAIKapTolerance(solver::HYPRESolver, kap_tolerance::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetFSAIKapTolerance, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(kap_tolerance)))
end

# ==============================================================================
# Restriction / AIR
# ==============================================================================

function HYPRE_BoomerAMGSetRestriction(solver::HYPRESolver, restr_par::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetRestriction, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(restr_par)))
end

function HYPRE_BoomerAMGSetIsTriangular(solver::HYPRESolver, is_triangular::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetIsTriangular, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(is_triangular)))
end

function HYPRE_BoomerAMGSetGMRESSwitchR(solver::HYPRESolver, gmres_switch::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetGMRESSwitchR, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(gmres_switch)))
end

# ==============================================================================
# A-dropping
# ==============================================================================

function HYPRE_BoomerAMGSetADropTol(solver::HYPRESolver, A_drop_tol::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetADropTol, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(A_drop_tol)))
end

function HYPRE_BoomerAMGSetADropType(solver::HYPRESolver, A_drop_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetADropType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(A_drop_type)))
end

# ==============================================================================
# Print / debug / logging
# ==============================================================================

function HYPRE_BoomerAMGSetPrintFileName(solver::HYPRESolver, print_file_name::Cstring)
    hypre_check(ccall((:HYPRE_BoomerAMGSetPrintFileName, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Cstring), solver, print_file_name))
end

function HYPRE_BoomerAMGSetPrintLevel(solver::HYPRESolver, print_level::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetPrintLevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(print_level)))
end

function HYPRE_BoomerAMGGetPrintLevel(solver::HYPRESolver, print_level::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetPrintLevel, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, print_level))
end

function HYPRE_BoomerAMGSetLogging(solver::HYPRESolver, logging::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetLogging, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(logging)))
end

function HYPRE_BoomerAMGGetLogging(solver::HYPRESolver, logging::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetLogging, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, logging))
end

function HYPRE_BoomerAMGSetDebugFlag(solver::HYPRESolver, debug_flag::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetDebugFlag, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(debug_flag)))
end

function HYPRE_BoomerAMGGetDebugFlag(solver::HYPRESolver, debug_flag::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetDebugFlag, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, debug_flag))
end

# ==============================================================================
# RAP / GPU optimization
# ==============================================================================

function HYPRE_BoomerAMGSetRAP2(solver::HYPRESolver, rap2::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetRAP2, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(rap2)))
end

function HYPRE_BoomerAMGSetModuleRAP2(solver::HYPRESolver, mod_rap2::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetModuleRAP2, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(mod_rap2)))
end

function HYPRE_BoomerAMGSetKeepTranspose(solver::HYPRESolver, keepTranspose::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetKeepTranspose, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(keepTranspose)))
end

# ==============================================================================
# Grid visualization
# ==============================================================================

function HYPRE_BoomerAMGSetPlotGrids(solver::HYPRESolver, plotgrids::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetPlotGrids, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(plotgrids)))
end

function HYPRE_BoomerAMGSetPlotFileName(solver::HYPRESolver, plotfilename::Cstring)
    hypre_check(ccall((:HYPRE_BoomerAMGSetPlotFileName, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Cstring), solver, plotfilename))
end

function HYPRE_BoomerAMGSetCoordDim(solver::HYPRESolver, coorddim::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCoordDim, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(coorddim)))
end

function HYPRE_BoomerAMGSetCoordinates(solver::HYPRESolver, coordinates::Ptr{Cfloat})
    hypre_check(ccall((:HYPRE_BoomerAMGSetCoordinates, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{Cfloat}), solver, coordinates))
end

function HYPRE_BoomerAMGGetGridHierarchy(solver::HYPRESolver, cgrid::Ptr{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetGridHierarchy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, cgrid))
end

# ==============================================================================
# C/F points
# ==============================================================================

function HYPRE_BoomerAMGSetCPoints(solver::HYPRESolver, cpt_coarse_level::Integer, num_cpt_coarse::Integer, cpt_coarse_index::Ptr{HYPREBigInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetCPoints, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, HYPREInt, Ptr{HYPREBigInt}),
                      solver, Cint(cpt_coarse_level), Cint(num_cpt_coarse), cpt_coarse_index))
end

function HYPRE_BoomerAMGSetCpointsToKeep(solver::HYPRESolver, cpt_coarse_level::Integer, num_cpt_coarse::Integer, cpt_coarse_index::Ptr{HYPREBigInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetCpointsToKeep, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, HYPREInt, Ptr{HYPREBigInt}),
                      solver, Cint(cpt_coarse_level), Cint(num_cpt_coarse), cpt_coarse_index))
end

function HYPRE_BoomerAMGSetFPoints(solver::HYPRESolver, num_fpt::Integer, fpt_index::Ptr{HYPREBigInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetFPoints, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, Ptr{HYPREBigInt}),
                      solver, Cint(num_fpt), fpt_index))
end

function HYPRE_BoomerAMGSetIsolatedFPoints(solver::HYPRESolver, num_isolated_fpt::Integer, isolated_fpt_index::Ptr{HYPREBigInt})
    hypre_check(ccall((:HYPRE_BoomerAMGSetIsolatedFPoints, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt, Ptr{HYPREBigInt}),
                      solver, Cint(num_isolated_fpt), isolated_fpt_index))
end

# ==============================================================================
# Miscellaneous
# ==============================================================================

function HYPRE_BoomerAMGSetSabs(solver::HYPRESolver, Sabs::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetSabs, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(Sabs)))
end

function HYPRE_BoomerAMGSetPostInterpType(solver::HYPRESolver, post_interp_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetPostInterpType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(post_interp_type)))
end

function HYPRE_BoomerAMGGetPostInterpType(solver::HYPRESolver, post_interp_type::Ref{HYPREInt})
    hypre_check(ccall((:HYPRE_BoomerAMGGetPostInterpType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREInt}), solver, post_interp_type))
end

function HYPRE_BoomerAMGSetJacobiTruncThreshold(solver::HYPRESolver, jacobi_trunc_threshold::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetJacobiTruncThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(jacobi_trunc_threshold)))
end

function HYPRE_BoomerAMGGetJacobiTruncThreshold(solver::HYPRESolver, jacobi_trunc_threshold::Ref{HYPREReal})
    hypre_check(ccall((:HYPRE_BoomerAMGGetJacobiTruncThreshold, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, Ptr{HYPREReal}), solver, jacobi_trunc_threshold))
end

function HYPRE_BoomerAMGSetNumCRRelaxSteps(solver::HYPRESolver, num_CR_relax_steps::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetNumCRRelaxSteps, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(num_CR_relax_steps)))
end

function HYPRE_BoomerAMGSetCRRate(solver::HYPRESolver, CR_rate::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCRRate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(CR_rate)))
end

function HYPRE_BoomerAMGSetCRStrongTh(solver::HYPRESolver, CR_strong_th::Real)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCRStrongTh, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREReal), solver, Cdouble(CR_strong_th)))
end

function HYPRE_BoomerAMGSetCRUseCG(solver::HYPRESolver, CR_use_CG::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetCRUseCG, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(CR_use_CG)))
end

function HYPRE_BoomerAMGSetISType(solver::HYPRESolver, IS_type::Integer)
    hypre_check(ccall((:HYPRE_BoomerAMGSetISType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPRESolver, HYPREInt), solver, Cint(IS_type)))
end
