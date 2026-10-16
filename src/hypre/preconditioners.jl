# ---------------------------------------------------------------------------
# High-level HYPRE preconditioner constructors
# Each returns a Preconditioner with work counting.
# ---------------------------------------------------------------------------

using SparseArrays

# Mutable handle container so we can attach a GC finalizer.
# The apply! closure captures this, keeping it alive as long as Preconditioner is.
mutable struct HypreHandles
    solver::HYPRESolver
    ij_A::HYPREIJMatrix
    ij_b::HYPREIJVector
    ij_x::HYPREIJVector
    parcsr_A::HYPREParCSRMatrix
    par_b::HYPREParVector
    par_x::HYPREParVector
end

"""
    _build_boomeramg_hierarchy(A; setup_kwargs...) -> OptionedPreconditioner

Internal: builds the AMG hierarchy once and returns an `OptionedPreconditioner`
whose `configure` closure accepts apply-time parameters (cycle_type, fcycle,
num_sweeps, relax_order, relax_wt, outer_wt) and returns `(Preconditioner, meta)`.

Only one `Preconditioner` from a given hierarchy should be in use at a time
(the apply closure shares Hypre's internal vectors).
"""
function _build_boomeramg_hierarchy(A::SparseMatrixCSC{Float64,Int};
                        # Convergence (preconditioner mode: 1 iter, 0 tol)
                        max_iter::Int = 1,
                        tol::Real = 0.0,
                        min_iter::Int = 0,
                        converge_type::Union{Symbol,Int} = :rhs,
                        # Coarsening
                        coarsen_type::Union{Symbol,Int} = :HMIS,
                        coarsen_cut_factor::Int = 0,
                        cgc_its::Int = 1,
                        strong_threshold::Real = 0.25,
                        strong_threshold_R::Real = 0.25,
                        filter_threshold_R::Real = 0.0,
                        max_row_sum::Real = 0.9,
                        measure_type::Union{Symbol,Int} = :local,
                        # Levels
                        max_levels::Int = 25,
                        max_coarse_size::Int = 9,
                        min_coarse_size::Int = 1,
                        # Interpolation
                        interp_type::Union{Symbol,Int} = :ext_plus_i,
                        interp_refine::Int = 0,
                        trunc_factor::Real = 0.0,
                        P_max_elmts::Int = 4,
                        sep_weight::Union{Symbol,Int,Bool} = false,
                        post_interp_type::Int = 0,
                        jacobi_trunc_threshold::Real = 0.0,
                        # Aggressive coarsening
                        agg_num_levels::Int = 0,
                        num_paths::Int = 1,
                        agg_interp_type::Union{Symbol,Int} = :multipass,
                        agg_trunc_factor::Real = 0.0,
                        agg_P12_trunc_factor::Real = 0.0,
                        agg_P_max_elmts::Int = 0,
                        agg_P12_max_elmts::Int = 0,
                        # Relaxation (setup-time: smoother type is baked in)
                        # Default matches Hypre's native: l1-GS-forward down, l1-GS-backward up, direct coarse.
                        # Set relax_type to override all phases uniformly (coarse always gets :direct).
                        # Set relax_down/relax_up/relax_coarse individually to override specific phases.
                        relax_type::Union{Nothing,Symbol,Int} = nothing,
                        relax_down::Union{Symbol,Int} = :l1_gauss_seidel_forward,
                        relax_up::Union{Symbol,Int} = :l1_gauss_seidel_backward,
                        relax_coarse::Union{Symbol,Int} = :direct,
                        # Chebyshev smoother (within AMG)
                        cheby_order::Int = 2,
                        cheby_fraction::Real = 0.3,
                        cheby_scale::Union{Symbol,Int,Bool} = true,
                        cheby_variant::Int = 0,
                        cheby_eig_est::Int = 10,
                        # Complex smoothers
                        smooth_type::Union{Symbol,Int} = :schwarz,
                        smooth_num_levels::Int = 0,
                        smooth_num_sweeps::Int = 1,
                        # Non-Galerkin
                        nongalerkin_tol::Real = 0.0,
                        # A-dropping
                        A_drop_tol::Real = 0.0,
                        A_drop_type::Union{Symbol,Int} = :none,
                        # Nodal
                        num_functions::Int = 1,
                        nodal::Union{Symbol,Int} = :unknown,
                        nodal_diag::Int = 0,
                        nodal_levels::Int = 0,
                        keep_same_sign::Int = 0,
                        # Sequential
                        seq_threshold::Int = 0,
                        redundant::Int = 0,
                        # RAP
                        rap2::Union{Symbol,Int,Bool} = false,
                        mod_rap2::Union{Symbol,Int,Bool} = false,
                        keep_transpose::Union{Symbol,Int,Bool} = false,
                        # CR
                        num_CR_relax_steps::Int = 2,
                        CR_rate::Real = 0.7,
                        CR_strong_th::Real = 0.0,
                        CR_use_CG::Union{Symbol,Int,Bool} = false,
                        IS_type::Int = 0,
                        # Sabs
                        Sabs::Union{Symbol,Int,Bool} = false,
                        # ILU smoother
                        ilu_type::Union{Symbol,Int} = :BJ_ILUk,
                        ilu_level::Int = 0,
                        ilu_max_row_nnz::Int = 0,
                        ilu_max_iter::Int = 1,
                        ilu_droptol::Real = 0.0,
                        ilu_tri_solve::Int = 1,
                        ilu_lower_jacobi_iters::Int = 1,
                        ilu_upper_jacobi_iters::Int = 1,
                        ilu_local_reordering::Union{Symbol,Int} = :none,
                        # FSAI smoother (within AMG)
                        fsai_algo_type::Union{Symbol,Int} = :adaptive,
                        fsai_local_solve_type::Union{Symbol,Int} = :gauss_jordan,
                        fsai_max_steps::Int = 5,
                        fsai_max_step_size::Int = 3,
                        fsai_max_nnz_row::Int = 0,
                        fsai_threshold::Real = 0.0,
                        fsai_eig_max_iters::Int = 5,
                        fsai_kap_tolerance::Real = 1.0e-3,
                        # ParaSails smoother (within AMG)
                        parasails_level::Int = 1,
                        parasails_threshold::Real = 0.1,
                        parasails_filter::Real = 0.1,
                        # Schwarz smoother (within AMG)
                        schwarz_domain_type::Union{Symbol,Int} = :agglomerate,
                        schwarz_overlap::Int = 1,
                        schwarz_variant::Union{Symbol,Int} = :multiplicative,
                        schwarz_rlx_weight::Real = 1.0,
                        # GSMG
                        gsmg::Union{Symbol,Int,Bool} = false,
                        num_samples::Int = 0,
                        # Debug / output
                        print_level::Int = 0,
                        logging::Int = 0,
                        debug_flag::Int = 0,
                        # Apply-time defaults (used for initial setup cycle measurement)
                        cycle_type::Union{Symbol,Int} = :V,
                        fcycle::Union{Symbol,Int,Bool} = false,
                        relax_order::Union{Symbol,Int,Bool} = false,
                        num_sweeps::Int = 1,
                        relax_wt::Real = 1.0,
                        outer_wt::Real = 1.0)

    # Resolve symbolic options to integers
    converge_type = resolve_option(converge_type, CONVERGE_TYPES, "converge_type")
    coarsen_type = resolve_option(coarsen_type, COARSEN_TYPES, "coarsen_type")
    measure_type = resolve_option(measure_type, MEASURE_TYPES, "measure_type")
    interp_type = resolve_option(interp_type, INTERP_TYPES, "interp_type")
    sep_weight = resolve_bool(sep_weight, "sep_weight")
    agg_interp_type = resolve_option(agg_interp_type, AGG_INTERP_TYPES, "agg_interp_type")
    cycle_type = resolve_option(cycle_type, CYCLE_TYPES, "cycle_type")
    fcycle = resolve_bool(fcycle, "fcycle")
    # Resolve relaxation: relax_type overrides all phases if given
    if !isnothing(relax_type)
        rt = resolve_option(relax_type, RELAX_TYPES, "relax_type")
        relax_down = rt
        relax_up = rt
        relax_coarse = resolve_option(:direct, RELAX_TYPES, "relax_coarse")
    else
        relax_down = resolve_option(relax_down, RELAX_TYPES, "relax_down")
        relax_up = resolve_option(relax_up, RELAX_TYPES, "relax_up")
        relax_coarse = resolve_option(relax_coarse, RELAX_TYPES, "relax_coarse")
    end
    relax_order = resolve_bool(relax_order, "relax_order")
    cheby_scale = resolve_bool(cheby_scale, "cheby_scale")
    smooth_type = resolve_option(smooth_type, SMOOTH_TYPES, "smooth_type")
    A_drop_type = resolve_option(A_drop_type, A_DROP_TYPES, "A_drop_type")
    nodal = resolve_option(nodal, NODAL_TYPES, "nodal")
    rap2 = resolve_bool(rap2, "rap2")
    mod_rap2 = resolve_bool(mod_rap2, "mod_rap2")
    keep_transpose = resolve_bool(keep_transpose, "keep_transpose")
    CR_use_CG = resolve_bool(CR_use_CG, "CR_use_CG")
    Sabs = resolve_bool(Sabs, "Sabs")
    ilu_type = resolve_option(ilu_type, ILU_TYPES, "ilu_type")
    ilu_local_reordering = resolve_option(ilu_local_reordering, ILU_REORDERING_TYPES, "ilu_local_reordering")
    fsai_algo_type = resolve_option(fsai_algo_type, FSAI_ALGO_TYPES, "fsai_algo_type")
    fsai_local_solve_type = resolve_option(fsai_local_solve_type, FSAI_SOLVE_TYPES, "fsai_local_solve_type")
    schwarz_domain_type = resolve_option(schwarz_domain_type, SCHWARZ_DOMAIN_TYPES, "schwarz_domain_type")
    schwarz_variant = resolve_option(schwarz_variant, SCHWARZ_VARIANT_TYPES, "schwarz_variant")
    gsmg = resolve_bool(gsmg, "gsmg")

    n = size(A, 1)
    ensure_initialized()

    ij_A, parcsr_A = julia_to_hypre_matrix(A)
    ij_b, par_b, ij_x, par_x = julia_to_hypre_vectors(n)

    solver_ref = Ref{HYPRESolver}(C_NULL)
    HYPRE_BoomerAMGCreate(solver_ref)
    s = solver_ref[]

    # Convergence (preconditioner mode)
    HYPRE_BoomerAMGSetMaxIter(s, max_iter)
    HYPRE_BoomerAMGSetTol(s, tol)
    HYPRE_BoomerAMGSetMinIter(s, min_iter)
    HYPRE_BoomerAMGSetConvergeType(s, converge_type)

    # Coarsening
    HYPRE_BoomerAMGSetCoarsenType(s, coarsen_type)
    HYPRE_BoomerAMGSetCoarsenCutFactor(s, coarsen_cut_factor)
    HYPRE_BoomerAMGSetCGCIts(s, cgc_its)
    HYPRE_BoomerAMGSetStrongThreshold(s, strong_threshold)
    HYPRE_BoomerAMGSetStrongThresholdR(s, strong_threshold_R)
    HYPRE_BoomerAMGSetFilterThresholdR(s, filter_threshold_R)
    HYPRE_BoomerAMGSetMaxRowSum(s, max_row_sum)
    HYPRE_BoomerAMGSetMeasureType(s, measure_type)

    # Levels
    HYPRE_BoomerAMGSetMaxLevels(s, max_levels)
    HYPRE_BoomerAMGSetMaxCoarseSize(s, max_coarse_size)
    HYPRE_BoomerAMGSetMinCoarseSize(s, min_coarse_size)

    # Interpolation
    HYPRE_BoomerAMGSetInterpType(s, interp_type)
    HYPRE_BoomerAMGSetInterpRefine(s, interp_refine)
    HYPRE_BoomerAMGSetTruncFactor(s, trunc_factor)
    HYPRE_BoomerAMGSetPMaxElmts(s, P_max_elmts)
    HYPRE_BoomerAMGSetSepWeight(s, sep_weight)
    HYPRE_BoomerAMGSetPostInterpType(s, post_interp_type)
    HYPRE_BoomerAMGSetJacobiTruncThreshold(s, jacobi_trunc_threshold)

    # Aggressive coarsening
    HYPRE_BoomerAMGSetAggNumLevels(s, agg_num_levels)
    HYPRE_BoomerAMGSetNumPaths(s, num_paths)
    HYPRE_BoomerAMGSetAggInterpType(s, agg_interp_type)
    HYPRE_BoomerAMGSetAggTruncFactor(s, agg_trunc_factor)
    HYPRE_BoomerAMGSetAggP12TruncFactor(s, agg_P12_trunc_factor)
    HYPRE_BoomerAMGSetAggPMaxElmts(s, agg_P_max_elmts)
    HYPRE_BoomerAMGSetAggP12MaxElmts(s, agg_P12_max_elmts)

    # Apply-time params set with initial defaults (configure will override)
    HYPRE_BoomerAMGSetCycleType(s, cycle_type)
    HYPRE_BoomerAMGSetFCycle(s, fcycle)
    HYPRE_BoomerAMGSetRelaxOrder(s, relax_order)
    HYPRE_BoomerAMGSetNumSweeps(s, num_sweeps)
    HYPRE_BoomerAMGSetRelaxWt(s, relax_wt)
    HYPRE_BoomerAMGSetOuterWt(s, outer_wt)

    # Setup-time relaxation type (baked in, per-phase)
    HYPRE_BoomerAMGSetCycleRelaxType(s, relax_down, 1)
    HYPRE_BoomerAMGSetCycleRelaxType(s, relax_up, 2)
    HYPRE_BoomerAMGSetCycleRelaxType(s, relax_coarse, 3)

    # Chebyshev smoother (within AMG)
    HYPRE_BoomerAMGSetChebyOrder(s, cheby_order)
    HYPRE_BoomerAMGSetChebyFraction(s, cheby_fraction)
    HYPRE_BoomerAMGSetChebyScale(s, cheby_scale)
    HYPRE_BoomerAMGSetChebyVariant(s, cheby_variant)
    HYPRE_BoomerAMGSetChebyEigEst(s, cheby_eig_est)

    # Complex smoothers
    HYPRE_BoomerAMGSetSmoothType(s, smooth_type)
    HYPRE_BoomerAMGSetSmoothNumLevels(s, smooth_num_levels)
    HYPRE_BoomerAMGSetSmoothNumSweeps(s, smooth_num_sweeps)

    # Non-Galerkin
    nongalerkin_tol > 0 && HYPRE_BoomerAMGSetNonGalerkinTol(s, nongalerkin_tol)

    # A-dropping
    HYPRE_BoomerAMGSetADropTol(s, A_drop_tol)
    HYPRE_BoomerAMGSetADropType(s, A_drop_type)

    # Nodal
    HYPRE_BoomerAMGSetNumFunctions(s, num_functions)
    HYPRE_BoomerAMGSetNodal(s, nodal)
    HYPRE_BoomerAMGSetNodalDiag(s, nodal_diag)
    HYPRE_BoomerAMGSetNodalLevels(s, nodal_levels)
    HYPRE_BoomerAMGSetKeepSameSign(s, keep_same_sign)

    # Sequential
    HYPRE_BoomerAMGSetSeqThreshold(s, seq_threshold)
    HYPRE_BoomerAMGSetRedundant(s, redundant)

    # RAP
    HYPRE_BoomerAMGSetRAP2(s, rap2)
    HYPRE_BoomerAMGSetModuleRAP2(s, mod_rap2)
    HYPRE_BoomerAMGSetKeepTranspose(s, keep_transpose)

    # CR
    HYPRE_BoomerAMGSetNumCRRelaxSteps(s, num_CR_relax_steps)
    HYPRE_BoomerAMGSetCRRate(s, CR_rate)
    HYPRE_BoomerAMGSetCRStrongTh(s, CR_strong_th)
    HYPRE_BoomerAMGSetCRUseCG(s, CR_use_CG)
    HYPRE_BoomerAMGSetISType(s, IS_type)

    # Sabs
    HYPRE_BoomerAMGSetSabs(s, Sabs)

    # ILU smoother
    HYPRE_BoomerAMGSetILUType(s, ilu_type)
    HYPRE_BoomerAMGSetILULevel(s, ilu_level)
    HYPRE_BoomerAMGSetILUMaxRowNnz(s, ilu_max_row_nnz)
    HYPRE_BoomerAMGSetILUMaxIter(s, ilu_max_iter)
    HYPRE_BoomerAMGSetILUDroptol(s, ilu_droptol)
    HYPRE_BoomerAMGSetILUTriSolve(s, ilu_tri_solve)
    HYPRE_BoomerAMGSetILULowerJacobiIters(s, ilu_lower_jacobi_iters)
    HYPRE_BoomerAMGSetILUUpperJacobiIters(s, ilu_upper_jacobi_iters)
    HYPRE_BoomerAMGSetILULocalReordering(s, ilu_local_reordering)

    # FSAI smoother (within AMG)
    HYPRE_BoomerAMGSetFSAIAlgoType(s, fsai_algo_type)
    HYPRE_BoomerAMGSetFSAILocalSolveType(s, fsai_local_solve_type)
    HYPRE_BoomerAMGSetFSAIMaxSteps(s, fsai_max_steps)
    HYPRE_BoomerAMGSetFSAIMaxStepSize(s, fsai_max_step_size)
    HYPRE_BoomerAMGSetFSAIMaxNnzRow(s, fsai_max_nnz_row)
    HYPRE_BoomerAMGSetFSAIThreshold(s, fsai_threshold)
    HYPRE_BoomerAMGSetFSAIEigMaxIters(s, fsai_eig_max_iters)
    HYPRE_BoomerAMGSetFSAIKapTolerance(s, fsai_kap_tolerance)

    # ParaSails smoother (within AMG)
    HYPRE_BoomerAMGSetSym(s, 1)
    HYPRE_BoomerAMGSetLevel(s, parasails_level)
    HYPRE_BoomerAMGSetThreshold(s, parasails_threshold)
    HYPRE_BoomerAMGSetFilter(s, parasails_filter)

    # Schwarz smoother (within AMG)
    HYPRE_BoomerAMGSetDomainType(s, schwarz_domain_type)
    HYPRE_BoomerAMGSetOverlap(s, schwarz_overlap)
    HYPRE_BoomerAMGSetVariant(s, schwarz_variant)
    HYPRE_BoomerAMGSetSchwarzRlxWeight(s, schwarz_rlx_weight)

    # GSMG
    HYPRE_BoomerAMGSetGSMG(s, gsmg)
    HYPRE_BoomerAMGSetNumSamples(s, num_samples)

    # Debug / output
    HYPRE_BoomerAMGSetPrintLevel(s, print_level)
    HYPRE_BoomerAMGSetLogging(s, logging)
    HYPRE_BoomerAMGSetDebugFlag(s, debug_flag)

    # Setup the hierarchy
    HYPRE_BoomerAMGSetup(s, parcsr_A, par_b, par_x)

    # Get setup costs (fixed per hierarchy)
    setup_flops_ref = Ref{HYPREReal}(0.0)
    setup_graph_ref = Ref{HYPREReal}(0.0)
    HYPRE_BoomerAMGGetSetupFlops(s, setup_flops_ref)
    HYPRE_BoomerAMGGetSetupGraphOps(s, setup_graph_ref)
    gen_fma   = round(UInt64, setup_flops_ref[])
    gen_graph = round(UInt64, setup_graph_ref[])
    nnz_A = nnz(A)

    # Pre-allocate buffers
    indices = collect(Cint(0):Cint(n - 1))

    h = HypreHandles(s, ij_A, ij_b, ij_x, parcsr_A, par_b, par_x)
    finalizer(h) do hh
        HYPRE_BoomerAMGDestroy(hh.solver)
        HYPRE_IJMatrixDestroy(hh.ij_A)
        HYPRE_IJVectorDestroy(hh.ij_b)
        HYPRE_IJVectorDestroy(hh.ij_x)
    end

    function configure(;
            cycle_type::Union{Symbol,Int} = :V,
            fcycle::Union{Symbol,Int,Bool} = false,
            num_sweeps::Int = 1,
            relax_order::Union{Symbol,Int,Bool} = false,
            relax_wt::Real = 1.0,
            outer_wt::Real = 1.0)

        _sleep_julia_workers!()
        # Resolve and set apply-time parameters
        ct = resolve_option(cycle_type, CYCLE_TYPES, "cycle_type")
        fc = resolve_bool(fcycle, "fcycle")
        ro = resolve_bool(relax_order, "relax_order")

        HYPRE_BoomerAMGSetCycleType(h.solver, ct)
        HYPRE_BoomerAMGSetFCycle(h.solver, fc)
        HYPRE_BoomerAMGSetNumSweeps(h.solver, num_sweeps)
        HYPRE_BoomerAMGSetRelaxOrder(h.solver, ro)
        HYPRE_BoomerAMGSetRelaxWt(h.solver, Float64(relax_wt))
        HYPRE_BoomerAMGSetOuterWt(h.solver, Float64(outer_wt))

        # Re-measure cycle cost with current apply-time settings
        apply_flops_ref = Ref{HYPREReal}(0.0)
        HYPRE_BoomerAMGGetApplyFlops(h.solver, apply_flops_ref)
        apply_fma = round(UInt64, apply_flops_ref[])
        cycle_cc = nnz_A > 0 ? apply_flops_ref[] / nnz_A : 0.0

        meta = Dict{String,Any}(
            "success" => true,
            "apply_fma" => apply_fma,
            "apply_cost_kind" => "fixed",
            "generation_fma" => gen_fma,
            "generation_graph" => gen_graph,
            "cycle_complexity" => cycle_cc,
        )

        function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
            _sleep_julia_workers!()
            HYPRE_IJVectorSetValues(h.ij_b, Cint(n), indices, r)
            HYPRE_IJVectorAssemble(h.ij_b)
            HYPRE_ParVectorSetConstantValues(h.par_x, 0.0)
            HYPRE_BoomerAMGSolve(h.solver, h.parcsr_A, h.par_b, h.par_x)
            HYPRE_IJVectorGetValues(h.ij_x, Cint(n), indices, z)
            return apply_fma
        end

        return Preconditioner(apply!, gen_fma, gen_graph), meta
    end

    return OptionedPreconditioner(configure)
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    HypreBoomerAMG(A; kwargs...) -> (Preconditioner, Dict)

Construct a BoomerAMG preconditioner (single cycle application).
Supports V, W, and F cycles. Every HYPRE BoomerAMG option is exposed as a keyword argument.

# Apply-time keyword arguments
These can also be varied without rebuilding when using the `OptionedPreconditioner` form:
- `cycle_type::Symbol = :V` — `:V`, `:W`
- `fcycle::Bool = false` — F-cycle (overrides cycle_type when true)
- `num_sweeps::Int = 1` — Smoother sweeps per level visit (V(k,k) for symmetry)
- `relax_order::Bool = false` — C/F ordering (true) vs natural ordering (false)
- `relax_wt::Float64 = 1.0` — Relaxation weight (damping)
- `outer_wt::Float64 = 1.0` — Outer relaxation weight
"""
function HypreBoomerAMG(A::SparseMatrixCSC{Float64,Int};
                        cycle_type::Union{Symbol,Int} = :V,
                        fcycle::Union{Symbol,Int,Bool} = false,
                        num_sweeps::Int = 1,
                        relax_order::Union{Symbol,Int,Bool} = false,
                        relax_wt::Real = 1.0,
                        outer_wt::Real = 1.0,
                        kwargs...)
    op = _build_boomeramg_hierarchy(A; cycle_type, fcycle, num_sweeps, relax_order,
                                    relax_wt, outer_wt, kwargs...)
    return op(; cycle_type, fcycle, num_sweeps, relax_order, relax_wt, outer_wt)
end

"""
    HypreBoomerAMG(A, OptionedPreconditioner; setup_kwargs...) -> OptionedPreconditioner

Build the AMG hierarchy once and return an `OptionedPreconditioner` that can be
reconfigured with different apply-time parameters (cycle_type, fcycle, num_sweeps,
relax_order, relax_wt, outer_wt) without rebuilding.

    op = HypreBoomerAMG(A, OptionedPreconditioner; coarsen_type=:HMIS)
    prec_v, meta_v = op(; cycle_type=:V)
    prec_w, meta_w = op(; cycle_type=:W)  # reuses same hierarchy
"""
function HypreBoomerAMG(A::SparseMatrixCSC{Float64,Int}, ::Type{OptionedPreconditioner};
                        kwargs...)
    return _build_boomeramg_hierarchy(A; kwargs...)
end

"""
    HypreFSAI(A; kwargs...) -> (Preconditioner, Dict)

Factorized Sparse Approximate Inverse preconditioner via Hypre.

The adaptive algorithm builds a lower triangular G ≈ L⁻¹ by iteratively growing the
sparsity pattern of each row using the Kaporin gradient. Quality and cost are controlled
by `max_steps`, `max_step_size`, and `kap_tolerance`. The Hypre docs recommend
`max_steps * max_step_size ≤ 30` for a reasonable cost/quality balance.

Ordering-dependent: the Kaporin gradient searches nonzero neighbors of each row,
so reordering A changes the candidate set and the resulting G.
"""
function HypreFSAI(A::SparseMatrixCSC{Float64,Int};
                   algo_type::Union{Symbol,Int} = :adaptive,
                   local_solve_type::Union{Symbol,Int} = :gauss_jordan,
                   max_steps::Int = 5,
                   max_step_size::Int = 3,
                   max_nnz_row::Int = 0,
                   threshold::Real = 0.0,
                   kap_tolerance::Real = 1.0e-3,
                   omega::Real = 1.0,
                   max_iterations::Int = 1,
                   eig_max_iters::Int = 5,
                   tolerance::Real = 0.0,
                   zero_guess::Union{Symbol,Int,Bool} = true,
                   print_level::Int = 0)

    # Resolve symbolic options
    algo_type = resolve_option(algo_type, FSAI_ALGO_TYPES, "algo_type")
    local_solve_type = resolve_option(local_solve_type, FSAI_SOLVE_TYPES, "local_solve_type")
    zero_guess = resolve_bool(zero_guess, "zero_guess")

    n = size(A, 1)
    ensure_initialized()

    ij_A, parcsr_A = julia_to_hypre_matrix(A)
    ij_b, par_b, ij_x, par_x = julia_to_hypre_vectors(n)

    solver_ref = Ref{HYPRESolver}(C_NULL)
    HYPRE_FSAICreate(solver_ref)
    s = solver_ref[]

    HYPRE_FSAISetAlgoType(s, algo_type)
    HYPRE_FSAISetLocalSolveType(s, local_solve_type)
    HYPRE_FSAISetMaxSteps(s, max_steps)
    HYPRE_FSAISetMaxStepSize(s, max_step_size)
    HYPRE_FSAISetMaxNnzRow(s, max_nnz_row)
    HYPRE_FSAISetThreshold(s, threshold)
    HYPRE_FSAISetKapTolerance(s, kap_tolerance)
    HYPRE_FSAISetOmega(s, omega)
    HYPRE_FSAISetMaxIterations(s, max_iterations)
    HYPRE_FSAISetEigMaxIters(s, eig_max_iters)
    HYPRE_FSAISetTolerance(s, tolerance)
    HYPRE_FSAISetZeroGuess(s, zero_guess)
    HYPRE_FSAISetPrintLevel(s, print_level)

    HYPRE_FSAISetup(s, parcsr_A, par_b, par_x)

    setup_flops_ref = Ref{HYPREReal}(0.0)
    setup_graph_ref = Ref{HYPREReal}(0.0)
    apply_flops_ref = Ref{HYPREReal}(0.0)
    HYPRE_FSAIGetSetupFlops(s, setup_flops_ref)
    HYPRE_FSAIGetSetupGraphOps(s, setup_graph_ref)
    HYPRE_FSAIGetApplyFlops(s, apply_flops_ref)

    apply_fma = round(UInt64, apply_flops_ref[])
    gen_fma   = round(UInt64, setup_flops_ref[])
    gen_graph = round(UInt64, setup_graph_ref[])

    # apply_flops = 2*nnz(G); compare G to tril(A)
    nnz_G = apply_flops_ref[] / 2.0
    nnz_tril_A = (nnz(A) - n) / 2 + n
    fill_ratio = nnz_tril_A > 0 ? nnz_G / nnz_tril_A : 0.0

    indices = collect(Cint(0):Cint(n - 1))

    h = HypreHandles(s, ij_A, ij_b, ij_x, parcsr_A, par_b, par_x)
    finalizer(h) do hh
        HYPRE_FSAIDestroy(hh.solver)
        HYPRE_IJMatrixDestroy(hh.ij_A)
        HYPRE_IJVectorDestroy(hh.ij_b)
        HYPRE_IJVectorDestroy(hh.ij_x)
    end

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        _sleep_julia_workers!()
        HYPRE_IJVectorSetValues(h.ij_b, Cint(n), indices, r)
        HYPRE_IJVectorAssemble(h.ij_b)
        HYPRE_ParVectorSetConstantValues(h.par_x, 0.0)
        HYPRE_FSAISolve(h.solver, h.parcsr_A, h.par_b, h.par_x)
        HYPRE_IJVectorGetValues(h.ij_x, Cint(n), indices, z)
        return apply_fma
    end

    prec = Preconditioner(apply!, gen_fma, gen_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => gen_fma,
        "generation_graph" => gen_graph,
        "fill_ratio" => fill_ratio,
    )
    return prec, meta
end

"""
    HypreCheby(A; kwargs...) -> Preconditioner

Construct a Chebyshev polynomial preconditioner via hypre.

Upstream clamps degrees above 4 to 4. This fork evaluates higher degrees on the
CPU using semi-iteration; degrees 1-4 retain the upstream implementation.

The polynomial interval `[lower, upper]` is derived from eigenvalue estimates:
`upper = 1.1 × max_eig` and `lower = (upper - min_eig) × eig_ratio + min_eig`.
So `eig_ratio` interpolates the lower bound between `min_eig` (at 0) and `upper` (at 1).

By default, hypre estimates eigenvalues internally. Pass `min_eig_est` and `max_eig_est`
to supply your own bounds (e.g., from `estimate_eigrange`), bypassing hypre's estimation.
`eig_ratio` is applied with either explicit or internally estimated bounds,
including zero to use the full supplied interval.

# Keyword arguments
- `order::Int = 2`: polynomial degree
- `variant::Symbol = :standard_residual`: Chebyshev variant (`:standard_residual` or `:alternative`)
- `scale::Bool = true`: enable diagonal scaling
- `eig_ratio::Float64 = 0.3`: interpolation fraction for lower polynomial bound
- `eig_est::Int = 10`: CG iterations for internal eigenvalue estimation (0 = Gershgorin)
- `min_eig_est::Float64 = nothing`: explicit λ_min (bypasses internal estimation)
- `max_eig_est::Float64 = nothing`: explicit λ_max (bypasses internal estimation)
"""
function HypreCheby(A::SparseMatrixCSC{Float64,Int};
                    order::Int = 2,
                    variant::Union{Symbol,Int} = :standard_residual,
                    scale::Union{Symbol,Int,Bool} = true,
                    eig_ratio::Real = 0.3,
                    eig_est::Int = 10,
                    min_eig_est::Union{Nothing,Real} = nothing,
                    max_eig_est::Union{Nothing,Real} = nothing,
                    max_iterations::Int = 1,
                    tolerance::Real = 0.0,
                    zero_guess::Union{Symbol,Int,Bool} = true,
                    print_level::Int = 0,
                    logging::Int = 0)

    order >= 1 || throw(ArgumentError("Chebyshev order must be positive"))
    isfinite(eig_ratio) && 0 <= eig_ratio <= 1 ||
        throw(ArgumentError("eig_ratio must be finite and in [0, 1]"))
    # Resolve symbolic options
    variant = resolve_option(variant, CHEBY_VARIANT_TYPES, "variant")
    scale = resolve_bool(scale, "scale")
    zero_guess = resolve_bool(zero_guess, "zero_guess")

    n = size(A, 1)
    ensure_initialized()

    ij_A, parcsr_A = julia_to_hypre_matrix(A)
    ij_b, par_b, ij_x, par_x = julia_to_hypre_vectors(n)

    solver_ref = Ref{HYPRESolver}(C_NULL)
    HYPRE_ParChebyCreate(solver_ref)
    s = solver_ref[]

    HYPRE_ParChebySetOrder(s, order)
    HYPRE_ParChebySetVariant(s, variant)
    HYPRE_ParChebySetScale(s, scale)
    HYPRE_ParChebySetEigRatio(s, eig_ratio)
    if !isnothing(min_eig_est) && !isnothing(max_eig_est)
        # Use explicit spectral bounds (bypasses hypre's internal estimation)
        HYPRE_ParChebySetMinMaxEigEst(s, min_eig_est, max_eig_est)
    else
        HYPRE_ParChebySetEigEst(s, eig_est)
    end
    HYPRE_ParChebySetMaxIterations(s, max_iterations)
    HYPRE_ParChebySetTolerance(s, tolerance)
    HYPRE_ParChebySetZeroGuess(s, zero_guess)
    HYPRE_ParChebySetPrintLevel(s, print_level)
    HYPRE_ParChebySetLogging(s, logging)

    HYPRE_ParChebySetup(s, parcsr_A, par_b, par_x)

    setup_flops_ref = Ref{HYPREReal}(0.0)
    setup_graph_ref = Ref{HYPREReal}(0.0)
    apply_flops_ref = Ref{HYPREReal}(0.0)
    HYPRE_ParChebyGetSetupFlops(s, setup_flops_ref)
    HYPRE_ParChebyGetSetupGraphOps(s, setup_graph_ref)
    HYPRE_ParChebyGetApplyFlops(s, apply_flops_ref)

    indices = collect(Cint(0):Cint(n - 1))

    h = HypreHandles(s, ij_A, ij_b, ij_x, parcsr_A, par_b, par_x)
    finalizer(h) do hh
        HYPRE_ParChebyDestroy(hh.solver)
        HYPRE_IJMatrixDestroy(hh.ij_A)
        HYPRE_IJVectorDestroy(hh.ij_b)
        HYPRE_IJVectorDestroy(hh.ij_x)
    end

    apply_fma = round(UInt64, apply_flops_ref[])

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        _sleep_julia_workers!()
        HYPRE_IJVectorSetValues(h.ij_b, Cint(n), indices, r)
        HYPRE_IJVectorAssemble(h.ij_b)
        HYPRE_ParVectorSetConstantValues(h.par_x, 0.0)
        HYPRE_ParChebySolve(h.solver, h.parcsr_A, h.par_b, h.par_x)
        HYPRE_IJVectorGetValues(h.ij_x, Cint(n), indices, z)
        return apply_fma
    end

    prec = Preconditioner(
        apply!,
        round(UInt64, setup_flops_ref[]),
        round(UInt64, setup_graph_ref[])
    )
    return prec, apply_fma
end

"""
    HypreParaSails(A; kwargs...) -> Preconditioner

Construct a ParaSails preconditioner (single application).
Uses symmetric mode (sym=1) by default for SPD problems.
"""
function HypreParaSails(A::SparseMatrixCSC{Float64,Int};
                        thresh::Real = 0.1,
                        nlevels::Int = 1,
                        filter::Real = 0.1,
                        loadbal::Real = 0.0,
                        reuse::Int = 0,
                        logging::Int = 0)

    n = size(A, 1)
    ensure_initialized()

    ij_A, parcsr_A = julia_to_hypre_matrix(A)
    ij_b, par_b, ij_x, par_x = julia_to_hypre_vectors(n)

    solver_ref = Ref{HYPRESolver}(C_NULL)
    HYPRE_ParaSailsCreate(HYPRE_MPI_COMM_WORLD, solver_ref)
    s = solver_ref[]

    HYPRE_ParaSailsSetParams(s, thresh, nlevels)
    HYPRE_ParaSailsSetFilter(s, filter)
    HYPRE_ParaSailsSetSym(s, 1)  # SPD (all our systems are symmetric)
    HYPRE_ParaSailsSetLoadbal(s, loadbal)
    HYPRE_ParaSailsSetReuse(s, reuse)
    HYPRE_ParaSailsSetLogging(s, logging)

    HYPRE_ParaSailsSetup(s, parcsr_A, par_b, par_x)

    setup_flops_ref = Ref{HYPREReal}(0.0)
    setup_graph_ref = Ref{HYPREReal}(0.0)
    apply_flops_ref = Ref{HYPREReal}(0.0)
    HYPRE_ParaSailsGetSetupFlops(s, setup_flops_ref)
    HYPRE_ParaSailsGetSetupGraphOps(s, setup_graph_ref)
    HYPRE_ParaSailsGetApplyFlops(s, apply_flops_ref)

    apply_fma = round(UInt64, apply_flops_ref[])
    gen_fma   = round(UInt64, setup_flops_ref[])
    gen_graph = round(UInt64, setup_graph_ref[])

    # apply_flops = 2*nnz(M) in symmetric mode; compare M to tril(A)
    nnz_M = apply_flops_ref[] / 2.0
    nnz_tril_A = (nnz(A) - n) / 2 + n
    fill_ratio = nnz_tril_A > 0 ? nnz_M / nnz_tril_A : 0.0

    indices = collect(Cint(0):Cint(n - 1))

    h = HypreHandles(s, ij_A, ij_b, ij_x, parcsr_A, par_b, par_x)
    finalizer(h) do hh
        HYPRE_ParaSailsDestroy(hh.solver)
        HYPRE_IJMatrixDestroy(hh.ij_A)
        HYPRE_IJVectorDestroy(hh.ij_b)
        HYPRE_IJVectorDestroy(hh.ij_x)
    end

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        _sleep_julia_workers!()
        HYPRE_IJVectorSetValues(h.ij_b, Cint(n), indices, r)
        HYPRE_IJVectorAssemble(h.ij_b)
        HYPRE_ParVectorSetConstantValues(h.par_x, 0.0)
        HYPRE_ParaSailsSolve(h.solver, h.parcsr_A, h.par_b, h.par_x)
        HYPRE_IJVectorGetValues(h.ij_x, Cint(n), indices, z)
        return apply_fma
    end

    prec = Preconditioner(apply!, gen_fma, gen_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => gen_fma,
        "generation_graph" => gen_graph,
        "fill_ratio" => fill_ratio,
    )
    return prec, meta
end

# ---------------------------------------------------------------------------
# Schwarz
# ---------------------------------------------------------------------------
function HypreSchwarz(A::SparseMatrixCSC{Float64,Int};
                      variant::Union{Symbol,Int} = :multiplicative,
                      overlap::Int = 1,
                      domain_type::Union{Symbol,Int} = :agglomerate,
                      num_functions::Int = 1,
                      relax_weight::Union{Real,Symbol} = 1.0)

    relax_weight isa Real && (relax_weight = Float64(relax_weight))

    # Resolve symbolic options
    variant = resolve_option(variant, SCHWARZ_VARIANT_TYPES, "variant")
    domain_type = resolve_option(domain_type, SCHWARZ_DOMAIN_TYPES, "domain_type")

    n = size(A, 1)
    ensure_initialized()

    ij_A, parcsr_A = julia_to_hypre_matrix(A)
    ij_b, par_b, ij_x, par_x = julia_to_hypre_vectors(n)

    solver_ref = Ref{HYPRESolver}(C_NULL)
    HYPRE_SchwarzCreate(solver_ref)
    s = solver_ref[]

    HYPRE_SchwarzSetVariant(s, variant)
    HYPRE_SchwarzSetOverlap(s, overlap)
    HYPRE_SchwarzSetDomainType(s, domain_type)
    HYPRE_SchwarzSetNumFunctions(s, num_functions)
    # relax_weight is set after setup if :inverse_overlap is requested
    if relax_weight isa Float64
        HYPRE_SchwarzSetRelaxWeight(s, relax_weight)
    end

    HYPRE_SchwarzSetup(s, parcsr_A, par_b, par_x)

    setup_flops_ref = Ref{HYPREReal}(0.0)
    setup_graph_ref = Ref{HYPREReal}(0.0)
    apply_flops_ref = Ref{HYPREReal}(0.0)
    HYPRE_SchwarzGetSetupFlops(s, setup_flops_ref)
    HYPRE_SchwarzGetSetupGraphOps(s, setup_graph_ref)
    HYPRE_SchwarzGetApplyFlops(s, apply_flops_ref)

    max_overlap_ref = Ref{HYPREInt}(Cint(0))
    HYPRE_SchwarzGetMaxDomainOverlap(s, max_overlap_ref)
    max_domain_overlap = Int(max_overlap_ref[])

    # Resolve :inverse_overlap now that we know max_domain_overlap
    if relax_weight === :inverse_overlap
        max_domain_overlap > 0 || error("max_domain_overlap is 0, cannot compute inverse")
        relax_weight = 1.0 / max_domain_overlap
        HYPRE_SchwarzSetRelaxWeight(s, relax_weight)
    elseif relax_weight isa Symbol
        error("Unknown symbolic relax_weight: :$relax_weight. Valid: :inverse_overlap")
    end

    apply_fma = round(UInt64, apply_flops_ref[])
    gen_fma   = round(UInt64, setup_flops_ref[])
    gen_graph = round(UInt64, setup_graph_ref[])

    indices = collect(Cint(0):Cint(n - 1))

    h = HypreHandles(s, ij_A, ij_b, ij_x, parcsr_A, par_b, par_x)
    finalizer(h) do hh
        HYPRE_SchwarzDestroy(hh.solver)
        HYPRE_IJMatrixDestroy(hh.ij_A)
        HYPRE_IJVectorDestroy(hh.ij_b)
        HYPRE_IJVectorDestroy(hh.ij_x)
    end

    function apply!(z::Vector{Float64}, r::Vector{Float64})::Float64
        _sleep_julia_workers!()
        HYPRE_IJVectorSetValues(h.ij_b, Cint(n), indices, r)
        HYPRE_IJVectorAssemble(h.ij_b)
        HYPRE_ParVectorSetConstantValues(h.par_x, 0.0)
        HYPRE_SchwarzSolve(h.solver, h.parcsr_A, h.par_b, h.par_x)
        HYPRE_IJVectorGetValues(h.ij_x, Cint(n), indices, z)
        return apply_fma
    end

    prec = Preconditioner(apply!, gen_fma, gen_graph)
    meta = Dict{String,Any}(
        "success" => true,
        "apply_fma" => apply_fma,
        "apply_cost_kind" => "fixed",
        "generation_fma" => gen_fma,
        "generation_graph" => gen_graph,
        "max_domain_overlap" => max_domain_overlap,
        "relax_weight" => relax_weight,
    )
    return prec, meta
end
