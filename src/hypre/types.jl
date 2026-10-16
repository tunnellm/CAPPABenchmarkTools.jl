# Type mappings for sequential HYPRE build (no BIGINT, no SINGLE, no COMPLEX)
const HYPREInt = Cint
const HYPREBigInt = Cint
const HYPREReal = Cdouble
const HYPREComplex = Cdouble

# Opaque handle types
const HYPRESolver = Ptr{Cvoid}
const HYPREIJMatrix = Ptr{Cvoid}
const HYPREIJVector = Ptr{Cvoid}
const HYPREParCSRMatrix = Ptr{Cvoid}
const HYPREParVector = Ptr{Cvoid}

# Sequential MPI stubs (HYPRE_SEQUENTIAL=1)
const HYPREMPIComm = Cint
const HYPRE_MPI_COMM_WORLD = Cint(0)

# Object type constant for ParCSR format
const HYPRE_PARCSR = Cint(5555)

# ---------------------------------------------------------------------------
# Named-symbol mappings for enum-like integer parameters
# ---------------------------------------------------------------------------

"""
    resolve_option(val, mapping, param_name) -> Int

Resolve a symbolic or integer option to its integer value.
Passes through `Int` values unchanged; looks up `Symbol` values in `mapping`.
"""
function resolve_option(val::Union{Symbol,Int}, mapping::Dict{Symbol,Int}, param_name::String)::Int
    val isa Int && return val
    haskey(mapping, val) || error("Unknown $param_name: :$val. Valid: $(join(keys(mapping), ", "))")
    return mapping[val]
end

"""
    resolve_bool(val, param_name) -> Int

Resolve a boolean-like parameter that accepts `Bool`, `Int`, or `:on`/`:off`.
"""
function resolve_bool(val::Union{Symbol,Int,Bool}, param_name::String)::Int
    val isa Bool && return Int(val)
    val isa Int && return val
    return resolve_option(val, Dict{Symbol,Int}(:off => 0, :on => 1), param_name)
end

# ---- BoomerAMG mappings ----

const COARSEN_TYPES = Dict{Symbol,Int}(
    :CLJP => 0, :ruge_stueben => 1, :falgout => 3,
    :PMIS => 8, :HMIS => 10, :CGC => 21, :CGC_E => 22,
)

const MEASURE_TYPES = Dict{Symbol,Int}(
    :local => 0, :global => 1,
)

const INTERP_TYPES = Dict{Symbol,Int}(
    :classical => 0, :direct => 3, :multipass => 4,
    :ext_plus_i => 6, :ext_plus_i_cc => 7, :standard => 8,
    :FF => 12, :FF1 => 13, :extended => 14,
    :ext_mm => 16, :ext_plus_i_mm => 17, :ext_plus_e_mm => 18,
)

const AGG_INTERP_TYPES = Dict{Symbol,Int}(
    :two_stage_ext_plus_i => 1, :two_stage_standard => 2,
    :two_stage_extended => 3, :multipass => 4,
    :two_stage_ext_mm => 5, :two_stage_ext_plus_i_mm => 6,
    :two_stage_ext_plus_e_mm => 7, :multipass_mm => 8,
)

const CYCLE_TYPES = Dict{Symbol,Int}(
    :V => 1, :W => 2,
)

const RELAX_TYPES = Dict{Symbol,Int}(
    :jacobi => 0,
    :gauss_seidel_forward => 3, :gauss_seidel_backward => 4,
    :symmetric_gauss_seidel => 6,
    :l1_symmetric_gauss_seidel => 8, :direct => 9,
    :l1_gauss_seidel_forward => 13, :l1_gauss_seidel_backward => 14,
    :CG => 15, :chebyshev => 16, :FCF_jacobi => 17, :l1_jacobi => 18,
)

const SMOOTH_TYPES = Dict{Symbol,Int}(
    :FSAI => 4, :ILU => 5, :schwarz => 6,
    :parasails => 8,
)

const CONVERGE_TYPES = Dict{Symbol,Int}(
    :rhs => 0, :initial => 1,
)

const RESTRICTION_TYPES = Dict{Symbol,Int}(
    :transpose => 0, :AIR_1 => 1, :AIR_2 => 2,
)

const ILU_TYPES = Dict{Symbol,Int}(
    :BJ_ILUk => 0, :BJ_ILUT => 1,
    :GMRES_ILUk => 10, :GMRES_ILUT => 11,
    :NSH_ILUk => 20, :NSH_ILUT => 21,
    :RAS_ILUk => 30, :RAS_ILUT => 31,
)

const ILU_REORDERING_TYPES = Dict{Symbol,Int}(
    :none => 0, :RCM => 1,
)

const NODAL_TYPES = Dict{Symbol,Int}(
    :unknown => 0, :frobenius => 1, :sum_abs => 2,
    :largest => 3, :row_sum => 4, :sum_all => 6,
)

const A_DROP_TYPES = Dict{Symbol,Int}(
    :none => 0, :inf_norm => -1, :one_norm => 1, :two_norm => 2,
)

# ---- FSAI mappings ----

# :adaptive and :adaptive_dynamic use the same algorithm (adaptive Kaporin gradient).
# The only difference is OpenMP scheduling: :adaptive uses static partitioning,
# :adaptive_dynamic uses omp schedule(dynamic). Identical when single-threaded.
const FSAI_ALGO_TYPES = Dict{Symbol,Int}(
    :adaptive => 1, :adaptive_dynamic => 2,
)

const FSAI_SOLVE_TYPES = Dict{Symbol,Int}(
    :gauss_jordan => 0, :vendor => 1, :magma => 2,
)

# ---- Schwarz mappings ----

const CHEBY_VARIANT_TYPES = Dict{Symbol,Int}(
    :standard_residual => 0, :alternative => 1,
)

const SCHWARZ_VARIANT_TYPES = Dict{Symbol,Int}(
    :multiplicative => 0, :additive => 1,
    :parallel_additive => 2, :parallel_multiplicative => 3,
    :forward_multiplicative => 4,
)

const SCHWARZ_DOMAIN_TYPES = Dict{Symbol,Int}(
    :point => 0, :node => 1, :agglomerate => 2,
)
