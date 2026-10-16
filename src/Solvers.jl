module Solvers

using SparseArrays
using LinearAlgebra

export pcg, lanczos_pcg, minres, pcr, fpcg1, fpcg2
export write_npy_zstd, read_npy_zstd, open_diagnostic_data

using ..Utilities: compute_tolerances, write_npy_zstd_1d_f64,
    NOOP_PHASE_CALLBACK, run_scoped_phase, NumericalProgress, progress_due!, emit_numerical_progress!,
    begin_reported_phase, finish_reported_phase, phase_reporting_enabled, run_phase
using ..PreconditionerTypes: Preconditioner
using ..ThreadedUtilities: threaded_spmv!, threaded_norm, threaded_dot, _norm_from_squared, _spmv_plan,
    FusedReductionWorkspace, _fused_diagnostics!, _fused_update_norm!

# File I/O
using CodecZstd, TranscodingStreams
using JSON3
include("diagnostic_buffers.jl")
include("diagnostic_storage.jl")
include("coefficient_storage.jl")

const tolerances = [
    10^-1, 10^-1.5,
    10^-2, 10^-2.5,
    10^-3, 10^-3.5,
    10^-4, 10^-4.5,
    10^-5, 10^-5.5,
    10^-6, 10^-6.5,
    10^-7, 10^-7.5,
    10^-8, 10^-8.5,
    10^-9, 10^-9.5,
    10^-10
    ]
const minimum_tolerance = minimum(tolerances)
# Every solver uses the requested convergence tolerance as the effective
# backward-error gate. sqrt(eps) is a ceiling for unusually loose solver
# tolerances, not an additional benchmark-specific accuracy target.
const correction_restart_normwise_tolerance_default = sqrt(eps(Float64))

"""
    SolverStatus

Exit status for iterative solvers.
"""
@enum SolverStatus begin
    CONVERGED        # All tolerance thresholds met
    MAX_ITERS        # Reached maximum iteration count
    WORK_LIMIT       # Exceeded computational work budget
    NAN_DETECTED     # NaN in scalar recurrence
    BREAKDOWN        # Lanczos breakdown (non-positive β²)
    STAGNATED        # Legacy status retained for existing result metadata
    KILLED           # Externally interrupted (signal or kill flag)
    CONTINUATION_LIMIT # Exhausted work allowed after the first native crossing
    SKIPPED_COST_GUARD # No solve attempted: one application exceeds the entire work budget
end

include("solver_policy.jl")

export SolverStatus, CONVERGED, MAX_ITERS, WORK_LIMIT, NAN_DETECTED, BREAKDOWN, STAGNATED, KILLED, CONTINUATION_LIMIT
export SKIPPED_COST_GUARD

"""
    KILL_FLAG

Module-level atomic flag for graceful solver termination. Set to `true` from a signal
handler or another thread to stop the running solver at the next iteration boundary.
Reset automatically at solver entry.
"""
const KILL_FLAG = Threads.Atomic{Bool}(false)

export KILL_FLAG

"""
    SolverResult

Return value from iterative solvers.

# Fields
- `status::SolverStatus`: Exit condition.
- `solver_policy::Dict{String,Any}`: Mode, first-crossing work, continuation cap,
  overall work limit, and terminal reason. Empty for legacy constructors.
- `iterations::Int`: Number of iterations performed.
- `wall_time_seconds::Float64`: Wall clock time for the solver (excludes I/O).
- `total_fma::Float64`: Cumulative FMA at termination (base + prec + vector, excludes generation).
- `final_relative_residual::Float64`: Relative residual at termination.
- `residual_replacements::Int`: Number of method-native residual-vector
  repairs. This is retained for legacy metadata; the benchmark certification
  policy now uses correction restarts instead.
- `residual_replacement_records::Vector{Dict{String,Any}}`: Legacy artifact and
  residual provenance for residual-vector repairs.
- `correction_restarts::Int`: Number of correction solves started from the
  explicitly recomputed global residual after failed certification.
- `correction_restart_records::Vector{Dict{String,Any}}`: Artifact and
  trigger provenance for scalar correction restarts.
- `physical_true_residual_evals::Int`: Number of true-residual evaluations
  physically computed by the benchmark harness for this solve.
- `diagnostic_true_residual_evals::Int`: Number of physical true-residual
  evaluations that remained diagnostic-only.
- `certification_true_residual_evals::Int`: Number of true-residual checks made
  visible to the declared solver policy and charged algorithmically.
- `certification_check_iters::Vector{Int}`: Iterations where true residual was
  load-bearing for certification.
- `true_residual_used_for_decision_iters::Vector{Int}`: Iterations where true
  residual influenced stopping, correction restart, or continuation policy.
- `false_native_crossings::Int`: Native residual tolerance crossings whose
  charged true-residual certification failed.
- `native_candidate_iter::Union{Nothing,Int}`: First native candidate iteration.
- `native_candidate_true_relres::Union{Nothing,Float64}`: True relative
  residual at the first native candidate.
- `native_candidate_normwise::Union{Nothing,Float64}`: Normwise metric at the
  first native candidate.
- `certification_records::Vector{Dict{String,Any}}`: Provenance for charged
  certification checks and false native crossings. Exact zero-RHS acceptance
  is recorded at iteration zero without charging a residual evaluation.
"""
struct SolverResult
    status::SolverStatus
    iterations::Int
    wall_time_seconds::Float64
    total_fma::Float64
    final_relative_residual::Float64
    residual_replacements::Int
    residual_replacement_records::Vector{Dict{String,Any}}
    correction_restarts::Int
    correction_restart_records::Vector{Dict{String,Any}}
    physical_true_residual_evals::Int
    diagnostic_true_residual_evals::Int
    certification_true_residual_evals::Int
    certification_check_iters::Vector{Int}
    true_residual_used_for_decision_iters::Vector{Int}
    false_native_crossings::Int
    native_candidate_iter::Union{Nothing,Int}
    native_candidate_true_relres::Union{Nothing,Float64}
    native_candidate_normwise::Union{Nothing,Float64}
    certification_records::Vector{Dict{String,Any}}
    solver_policy::Dict{String,Any}
end

# Preserve callers and metadata readers that predate solver policy fields.
SolverResult(
    status::SolverStatus,
    iterations::Int,
    wall_time_seconds::Float64,
    total_fma::Float64,
    final_relative_residual::Float64,
    residual_replacements::Int,
    residual_replacement_records::Vector{Dict{String,Any}},
    correction_restarts::Int,
    correction_restart_records::Vector{Dict{String,Any}},
    physical_true_residual_evals::Int,
    diagnostic_true_residual_evals::Int,
    certification_true_residual_evals::Int,
    certification_check_iters::Vector{Int},
    true_residual_used_for_decision_iters::Vector{Int},
    false_native_crossings::Int,
    native_candidate_iter::Union{Nothing,Int},
    native_candidate_true_relres::Union{Nothing,Float64},
    native_candidate_normwise::Union{Nothing,Float64},
    certification_records::Vector{Dict{String,Any}},
) = SolverResult(
    status, iterations, wall_time_seconds, total_fma, final_relative_residual,
    residual_replacements, residual_replacement_records, correction_restarts,
    correction_restart_records, physical_true_residual_evals, diagnostic_true_residual_evals,
    certification_true_residual_evals, certification_check_iters, true_residual_used_for_decision_iters,
    false_native_crossings, native_candidate_iter, native_candidate_true_relres,
    native_candidate_normwise, certification_records, Dict{String,Any}())

function _default_certification_fields()
    return (
        0,
        Dict{String,Any}[],
        0,
        0,
        0,
        Int[],
        Int[],
        0,
        nothing,
        nothing,
        nothing,
        Dict{String,Any}[],
    )
end

SolverResult(
    status::SolverStatus,
    iterations::Int,
    wall_time_seconds::Float64,
    total_fma::Float64,
    final_relative_residual::Float64,
) = SolverResult(status, iterations, wall_time_seconds, total_fma, final_relative_residual, 0)

SolverResult(
    status::SolverStatus,
    iterations::Int,
    wall_time_seconds::Float64,
    total_fma::Float64,
    final_relative_residual::Float64,
    residual_replacements::Int,
) = SolverResult(
    status,
    iterations,
    wall_time_seconds,
    total_fma,
    final_relative_residual,
    residual_replacements,
    Dict{String,Any}[],
)

SolverResult(
    status::SolverStatus,
    iterations::Int,
    wall_time_seconds::Float64,
    total_fma::Float64,
    final_relative_residual::Float64,
    residual_replacements::Int,
    residual_replacement_records::Vector{Dict{String,Any}},
) = SolverResult(
    status,
    iterations,
    wall_time_seconds,
    total_fma,
    final_relative_residual,
    residual_replacements,
    residual_replacement_records,
    _default_certification_fields()...,
)

export SolverResult

include("application_cost_guard.jl")

# Column indices for the per-iteration data matrix
const PCG_COL_ITER                = 1
const PCG_COL_RELATIVE_RESIDUAL   = 2
const PCG_COL_RELATIVE_NORMWISE_2 = 3
const PCG_COL_ERROR               = 4
const PCG_COL_CUM_BASE_FMA        = 5
const PCG_COL_CUM_PREC_FMA        = 6
const PCG_COL_CUM_VECTOR_FMA      = 7
const PCG_COL_MU                  = 8
const PCG_COL_NORM_R              = 9
const PCG_COL_NORM_R_TRUE         = 10
const PCG_COL_RZ                  = 11
const PCG_COL_PAP                 = 12
const PCG_COL_ENERGY              = 13
const PCG_COL_ORTHO_LOSS          = 14
const PCG_COL_DIAG_ABS_FLAG       = 15
const PCG_COL_KRYLOV_BLOCK       = 16
const PCG_NUM_COLS               = 16

# Per-iteration vector FMA for PCG:
#   threaded_dot(r, z, reduction_workspace) (dot): n
#   p .= p .* τ .+ z (axpy): n  [skipped iter 1]
#   threaded_dot(p, prod, reduction_workspace) (dot): n
#   x .+= μ .* p (axpy): n
#   r .-= μ .* prod (axpy): n
# Total: 5n per iteration (4n at iteration 1, 5n thereafter)
# Setup: 0 (preconditioner apply counted separately)
# A load-bearing true-residual certification charges vector work for forming
# b - A*x, computing ||b - A*x||, and computing ||x|| for the normwise metric.
# A correction restart adds one vector operation for r = r_true before
# starting the correction equation.
# Orthogonality loss (z'*r_prev dot + r_prev copy) and tridiagonal recovery are
# diagnostic overhead, NOT counted in vector FMA.
const TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA = 3
const PCG_VECTOR_FMA_PER_ITER = 5
const PCG_VECTOR_FMA_SETUP = 0
const PCG_VECTOR_FMA_RESIDUAL_REPLACEMENT = 1

const PCG_COLUMN_NAMES = [
    "iteration",
    "relative_residual",
    "relative_normwise_2",
    "relative_error",
    "cumulative_base_fma",
    "cumulative_prec_fma",
    "cumulative_vector_fma",
    "mu",
    "norm_r",
    "norm_r_true",
    "rz",
    "pAp",
    "energy",
    "orthogonality_loss",
    "diag_abs_flag",
    "krylov_block",
]

"""
    write_npy_zstd(path, data)

Write a Float64 matrix to a zstd-compressed .npy file.
The .npy format: magic \\x93NUMPY, version 1.0, header length (LE uint16),
header dict string (padded to 64-byte alignment), then raw data bytes.
Julia stores matrices column-major (Fortran order).
"""
function write_npy_zstd(path::String, data::AbstractMatrix{Float64})
    nrows, ncols = size(data)
    # Build header dict — Fortran order since Julia is column-major
    header_dict = "{'descr': '<f8', 'fortran_order': True, 'shape': ($nrows, $ncols), }"
    # Pad header to 64-byte alignment (magic=6 + version=2 + header_len=2 = 10 prefix bytes)
    prefix_len = 10
    pad_len = 64 - mod(prefix_len + length(header_dict) + 1, 64)  # +1 for trailing newline
    if pad_len == 64
        pad_len = 0
    end
    padded_header = header_dict * " "^pad_len * "\n"
    header_len = length(padded_header)

    open(path, "w") do f
        zstd = ZstdCompressorStream(f, level = 1)
        # Magic and version
        write(zstd, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])  # \x93NUMPY
        write(zstd, UInt8(1))   # major version
        write(zstd, UInt8(0))   # minor version
        write(zstd, UInt16(header_len))  # header length (little-endian on LE systems)
        write(zstd, padded_header)
        # Raw data — contiguous Float64 bytes in column-major order
        write(zstd, reinterpret(UInt8, vec(data)))
        close(zstd)
    end
end

"""
    _write_solution_vec(path, x::Vector{Float64})

Write a solver's converged solution as a 1-D NPY (Float64) inside zstd.
Loaded back via `Utilities.read_npy_zstd_1d_f64` for use as a series seed.
"""
function _write_solution_vec(path::String, x::Vector{Float64})
    write_npy_zstd_1d_f64(path, x)
    return path
end

"""
    _read_npy_zstd_file(path) -> Matrix{Float64}

Read a zstd-compressed .npy file containing a Float64 matrix.
"""
function _read_npy_zstd_file(path::String; expected_size=nothing)
    open(path, "r") do f
        stream = ZstdDecompressorStream(f)
        try
            read(stream, 6) == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] || error("not an NPY file: $path")
            version = (read(stream, UInt8), read(stream, UInt8))
            header_len = version == (1, 0) ? Int(ltoh(read(stream, UInt16))) :
                version == (2, 0) ? Int(ltoh(read(stream, UInt32))) : error("unsupported NPY version: $version")
            0 < header_len <= 1_048_576 || error("invalid NPY header length: $path")
            header = String(read(stream, header_len))
            occursin(r"'descr':\s*'<f8'", header) || error("expected Float64 NPY: $path")
            occursin(r"'fortran_order':\s*True", header) || error("expected column-major NPY: $path")
            shape = match(r"'shape':\s*\((\d+),\s*(\d+)\)", header)
            shape === nothing && error("expected matrix NPY shape: $path")
            dims = (parse(Int, shape[1]), parse(Int, shape[2]))
            expected_size === nothing || dims == expected_size || error("diagnostic block shape mismatch: $path")
            data = Matrix{Float64}(undef, dims)
            read!(stream, reinterpret(UInt8, vec(data)))
            eof(stream) || error("unexpected trailing NPY data: $path")
            return data
        finally
            close(stream)
        end
    end
end

function _restart_solution_path(raw_data_out::String, replacement_index::Int)
    return replace(raw_data_out, r"\.[^/]*$" => "_restart_r$(replacement_index)_solution.npy.zstd")
end

function _correction_restart_solution_path(raw_data_out::String, restart_index::Int)
    return replace(raw_data_out, r"\.[^/]*$" => "_correction_r$(restart_index)_base_solution.npy.zstd")
end

function _record_residual_replacement!(
    records::Vector{Dict{String,Any}},
    raw_data_out::String,
    replacement_index::Int,
    iteration::Int,
    recursive_relative_residual::Float64,
    true_relative_residual::Float64,
    true_relative_normwise_2::Float64,
    norm_r::Float64,
    norm_r_true::Float64,
    x::Vector{Float64};
    closed_tridiag_path = nothing,
    next_tridiag_path = _tridiag_block_path(raw_data_out, replacement_index),
)
    restart_solution = _restart_solution_path(raw_data_out, replacement_index)
    _write_solution_vec(restart_solution, x)
    push!(records, Dict{String,Any}(
        "replacement_index" => replacement_index,
        "after_iteration" => iteration,
        "recursive_relative_residual" => recursive_relative_residual,
        "true_relative_residual" => true_relative_residual,
        "true_relative_normwise_2" => true_relative_normwise_2,
        "norm_r" => norm_r,
        "norm_r_true" => norm_r_true,
        "restart_solution_path" => restart_solution,
        "closed_tridiag_path" => closed_tridiag_path,
        "next_tridiag_path" => next_tridiag_path,
    ))
end

function _record_correction_restart!(
    records::Vector{Dict{String,Any}},
    raw_data_out::String,
    restart_index::Int,
    iteration::Int,
    native_relative_residual::Float64,
    true_relative_residual::Float64,
    true_relative_normwise_2::Float64,
    native_norm::Float64,
    norm_r_true::Float64,
    x::Vector{Float64};
    closed_tridiag_path = nothing,
    next_tridiag_path = _tridiag_block_path(raw_data_out, restart_index),
)
    base_solution = _correction_restart_solution_path(raw_data_out, restart_index)
    _write_solution_vec(base_solution, x)
    push!(records, Dict{String,Any}(
        "restart_index" => restart_index,
        "kind" => "global_correction_restart",
        "solution_accumulation" => "separate_correction",
        "candidate_assembly_fma_per_iteration" => length(x),
        "after_iteration" => iteration,
        "native_relative_residual" => native_relative_residual,
        "true_relative_residual" => true_relative_residual,
        "true_relative_normwise_2" => true_relative_normwise_2,
        "native_norm" => native_norm,
        "norm_r_true" => norm_r_true,
        "correction_rhs_norm" => norm_r_true,
        "base_solution_path" => base_solution,
        "closed_tridiag_path" => closed_tridiag_path,
        "next_tridiag_path" => next_tridiag_path,
    ))
end

function _record_certification_check!(
    records::Vector{Dict{String,Any}},
    iteration::Int,
    event::String,
    native_relative_residual::Float64,
    true_relative_residual::Float64,
    true_relative_normwise_2::Float64,
    native_norm::Float64,
    norm_r_true::Float64;
    certified::Bool,
    action::String,
    source::String = "diagnostic_cache",
    charged::Bool = true,
)
    push!(records, Dict{String,Any}(
        "iter" => iteration,
        "event" => event,
        "eta" => native_relative_residual,
        "true_relres" => true_relative_residual,
        "relative_normwise_2" => true_relative_normwise_2,
        "native_norm" => native_norm,
        "norm_r_true" => norm_r_true,
        "true_residual_role" => "certification",
        "certification_charged" => charged,
        "certification_check_source" => source,
        "certified" => certified,
        "action" => action,
    ))
end

function _clear_unwritten_next_tridiag!(records::Vector{Dict{String,Any}}, final_path;
                                       key="next_tridiag_path")
    if !isempty(records) && final_path === nothing
        last(records)[key] = nothing
    end
    for record in records
        next_path = get(record, key, nothing)
        if next_path isa String && !isfile(next_path)
            record[key] = nothing
        end
    end
    return records
end

"""
    pcg(A, b, seed, spectral_norm, preconditioner;
        maxiters, work_limit=-1, raw_data_out, convergence_data_out)

Solves `Ax = b` using Preconditioned Conjugate Gradient.

Iteration data is collected in a geometrically growing matrix, then written to disk
as a zstd-compressed .npy file after termination. A companion JSON file
with column names is written alongside it. Convergence threshold crossings
are extracted post-hoc into CSV files.

# Arguments
- `A::SparseMatrixCSC`: The system matrix (SPD).
- `b::Vector`: Right-hand side vector.
- `seed::Vector`: Known solution for error tracking.
- `spectral_norm::Float64`: Spectral norm of `A`, used for normwise backward error.
- `preconditioner::Preconditioner`: The preconditioner to apply.
- `maxiters::Int64`: Maximum number of iterations.
- `work_limit::Int64`: Maximum cumulative FMA before termination. -1 for no limit.
- `convergence_tolerance::Float64`: Certified true-residual target.
- `correction_restart_normwise_tolerance::Float64`: A failed native
  certification starts one correction solve only when
  `relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)`.
- `raw_data_out::String`: Logical .npy.zstd path; v2 blocks are written to its sibling directory.
- `convergence_data_out::String`: Path prefix for convergence threshold CSVs.

# Returns
- `SolverResult`: Contains `status::SolverStatus` and `iterations::Int`.
"""
function _pcg_with_progress(
    A::SparseMatrixCSC,
    b::Vector{Float64},
    seed::Vector{Float64},
    spectral_norm::Float64,
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
    convergence_tolerance::Float64 = minimum_tolerance,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    write_solution::Union{Nothing,String} = nothing,
    _experimental_final_iterate::Union{Nothing,Vector{Float64}} = nothing,
    _diagnostic_block_rows::Int = DIAGNOSTIC_BLOCK_ROWS,
    progress = NumericalProgress(NOOP_PHASE_CALLBACK),
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    n = size(A, 1)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    spmv_plan = _spmv_plan(A)
    reduction_workspace = FusedReductionWorkspace()
    base_fma = UInt64(nnz(A))

    norm_b = threaded_norm(b)
    norm_seed = threaded_norm(seed)

    # Reuse a fixed raw-history buffer; row 1 is iteration 0.
    data = DiagnosticBlockWriter(raw_data_out, maxiters + 1, PCG_COLUMN_NAMES;
        block_rows=_diagnostic_block_rows, callback=progress.callback)

    # Working vectors
    x = zeros(n)
    # Before recovery, updates target x directly. Allocate correction storage lazily.
    correction_base = nothing
    update_x = x
    correction_assembly_steps = 0
    r = copy(b)
    z = zeros(n)
    Ax = zeros(n)
    r_true = similar(b)
    r_prev = similar(b)  # for orthogonality loss (diagnostic only)

    # Tridiagonal recovery from CG scalars (diagnostic only)
    μ_prev = 0.0

    # Initial preconditioner application
    cum_prec = Float64(preconditioner.LinearOperator(z, r))
    p = copy(z)

    # Initial direction: Ap (symmetric A, CSC row-parallel)
    prod = zeros(n)
    threaded_spmv!(prod, A, p, spmv_plan)
    pAp = threaded_dot(p, prod, reduction_workspace)
    rz = threaded_dot(r, z, reduction_workspace)
    μ = rz / pAp

    # Record iteration 0
    norm_r = threaded_norm(r)
    data[1, PCG_COL_ITER]                = 0.0
    data[1, PCG_COL_RELATIVE_RESIDUAL]   = norm_r / norm_b
    data[1, PCG_COL_RELATIVE_NORMWISE_2] = norm_r / (norm_b + spectral_norm * threaded_norm(x))
    data[1, PCG_COL_ERROR]               = 1.0
    data[1, PCG_COL_CUM_BASE_FMA]        = 0.0
    data[1, PCG_COL_CUM_PREC_FMA]        = cum_prec
    data[1, PCG_COL_CUM_VECTOR_FMA]      = Float64(PCG_VECTOR_FMA_SETUP) * n
    data[1, PCG_COL_MU]                  = μ
    data[1, PCG_COL_NORM_R]              = norm_r
    data[1, PCG_COL_NORM_R_TRUE]         = norm_r  # at iteration 0, r = b - A*0 = b = r_true
    data[1, PCG_COL_RZ]                  = rz
    data[1, PCG_COL_PAP]                 = pAp
    data[1, PCG_COL_ENERGY]              = 0.0  # ½x'Ax - x'b = 0 at x=0
    data[1, PCG_COL_ORTHO_LOSS]          = 0.0  # no previous Lanczos vector at iteration 0
    data[1, PCG_COL_DIAG_ABS_FLAG]       = 0.0

    data[1, PCG_COL_KRYLOV_BLOCK] = 0.0

    iter = 0
    nrows = 1  # row 1 (iteration 0) is already filled
    status = MAX_ITERS
    final_relative_residual = norm_r / norm_b
    cum_base = 0.0
    cum_vector = Float64(PCG_VECTOR_FMA_SETUP) * n
    residual_replacements = 0
    residual_replacement_records = Dict{String,Any}[]
    correction_restarts = 0
    correction_restart_records = Dict{String,Any}[]
    local_iter = 0
    tridiag_block_index = 0
    tridiag_stream = CoefficientStream(raw_data_out, maxiters, _diagnostic_block_rows;
        callback=progress.callback, source_attempt=basename(data.attempt))
    physical_true_residual_evals = 0
    certification_true_residual_evals = 0
    certification_check_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    true_residual_used_for_decision_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    certification_check_count = 0
    false_native_crossings = 0
    native_candidate_iter = nothing
    native_candidate_true_relres = nothing
    native_candidate_normwise = nothing
    certification_records = Dict{String,Any}[]
    certification_pending = false
    # One correction restart is allowed per solve under the current policy.
    correction_restart_consumed = false

    Threads.atomic_xchg!(KILL_FLAG, false)
    t0 = time()

    try
    for i in 1:maxiters
        _prepare_diagnostic_row!(data, i + 1, maxiters + 1)
        phase_reporting_enabled(progress.callback) &&
        _report_solver_progress!(progress, data, nrows, (iteration=PCG_COL_ITER, native=PCG_COL_NORM_R, true_rr=PCG_COL_RELATIVE_RESIDUAL, normwise=PCG_COL_RELATIVE_NORMWISE_2, block=PCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; policy=policy)
        iter = i
        row = i + 1  # row index in data matrix
        nrows = row
        local_iter += 1

        rz_new = threaded_dot(r, z, reduction_workspace)

        if local_iter > 1
            τ = rz_new / rz
            p .= p .* τ .+ z
        else
            τ = 0.0
            p .= z
        end

        threaded_spmv!(prod, A', p, spmv_plan)
        pAp = threaded_dot(p, prod, reduction_workspace)
        μ = rz_new / pAp

        # --- Diagnostic: orthogonality loss and tridiagonal recovery ---
        diag_abs_flag = 0.0
        # Orthogonality loss: v_{k+1}' M v_k = z_k' r_{k-1} / sqrt(rz_k * rz_{k-1})
        if local_iter == 1
            ortho_loss = 0.0
        else
            ortho_prod = rz_new * rz
            if ortho_prod < 0.0
                diag_abs_flag = 1.0
            end
            ortho_loss = (threaded_dot(z, r_prev, reduction_workspace)) / sqrt(abs(ortho_prod))
        end

        # Tridiagonal recovery: δ_k = 1/μ_k + τ_{k-1}/μ_{k-1}
        if local_iter == 1
            tridiag_diagonal = 1.0 / μ
        else
            tridiag_diagonal = 1.0 / μ + τ / μ_prev
        end

        previous_tridiag_beta = 0.0
        # β_k = sqrt(rz_k) / (μ_{k-1} * sqrt(rz_{k-1}))
        if local_iter > 1
            if rz_new < 0.0 || rz < 0.0
                diag_abs_flag = 1.0
            end
            previous_tridiag_beta = sqrt(abs(rz_new)) / (μ_prev * sqrt(abs(rz)))
        end

        _append_delayed_tridiagonal!(tridiag_stream, tridiag_diagonal, previous_tridiag_beta)
        μ_prev = μ
        r_prev .= r  # save r before update (diagnostic only)
        # --- End diagnostic ---

        updated_norm_x = _fused_update_norm!(update_x, μ, p, reduction_workspace, r, prod, correction_base, x)
        if correction_base !== nothing
            correction_assembly_steps += 1
        end

        cum_prec += Float64(preconditioner.LinearOperator(z, r))

        rz = rz_new

        # True residual
        threaded_spmv!(Ax, A', x, spmv_plan)
        physical_true_residual_evals += 1
        d = _fused_diagnostics!(r_true, Ax, b, seed, x, r,
            reduction_workspace, false)
        norm_r_true = d.norm_true
        norm_x = updated_norm_x
        error_val = d.norm_error / norm_seed
        energy = 0.5 * d.xAx - d.xb
        norm_r = d.norm_native
        relative_residual, relative_normwise_2 =
            compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)

        native_relative_residual = norm_r / norm_b
        native_candidate = native_relative_residual <= convergence_tolerance
        new_native_candidate = native_candidate && !certification_pending
        certification_iter = certification_pending || native_candidate
        if certification_iter
            certification_true_residual_evals += 1
            certification_check_count += 1
            _grow_diagnostic_vector!(certification_check_iters, certification_check_count, maxiters)
            _grow_diagnostic_vector!(true_residual_used_for_decision_iters, certification_check_count, maxiters)
            certification_check_iters[certification_check_count] = i
            true_residual_used_for_decision_iters[certification_check_count] = i
            if native_candidate && native_candidate_iter === nothing
                native_candidate_iter = i
                native_candidate_true_relres = relative_residual
                native_candidate_normwise = relative_normwise_2
            end
        end

        cum_base = Float64(base_fma) * (i + certification_true_residual_evals)
        cum_vector = Float64(
            PCG_VECTOR_FMA_SETUP +
            PCG_VECTOR_FMA_PER_ITER * i +
            TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * certification_true_residual_evals +
            correction_assembly_steps +
            PCG_VECTOR_FMA_RESIDUAL_REPLACEMENT * (residual_replacements + correction_restarts)
        ) * n
        _observe_native_candidate!(policy, native_candidate, i, cum_base + cum_prec + cum_vector)
        final_relative_residual = relative_residual

        data[row, PCG_COL_KRYLOV_BLOCK] = Float64(tridiag_block_index)
        data[row, PCG_COL_ITER]                = Float64(i)
        data[row, PCG_COL_RELATIVE_RESIDUAL]   = relative_residual
        data[row, PCG_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
        data[row, PCG_COL_ERROR]               = error_val
        data[row, PCG_COL_CUM_BASE_FMA]        = cum_base
        data[row, PCG_COL_CUM_PREC_FMA]        = cum_prec
        data[row, PCG_COL_CUM_VECTOR_FMA]      = cum_vector
        data[row, PCG_COL_MU]                  = μ
        data[row, PCG_COL_NORM_R]              = norm_r
        data[row, PCG_COL_NORM_R_TRUE]         = norm_r_true
        data[row, PCG_COL_RZ]                  = rz_new
        data[row, PCG_COL_PAP]                 = pAp
        data[row, PCG_COL_ENERGY]              = energy
        data[row, PCG_COL_ORTHO_LOSS]          = abs(ortho_loss)
        data[row, PCG_COL_DIAG_ABS_FLAG]       = diag_abs_flag

        if isnan(μ)
            status = NAN_DETECTED
            break
        end

        if KILL_FLAG[]
            status = KILLED
            break
        end

        certified_this_iter = false
        if certification_iter
            certified = relative_residual <= convergence_tolerance &&
                        relative_normwise_2 <= convergence_tolerance
            if solver_mode == :bounded_continuation
                cap = policy.continuation_work_limit
                within_budget = (work_limit <= 0 || cum_base + cum_prec + cum_vector <= work_limit) &&
                    (cap === nothing || cum_base + cum_prec + cum_vector <= cap)
                certified &= within_budget
            end
            certified_this_iter = certified
            correction_restart_this_iter =
                solver_mode == :certified &&
                !certified &&
                new_native_candidate &&
                !correction_restart_consumed &&
                relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)
            correction_restart_action = correction_restart_this_iter ? "correction_restart" : "continue_recurrence"
            if correction_restart_this_iter
                # Restart on r_true; successful setup freezes x as the base.
                # Later increments accumulate separately before forming global x.
                r .= r_true
                cum_prec += Float64(preconditioner.LinearOperator(z, r))
                rz_restart = threaded_dot(r, z, reduction_workspace)
                cum_vector += Float64(PCG_VECTOR_FMA_RESIDUAL_REPLACEMENT) * n
                data[row, PCG_COL_CUM_PREC_FMA] = cum_prec
                data[row, PCG_COL_CUM_VECTOR_FMA] = cum_vector
                if work_limit > 0 && (cum_base + cum_prec + cum_vector) > work_limit
                    correction_restart_action = "correction_restart_work_limit"
                    status = WORK_LIMIT
                elseif rz_restart <= 0.0 || !isfinite(rz_restart)
                    correction_restart_action = "correction_restart_breakdown"
                    status = BREAKDOWN
                else
                    restart_index = correction_restarts + 1
                    closed_tridiag_path = _close_coefficients!(tridiag_stream; next_iteration=i+1)
                    tridiag_block_index += 1
                    _record_correction_restart!(
                        correction_restart_records,
                        raw_data_out,
                        restart_index,
                        i,
                        native_relative_residual,
                        relative_residual,
                        relative_normwise_2,
                        norm_r,
                        norm_r_true,
                        x;
                        closed_tridiag_path = closed_tridiag_path,
                        next_tridiag_path = _tridiag_block_path(raw_data_out, tridiag_block_index),
                    )
                    rz = rz_restart
                    local_iter = 0
                    μ_prev = 0.0
                    correction_restarts += 1
                    correction_base = copy(x)
                    update_x = zeros(n)
                    correction_restart_consumed = true
                end
            end
            if new_native_candidate || certified
                event = new_native_candidate ? "native_candidate" : "certification_audit"
                action = certified ? "accept_convergence" : correction_restart_action
                _record_certification_check!(
                    certification_records,
                    i,
                    event,
                    native_relative_residual,
                    relative_residual,
                    relative_normwise_2,
                    norm_r,
                    norm_r_true;
                    certified = certified,
                    action = action,
                )
            end
            if certified
                certification_pending = false
            else
                new_native_candidate && (false_native_crossings += 1)
                certification_pending = true
            end
        end

        if status == BREAKDOWN || status == WORK_LIMIT
            break
        end

        stop_status = _policy_stop_status(policy, cum_base + cum_prec + cum_vector, certified_this_iter)
        if stop_status !== nothing
            status = stop_status
            break
        end
    end
    catch e
        e isa InterruptException || rethrow(e)
        status = KILLED
    end

    wall_time = time() - t0
    _experimental_final_iterate === nothing || copyto!(_experimental_final_iterate, x)
    _report_solver_progress!(progress, data, nrows, (iteration=PCG_COL_ITER, native=PCG_COL_NORM_R, true_rr=PCG_COL_RELATIVE_RESIDUAL, normwise=PCG_COL_RELATIVE_NORMWISE_2, block=PCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; final=true, status=string(status), policy=policy)
    total_fma = cum_base + cum_prec + cum_vector
    resize!(certification_check_iters, certification_check_count)
    resize!(true_residual_used_for_decision_iters, certification_check_count)

    # Finalize the bounded raw-data stream before deriving CSV summaries.
    output_operation = begin_reported_phase(progress.callback, "result_write")
    try
        used = _finish_diagnostic_data!(data, nrows;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)

        # Write tridiagonal coefficients (diagnostic)
        final_tridiag_path = _close_coefficients!(tridiag_stream;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)
        _clear_unwritten_next_tridiag!(residual_replacement_records, final_tridiag_path)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_tridiag_path)

        # Write column metadata
        columns_path = replace(raw_data_out, r"\.[^/]*$" => "_columns.json")
        open(columns_path, "w") do f
            JSON3.write(f, PCG_COLUMN_NAMES)
        end

        # Extract convergence thresholds post-hoc
        _write_convergence(
            convergence_data_out * "_relative_residual.csv",
            used,
            PCG_COL_RELATIVE_RESIDUAL,
        )
        _write_convergence(
            convergence_data_out * "_relative_normwise_2.csv",
            used,
            PCG_COL_RELATIVE_NORMWISE_2,
        )
        _write_certification_convergence(
            convergence_data_out * "_certification.csv",
            used,
            PCG_COL_NORM_R,
            norm_b,
            certification_records,
            certification_check_iters,
            solver_policy = :explicit,
            n = n,
            base_fma = base_fma,
            correction_restart_normwise_tolerance = correction_restart_normwise_tolerance,
            solver_mode = solver_mode,
            continuation_work_factor = continuation_work_factor,
            work_limit = work_limit,
        )

        if write_solution !== nothing && status == CONVERGED
            _write_solution_vec(write_solution, x)
        end
    catch e
        finish_reported_phase(output_operation; error=e)
        rethrow()
    end
    finish_reported_phase(output_operation)

    return SolverResult(
        status,
        iter,
        wall_time,
        total_fma,
        final_relative_residual,
        residual_replacements,
        residual_replacement_records,
        correction_restarts,
        correction_restart_records,
        physical_true_residual_evals,
        physical_true_residual_evals - certification_true_residual_evals,
        certification_true_residual_evals,
        certification_check_iters,
        true_residual_used_for_decision_iters,
        false_native_crossings,
        native_candidate_iter,
        native_candidate_true_relres,
        native_candidate_normwise,
        certification_records,
        _solver_policy_metadata(policy; status=status, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance),
    )
end

"""
Scan a tolerance column and write convergence threshold crossings to CSV.
"""
function _write_convergence(
    path::String,
    data::AbstractMatrix{Float64},
    col::Int,
)
    open(path, "w") do f
        write(f, "iteration,tolerance,actual,cumulative_base_fma,cumulative_prec_fma,cumulative_vector_fma\n")
        remaining = copy(tolerances)
        nrows = size(data, 1)
        for i in 1:nrows
            val = data[i, col]
            iter_num = data[i, PCG_COL_ITER]
            j = 1
            while j <= length(remaining)
                if val <= remaining[j]
                    write(f, string(Int(iter_num)))
                    write(f, ",")
                    write(f, string(remaining[j]))
                    write(f, ",")
                    write(f, string(val))
                    write(f, ",")
                    write(f, string(data[i, PCG_COL_CUM_BASE_FMA]))
                    write(f, ",")
                    write(f, string(data[i, PCG_COL_CUM_PREC_FMA]))
                    write(f, ",")
                    write(f, string(data[i, PCG_COL_CUM_VECTOR_FMA]))
                    write(f, "\n")
                    deleteat!(remaining, j)
                else
                    j += 1
                end
            end
            if isempty(remaining)
                break
            end
        end
    end
end

function _row_index_for_iter(data::AbstractMatrix{Float64}, iter::Int)
    for row in 1:size(data, 1)
        if Int(data[row, PCG_COL_ITER]) == iter
            return row
        end
    end
    return nothing
end

function _row_values(data::AbstractMatrix{Float64}, row, native_col::Int, norm_b::Float64)
    row === nothing && return (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    native_norm = abs(data[row, native_col])
    native = iszero(norm_b) && iszero(native_norm) ? 0.0 : native_norm / norm_b
    true_relres = data[row, PCG_COL_RELATIVE_RESIDUAL]
    normwise = data[row, PCG_COL_RELATIVE_NORMWISE_2]
    base = data[row, PCG_COL_CUM_BASE_FMA]
    prec = data[row, PCG_COL_CUM_PREC_FMA]
    vector = data[row, PCG_COL_CUM_VECTOR_FMA]
    return (
        Int(data[row, PCG_COL_ITER]),
        native,
        true_relres,
        normwise,
        base,
        prec,
        vector,
        base + prec + vector,
    )
end

function _first_true_hit_row(data::AbstractMatrix{Float64}, tol::Float64)
    for row in 1:size(data, 1)
        if data[row, PCG_COL_RELATIVE_RESIDUAL] <= tol &&
           data[row, PCG_COL_RELATIVE_NORMWISE_2] <= tol
            return row
        end
    end
    return nothing
end

function _first_native_true_hit_row(
    data::AbstractMatrix{Float64},
    native_col::Int,
    norm_b::Float64,
    tol::Float64,
)
    threshold = tol * norm_b
    for row in 1:size(data, 1)
        if abs(data[row, native_col]) <= threshold &&
           data[row, PCG_COL_RELATIVE_RESIDUAL] <= tol &&
           data[row, PCG_COL_RELATIVE_NORMWISE_2] <= tol
            return row
        end
    end
    return nothing
end

function _first_certification_record(records::Vector{Dict{String,Any}}, tol::Float64)
    for (idx, record) in pairs(records)
        if get(record, "certified", false) == true &&
           Float64(get(record, "true_relres", Inf)) <= tol &&
           Float64(get(record, "relative_normwise_2", Inf)) <= tol
            return idx, record
        end
    end
    return nothing, nothing
end

function _false_native_crossings_through(records::Vector{Dict{String,Any}}, idx)
    idx === nothing && return nothing
    count = 0
    for j in 1:idx
        record = records[j]
        if get(record, "event", "") == "native_candidate" &&
           get(record, "certified", false) != true
            count += 1
        end
    end
    return count
end

function _certification_checks_through(check_iters::Vector{Int}, iter)
    iter === nothing && return nothing
    count = 0
    for check_iter in check_iters
        check_iter <= iter && (count += 1)
    end
    return count
end

function _simulate_certification_policy(
    data::AbstractMatrix{Float64},
    native_col::Int,
    norm_b::Float64,
    tol::Float64,
    solver_policy::Symbol,
    correction_restart_normwise_tolerance::Float64;
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
    work_limit::Real = -1,
    n::Int = 0,
    base_fma::Integer = 0,
    actual_check_iters::Vector{Int} = Int[],
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    pending = false
    certification_checks = 0
    false_native_crossings = 0
    threshold = tol * norm_b
    effective_restart_tol = min(tol, correction_restart_normwise_tolerance)

    for row in 1:size(data, 1)
        native_norm = abs(data[row, native_col])
        true_relres = data[row, PCG_COL_RELATIVE_RESIDUAL]
        normwise = data[row, PCG_COL_RELATIVE_NORMWISE_2]
        if !isfinite(native_norm) || !isfinite(true_relres) || !isfinite(normwise)
            continue
        end

        native_ok = native_norm <= threshold
        if !pending && !native_ok
            continue
        end

        true_ok = true_relres <= tol && normwise <= tol
        iter = Int(round(data[row, PCG_COL_ITER]))
        # x0 = b = 0 is an exact solution, not a charged matrix evaluation.
        if iter == 0 && iszero(norm_b) && iszero(native_norm) &&
                iszero(true_relres) && iszero(normwise)
            return (row=row, iter=0, certification_checks=0,
                false_native_crossings=0, status="certified_from_raw",
                divergence_iter=nothing, divergence_reason="")
        end
        certification_checks += 1
        if solver_mode == :bounded_continuation
            actual_checks = _certification_checks_through(actual_check_iters, iter)
            total = sum(data[row, PCG_COL_CUM_BASE_FMA:PCG_COL_CUM_VECTOR_FMA]) +
                (certification_checks - actual_checks) * (Float64(base_fma) + TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * n)
            _observe_native_candidate!(policy, native_ok, iter, total)
            stop = _policy_stop_status(policy, total, true_ok)
            if stop in (WORK_LIMIT, CONTINUATION_LIMIT)
                return (row=nothing, iter=nothing, certification_checks=certification_checks,
                    false_native_crossings=false_native_crossings,
                    status=stop == WORK_LIMIT ? "work_limit" : "continuation_work_limit",
                    divergence_iter=iter, divergence_reason="simulated certification budget exhausted")
            end
        end
        if true_ok
            return (
                row = row,
                iter = iter,
                certification_checks = certification_checks,
                false_native_crossings = false_native_crossings,
                status = "certified_from_raw",
                divergence_iter = nothing,
                divergence_reason = "",
            )
        end

        if !pending
            false_native_crossings += 1
            if solver_mode == :certified && solver_policy == :explicit && normwise <= effective_restart_tol
                return (
                    row = nothing,
                    iter = nothing,
                    certification_checks = certification_checks,
                    false_native_crossings = false_native_crossings,
                    status = "trajectory_diverged_correction_restart",
                    divergence_iter = iter,
                    divergence_reason = "first native candidate fails true certification and satisfies correction-restart trigger",
                )
            end
        end

        if solver_mode == :certified && solver_policy == :scalar
            if normwise <= effective_restart_tol
                return (
                    row = nothing,
                    iter = nothing,
                    certification_checks = certification_checks,
                    false_native_crossings = false_native_crossings,
                    status = "trajectory_diverged_correction_restart",
                    divergence_iter = iter,
                    divergence_reason = "charged scalar audit fails true certification and satisfies correction-restart trigger",
                )
            end
        end

        pending = true
    end

    return (
        row = nothing,
        iter = nothing,
        certification_checks = certification_checks,
        false_native_crossings = false_native_crossings,
        status = pending ? "not_certified_in_raw_after_native_candidate" : "native_not_reached_in_raw",
        divergence_iter = nothing,
        divergence_reason = "",
    )
end

function _row_values_with_simulated_certification_cost(
    data::AbstractMatrix{Float64},
    row,
    native_col::Int,
    norm_b::Float64,
    n::Int,
    base_fma::Integer,
    simulated_checks::Int,
    certification_check_iters::Vector{Int},
)
    vals = _row_values(data, row, native_col, norm_b)
    vals[1] === nothing && return vals
    actual_checks = _certification_checks_through(certification_check_iters, vals[1])
    base = vals[5] -
        Float64(actual_checks * base_fma) +
        Float64(simulated_checks * base_fma)
    vector = vals[7] -
        Float64(actual_checks * TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * n) +
        Float64(simulated_checks * TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * n)
    return (
        vals[1],
        vals[2],
        vals[3],
        vals[4],
        base,
        vals[6],
        vector,
        base + vals[6] + vector,
    )
end

function _write_csv_row(io, row)
    for (idx, value) in pairs(row)
        idx > 1 && write(io, ",")
        value === nothing || print(io, value)
    end
    write(io, "\n")
end

"""
Write tolerance-level diagnostic and policy certification crossings.

`true_hit_*` is the first diagnostic true-residual crossing. It may be an oracle
observation. `native_true_hit_*` is the first iteration where the native
residual estimate and true residual criteria are simultaneously below the
tolerance. `certified_*` is the first policy certification record that
accepted convergence (including exact zero-RHS acceptance without an evaluation
at iteration zero); it is empty for solvers/policies that do not yet emit
certification records. `sim_*` is a post-hoc simulation of the declared
certification policy at each tolerance using the raw trajectory; it reports
when that policy can be reconstructed from raw data and when it would diverge
because it needs a state-changing correction restart.
"""
function _write_certification_convergence(
    path::String,
    data::AbstractMatrix{Float64},
    native_col::Int,
    norm_b::Float64,
    certification_records::Vector{Dict{String,Any}} = Dict{String,Any}[],
    certification_check_iters::Vector{Int} = Int[],
    ;
    solver_policy::Symbol = :explicit,
    n::Int = 0,
    base_fma::Integer = 0,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
    work_limit::Real = -1,
)
    header = [
        "tolerance",
        "true_hit_iter",
        "true_hit_native_relres",
        "true_hit_relres",
        "true_hit_normwise",
        "true_hit_base_fma",
        "true_hit_prec_fma",
        "true_hit_vector_fma",
        "true_hit_total_fma",
        "true_hit_is_load_bearing",
        "native_true_hit_iter",
        "native_true_hit_native_relres",
        "native_true_hit_relres",
        "native_true_hit_normwise",
        "native_true_hit_base_fma",
        "native_true_hit_prec_fma",
        "native_true_hit_vector_fma",
        "native_true_hit_total_fma",
        "certified_iter",
        "certified_native_relres",
        "certified_true_relres",
        "certified_normwise",
        "certified_base_fma",
        "certified_prec_fma",
        "certified_vector_fma",
        "certified_total_fma",
        "certification_checks_to_certified",
        "false_native_crossings_to_certified",
        "sim_certified_iter",
        "sim_certified_native_relres",
        "sim_certified_true_relres",
        "sim_certified_normwise",
        "sim_certified_base_fma",
        "sim_certified_prec_fma",
        "sim_certified_vector_fma",
        "sim_certified_total_fma",
        "sim_certification_checks_to_certified",
        "sim_false_native_crossings_to_certified",
        "sim_status",
        "sim_divergence_iter",
        "sim_divergence_reason",
    ]
    decision_iters = Set{Int}()
    for record in certification_records
        push!(decision_iters, Int(get(record, "iter", -1)))
    end

    open(path, "w") do f
        write(f, join(header, ","))
        write(f, "\n")
        for tol in tolerances
            true_row = _first_true_hit_row(data, tol)
            native_true_row = _first_native_true_hit_row(data, native_col, norm_b, tol)
            cert_idx, cert_record = _first_certification_record(certification_records, tol)
            cert_row = cert_record === nothing ? nothing :
                _row_index_for_iter(data, Int(cert_record["iter"]))
            cert_iter = cert_record === nothing ? nothing : Int(cert_record["iter"])

            true_vals = _row_values(data, true_row, native_col, norm_b)
            native_true_vals = _row_values(data, native_true_row, native_col, norm_b)
            cert_vals = _row_values(data, cert_row, native_col, norm_b)
            true_iter = true_vals[1]
            true_load_bearing = true_iter === nothing ? nothing : (true_iter in decision_iters)
            sim = _simulate_certification_policy(
                data,
                native_col,
                norm_b,
                tol,
                solver_policy,
                correction_restart_normwise_tolerance;
                solver_mode=solver_mode, continuation_work_factor=continuation_work_factor,
                work_limit=work_limit, n=n, base_fma=base_fma,
                actual_check_iters=certification_check_iters,
            )
            sim_vals = _row_values_with_simulated_certification_cost(
                data,
                sim.row,
                native_col,
                norm_b,
                n,
                base_fma,
                sim.certification_checks,
                certification_check_iters,
            )

            row = Any[
                tol,
                true_vals[1],
                true_vals[2],
                true_vals[3],
                true_vals[4],
                true_vals[5],
                true_vals[6],
                true_vals[7],
                true_vals[8],
                true_load_bearing,
                native_true_vals[1],
                native_true_vals[2],
                native_true_vals[3],
                native_true_vals[4],
                native_true_vals[5],
                native_true_vals[6],
                native_true_vals[7],
                native_true_vals[8],
                cert_vals[1],
                cert_record === nothing ? nothing : get(cert_record, "eta", nothing),
                cert_record === nothing ? nothing : get(cert_record, "true_relres", nothing),
                cert_record === nothing ? nothing : get(cert_record, "relative_normwise_2", nothing),
                cert_vals[5],
                cert_vals[6],
                cert_vals[7],
                cert_vals[8],
                _certification_checks_through(certification_check_iters, cert_iter),
                _false_native_crossings_through(certification_records, cert_idx),
                sim_vals[1],
                sim_vals[2],
                sim_vals[3],
                sim_vals[4],
                sim_vals[5],
                sim_vals[6],
                sim_vals[7],
                sim_vals[8],
                sim.row === nothing ? nothing : sim.certification_checks,
                sim.row === nothing ? nothing : sim.false_native_crossings,
                sim.status,
                sim.divergence_iter,
                sim.divergence_reason,
            ]
            _write_csv_row(f, row)
        end
    end
end

# Lanczos PCG column indices (extends the base PCG columns)
const LPCG_COL_ITER                = 1
const LPCG_COL_RELATIVE_RESIDUAL   = 2
const LPCG_COL_RELATIVE_NORMWISE_2 = 3
const LPCG_COL_ERROR               = 4
const LPCG_COL_CUM_BASE_FMA        = 5
const LPCG_COL_CUM_PREC_FMA        = 6
const LPCG_COL_CUM_VECTOR_FMA      = 7
const LPCG_COL_GAMMA               = 8
const LPCG_COL_NORM_R              = 9
const LPCG_COL_NORM_R_TRUE         = 10
const LPCG_COL_DELTA               = 11
const LPCG_COL_BETA                = 12
const LPCG_COL_ENERGY              = 13
const LPCG_COL_A_NORM_ERROR        = 14
const LPCG_COL_A_NORM_LANCZOS      = 15
const LPCG_COL_ORTHO_LOSS          = 16
const LPCG_COL_DIAG_ABS_FLAG       = 17
const LPCG_COL_KRYLOV_BLOCK        = 18
const LPCG_NUM_COLS                = 18

# Lanczos PCG setup vector FMA:
#   threaded_dot(v, Mv, reduction_workspace) (β₁²): n
# Total setup: n
# Note: v./=β, Mv./=β normalization is equivalent to per-iteration normalization
# (MINRES counts its normalization per-iteration, so we do the same for consistency)
#
# Per-iteration vector FMA for Lanczos PCG:
#   threaded_dot(v, Mv_next, reduction_workspace) (δ): n
#   Mv_next .-= δ .* Mv (axpy): n
#   Mv_next .-= β .* Mv_prev (axpy): n  [skipped iter 1]
#   threaded_dot(v, Mv, reduction_workspace) (β²): n
#   v ./= β, Mv ./= β (2 scales): 2n
#   x .+= γ .* p (axpy): n
#   p .= σ .* v .+ ω .* p (fused): 2n
# Total: 9n per iteration (8n at iteration 1, 9n thereafter)
# Scalar native crossings charge certification SpMVs separately in cum_base.
# This replacement constant is reserved for future recovery restarts.
const LPCG_VECTOR_FMA_PER_ITER = 9
const LPCG_VECTOR_FMA_SETUP = 1
const LPCG_VECTOR_FMA_RESIDUAL_REPLACEMENT = 1

const LPCG_COLUMN_NAMES = [
    "iteration",
    "relative_residual",
    "relative_normwise_2",
    "relative_error",
    "cumulative_base_fma",
    "cumulative_prec_fma",
    "cumulative_vector_fma",
    "gamma",
    "norm_r",
    "norm_r_true",
    "delta",
    "beta",
    "energy",
    "a_norm_error",
    "a_norm_error_lanczos",
    "orthogonality_loss",
    "diag_abs_flag",
    "krylov_block",
]

"""
    lanczos_pcg(A, b, seed, spectral_norm, preconditioner,
                maxiters, raw_data_out, convergence_data_out;
                work_limit=-1)

Solves `Ax = b` using the Lanczos form of Preconditioned Conjugate Gradient.

The Lanczos recurrence explicitly builds orthogonal Lanczos vectors and
reconstructs the solution from them, rather than using the polynomial CG
recurrence. This provides access to the Lanczos tridiagonal matrix T_k,
from which Ritz values and condition number estimates are computed.

The tridiagonal coefficients (delta, beta) are streamed in numbered blocks
alongside the iteration data. Each correction restart has a separate manifest
for post-hoc spectral analysis.

# Arguments
- `A::SparseMatrixCSC`: The system matrix (SPD).
- `b::Vector`: Right-hand side vector.
- `seed::Vector`: Known solution for error tracking.
- `spectral_norm::Float64`: Spectral norm of `A`, used for normwise backward error.
- `preconditioner::Preconditioner`: The preconditioner to apply.
- `maxiters::Int64`: Maximum number of iterations.
- `work_limit::Int64`: Maximum cumulative FMA before termination. -1 for no limit.
- `correction_restart_normwise_tolerance::Float64`: A failed native
  certification whose global normwise backward error is below
  `min(convergence_tolerance, correction_restart_normwise_tolerance)`
  restarts Lanczos-PCG on the explicitly recomputed global residual while
  keeping all diagnostics relative to the original system. Eligibility is
  checked at every charged pending audit.
- `raw_data_out::String`: Logical .npy.zstd path; v2 blocks are written to its sibling directory.
- `convergence_data_out::String`: Path prefix for convergence threshold CSVs.

# Returns
- `SolverResult`: Contains `status::SolverStatus` and `iterations::Int`.

# References
- Frommer & Maass, Fast CG-Based Methods for Tikhonov-Phillips Regularization,
  SIAM J. Sci. Comput. 20(5), 1999.
- Paige & Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
  SIAM J. Numer. Anal. 12(4), 1975.
"""
function _lanczos_pcg_with_progress(
    A::SparseMatrixCSC,
    b::Vector{Float64},
    seed::Vector{Float64},
    spectral_norm::Float64,
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
    convergence_tolerance::Float64 = minimum_tolerance,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    write_solution::Union{Nothing,String} = nothing,
    _experimental_final_iterate::Union{Nothing,Vector{Float64}} = nothing,
    _diagnostic_block_rows::Int = DIAGNOSTIC_BLOCK_ROWS,
    progress = NumericalProgress(NOOP_PHASE_CALLBACK),
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    n = size(A, 1)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    spmv_plan = _spmv_plan(A)
    reduction_workspace = FusedReductionWorkspace()
    base_fma = UInt64(nnz(A))
    norm_b = threaded_norm(b)
    norm_seed = threaded_norm(seed)

    # Preallocate iteration data
    data = DiagnosticBlockWriter(raw_data_out, maxiters + 1, LPCG_COLUMN_NAMES;
        block_rows=_diagnostic_block_rows, callback=progress.callback)

    # Tridiagonal coefficients for post-hoc Ritz analysis

    # Working vectors
    x = zeros(n)
    # Before recovery, updates target x directly. Allocate correction storage lazily.
    correction_base = nothing
    update_x = x
    correction_assembly_steps = 0
    Mv = copy(b)            # Mv_0 = b (= r_0 since x_0 = 0)
    v = zeros(n)             # v = M^{-1} Mv
    Mv_prev = zeros(n)
    Mv_next = zeros(n)
    p = zeros(n)
    Ax = zeros(n)
    r_true = similar(b)

    # Initialize Lanczos vectors: v = M^{-1} r_0, β_1 = sqrt(v' M v)
    # p_0 = M^{-1} r_0 (unnormalized), then v and Mv are normalized by β_1
    x .= 0.0
    Mv .= b
    cum_prec = Float64(preconditioner.LinearOperator(v, Mv))
    β_sq = threaded_dot(v, Mv, reduction_workspace)
    if β_sq <= 0.0 || !isfinite(β_sq)
        return _initial_breakdown_result(data, b, seed, cum_prec,
            Float64(LPCG_VECTOR_FMA_SETUP) * n, policy, convergence_data_out,
            raw_data_out, progress, convergence_tolerance, correction_restart_normwise_tolerance)
    end
    β = sqrt(β_sq)
    σ = β
    p .= v
    v ./= β
    Mv ./= β
    Mv_prev .= Mv

    ω = 0.0
    γ = 1.0
    β_prev = 0.0

    # ||e_0||_A^2 = seed' b (since b = A * seed and x_0 = 0)
    e0_A_sq = threaded_dot(seed, b, reduction_workspace)

    norm_r_actual = threaded_norm(b)
    data[1, LPCG_COL_ITER]                = 0.0
    data[1, LPCG_COL_RELATIVE_RESIDUAL]   = 1.0
    data[1, LPCG_COL_RELATIVE_NORMWISE_2] = 1.0
    data[1, LPCG_COL_ERROR]               = 1.0
    data[1, LPCG_COL_CUM_BASE_FMA]        = 0.0
    data[1, LPCG_COL_CUM_PREC_FMA]        = cum_prec
    data[1, LPCG_COL_CUM_VECTOR_FMA]      = Float64(LPCG_VECTOR_FMA_SETUP) * n
    data[1, LPCG_COL_GAMMA]               = 0.0
    data[1, LPCG_COL_NORM_R]              = norm_r_actual
    data[1, LPCG_COL_NORM_R_TRUE]         = norm_r_actual
    data[1, LPCG_COL_DELTA]               = 0.0
    data[1, LPCG_COL_BETA]                = β
    data[1, LPCG_COL_ENERGY]              = 0.0
    data[1, LPCG_COL_A_NORM_ERROR]        = sqrt(abs(e0_A_sq))
    data[1, LPCG_COL_A_NORM_LANCZOS]      = sqrt(abs(e0_A_sq))
    data[1, LPCG_COL_ORTHO_LOSS]          = 0.0
    data[1, LPCG_COL_DIAG_ABS_FLAG]       = e0_A_sq < 0.0 ? 1.0 : 0.0
    data[1, LPCG_COL_KRYLOV_BLOCK]        = 0.0

    iter = 0
    nrows = 1
    status = MAX_ITERS
    final_relative_residual = 1.0
    cum_base = 0.0
    cum_vector = Float64(LPCG_VECTOR_FMA_SETUP) * n
    local_iter = 0
    residual_replacements = 0
    residual_replacement_records = Dict{String,Any}[]
    correction_restarts = 0
    correction_restart_records = Dict{String,Any}[]
    tridiag_block_index = 0
    tridiag_stream = CoefficientStream(raw_data_out, maxiters, _diagnostic_block_rows;
        callback=progress.callback, source_attempt=basename(data.attempt))
    physical_true_residual_evals = 0
    certification_true_residual_evals = 0
    certification_check_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    true_residual_used_for_decision_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    certification_check_count = 0
    false_native_crossings = 0
    native_candidate_iter = nothing
    native_candidate_true_relres = nothing
    native_candidate_normwise = nothing
    certification_records = Dict{String,Any}[]
    certification_pending = false
    # Current policy allows one scalar correction restart. Additional false
    # native crossings after that continue the same correction recurrence.
    correction_restart_consumed = false
    Threads.atomic_xchg!(KILL_FLAG, false)
    t0 = time()

    try
    for i in 1:maxiters
        _prepare_diagnostic_row!(data, i + 1, maxiters + 1)
        phase_reporting_enabled(progress.callback) &&
        _report_solver_progress!(progress, data, nrows, (iteration=LPCG_COL_ITER, native=LPCG_COL_NORM_R, true_rr=LPCG_COL_RELATIVE_RESIDUAL, normwise=LPCG_COL_RELATIVE_NORMWISE_2, block=LPCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; policy=policy)
        iter = i
        row = i + 1
        nrows = row
        local_iter += 1

        # Lanczos step: β_{k+1} Mv_{k+1} = A v_k - δ_k Mv_k - β_k Mv_{k-1}
        threaded_spmv!(Mv_next, A', v, spmv_plan)              # Mv_next = A v_k
        δ = threaded_dot(v, Mv_next, reduction_workspace)               # δ_k = v_k' A v_k

        # γ_k = 1 / (δ_k - ω_{k-1} / γ_{k-1})
        γ = inv(δ - ω / γ)

        Mv_next .-= δ .* Mv              # -= δ_k Mv_k
        if local_iter > 1
            Mv_next .-= β .* Mv_prev     # -= β_k Mv_{k-1}
            Mv_prev .= Mv
        end
        Mv .= Mv_next
        cum_prec += preconditioner.LinearOperator(v, Mv)  # v_{k+1} = M^{-1} Mv_{k+1}
        β_sq = threaded_dot(v, Mv, reduction_workspace)
        lanczos_exhausted = iszero(β_sq) && all(iszero, Mv)
        if (β_sq <= 0.0 && !lanczos_exhausted) || !isfinite(β_sq)
            _append_coefficients!(tridiag_stream, δ, β_sq)
            status = BREAKDOWN
            nrows -= 1  # row was not filled with data
            break
        end
        β = sqrt(β_sq)                   # β_{k+1}
        _append_coefficients!(tridiag_stream, δ, β)
        if !lanczos_exhausted
            v ./= β
            Mv ./= β
        end

        # No next Lanczos vector is needed for an exact final update.
        ortho_loss = lanczos_exhausted ? 0.0 : threaded_dot(v, Mv_prev, reduction_workspace)

        updated_norm_x = _fused_update_norm!(update_x, γ, p, reduction_workspace, nothing, nothing, correction_base, x)
        if correction_base !== nothing
            correction_assembly_steps += 1
        end

        ω = β * γ
        σ = -ω * σ
        ω = ω * ω
        p .= σ .* v .+ ω .* p

        # True residual and A-norm error
        threaded_spmv!(Ax, A', x, spmv_plan)
        physical_true_residual_evals += 1
        d = _fused_diagnostics!(r_true, Ax, b, seed, x, nothing,
            reduction_workspace, false)
        norm_r_true = d.norm_true
        norm_x = updated_norm_x
        error_val = d.norm_error / norm_seed
        energy = 0.5 * d.xAx - d.xb
        a_norm_error_sq = e0_A_sq - 2.0 * d.xb + d.xAx
        diag_abs_flag = a_norm_error_sq < 0.0 ? 1.0 : 0.0
        a_norm_error = sqrt(abs(a_norm_error_sq))
        a_norm_lanczos_sq = e0_A_sq + 2.0 * energy
        if a_norm_lanczos_sq < 0.0
            diag_abs_flag = 1.0
        end
        a_norm_lanczos = sqrt(abs(a_norm_lanczos_sq))
        relative_residual, relative_normwise_2 =
            compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)

        norm_r_lanczos = abs(σ)

        native_relative_residual = norm_r_lanczos / norm_b
        native_candidate = native_relative_residual <= convergence_tolerance
        new_native_candidate = native_candidate && !certification_pending
        certification_iter = certification_pending || native_candidate
        if certification_iter
            certification_true_residual_evals += 1
            certification_check_count += 1
            _grow_diagnostic_vector!(certification_check_iters, certification_check_count, maxiters)
            _grow_diagnostic_vector!(true_residual_used_for_decision_iters, certification_check_count, maxiters)
            certification_check_iters[certification_check_count] = i
            true_residual_used_for_decision_iters[certification_check_count] = i
            if native_candidate && native_candidate_iter === nothing
                native_candidate_iter = i
                native_candidate_true_relres = relative_residual
                native_candidate_normwise = relative_normwise_2
            end
        end

        cum_base = Float64(base_fma) * (
            i + residual_replacements + certification_true_residual_evals
        )
        cum_vector = Float64(
            LPCG_VECTOR_FMA_SETUP +
            LPCG_VECTOR_FMA_PER_ITER * i +
            TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * certification_true_residual_evals +
            correction_assembly_steps +
            LPCG_VECTOR_FMA_SETUP * correction_restarts +
            LPCG_VECTOR_FMA_RESIDUAL_REPLACEMENT * residual_replacements
        ) * n
        _observe_native_candidate!(policy, native_candidate, i, cum_base + cum_prec + cum_vector)
        final_relative_residual = relative_residual

        data[row, LPCG_COL_ITER]                = Float64(i)
        data[row, LPCG_COL_RELATIVE_RESIDUAL]   = relative_residual
        data[row, LPCG_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
        data[row, LPCG_COL_ERROR]               = error_val
        data[row, LPCG_COL_CUM_BASE_FMA]        = cum_base
        data[row, LPCG_COL_CUM_PREC_FMA]        = cum_prec
        data[row, LPCG_COL_CUM_VECTOR_FMA]      = cum_vector
        data[row, LPCG_COL_GAMMA]               = abs(γ)
        data[row, LPCG_COL_NORM_R]              = norm_r_lanczos
        data[row, LPCG_COL_NORM_R_TRUE]         = norm_r_true
        data[row, LPCG_COL_DELTA]               = δ
        data[row, LPCG_COL_BETA]                = β
        data[row, LPCG_COL_ENERGY]              = energy
        data[row, LPCG_COL_A_NORM_ERROR]        = a_norm_error
        data[row, LPCG_COL_A_NORM_LANCZOS]      = a_norm_lanczos
        data[row, LPCG_COL_ORTHO_LOSS]          = ortho_loss
        data[row, LPCG_COL_DIAG_ABS_FLAG]       = diag_abs_flag
        data[row, LPCG_COL_KRYLOV_BLOCK]        = Float64(tridiag_block_index)

        if isnan(γ) || isnan(σ)
            status = NAN_DETECTED
            break
        end

        if KILL_FLAG[]
            status = KILLED
            break
        end

        certified_this_iter = false
        if certification_iter
            certified = relative_residual <= convergence_tolerance &&
                        relative_normwise_2 <= convergence_tolerance
            if solver_mode == :bounded_continuation
                cap = policy.continuation_work_limit
                within_budget = (work_limit <= 0 || cum_base + cum_prec + cum_vector <= work_limit) &&
                    (cap === nothing || cum_base + cum_prec + cum_vector <= cap)
                certified &= within_budget
            end
            certified_this_iter = certified
            correction_restart_this_iter =
                solver_mode == :certified &&
                !certified &&
                certification_iter &&
                !correction_restart_consumed &&
                relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)
            correction_restart_action = correction_restart_this_iter ? "correction_restart" :
                                        "continue_recurrence"
            if correction_restart_this_iter
                # Restart on r_true; successful setup freezes x as the base.
                # Later increments accumulate separately before forming global x.
                Mv .= r_true
                cum_prec += Float64(preconditioner.LinearOperator(v, Mv))
                β_sq = threaded_dot(v, Mv, reduction_workspace)
                cum_vector += Float64(LPCG_VECTOR_FMA_SETUP) * n
                data[row, LPCG_COL_CUM_PREC_FMA] = cum_prec
                data[row, LPCG_COL_CUM_VECTOR_FMA] = cum_vector
                if work_limit > 0 && (cum_base + cum_prec + cum_vector) > work_limit
                    correction_restart_action = "correction_restart_work_limit"
                    status = WORK_LIMIT
                elseif β_sq <= 0.0
                    correction_restart_action = "correction_restart_breakdown"
                    status = BREAKDOWN
                else
                    restart_index = correction_restarts + 1
                    closed_tridiag_path = _close_coefficients!(tridiag_stream; next_iteration=i+1)
                    tridiag_block_index += 1
                    _record_correction_restart!(
                        correction_restart_records,
                        raw_data_out,
                        restart_index,
                        i,
                        native_relative_residual,
                        relative_residual,
                        relative_normwise_2,
                        norm_r_lanczos,
                        norm_r_true,
                        x;
                        closed_tridiag_path = closed_tridiag_path,
                        next_tridiag_path = _tridiag_block_path(raw_data_out, tridiag_block_index),
                    )

                    β = sqrt(β_sq)
                    σ = β
                    p .= v
                    v ./= β
                    Mv ./= β
                    Mv_prev .= Mv
                    fill!(Mv_next, 0.0)
                    ω = 0.0
                    γ = 1.0
                    local_iter = 0
                    correction_restarts += 1
                    correction_base = copy(x)
                    update_x = zeros(n)
                    correction_restart_consumed = true
                end
            end
            if new_native_candidate || certified || correction_restart_this_iter
                event = new_native_candidate ? "native_candidate" : "certification_audit"
                action = certified ? "accept_convergence" :
                         correction_restart_action
                _record_certification_check!(
                    certification_records,
                    i,
                    event,
                    native_relative_residual,
                    relative_residual,
                    relative_normwise_2,
                    norm_r_lanczos,
                    norm_r_true;
                    certified = certified,
                    action = action,
                )
            end
            if certified
                certification_pending = false
            else
                if new_native_candidate
                    false_native_crossings += 1
                end
                certification_pending = true
            end
        end

        if status == BREAKDOWN || status == WORK_LIMIT
            break
        end

        stop_status = _policy_stop_status(policy, cum_base + cum_prec + cum_vector, certified_this_iter)
        if stop_status !== nothing
            status = stop_status
            break
        end
        # A successful correction setup resets local_iter and supplies a new
        # basis. Otherwise an exhausted space cannot take another step.
        if lanczos_exhausted && local_iter > 0
            status = BREAKDOWN
            break
        end
    end
    catch e
        e isa InterruptException || rethrow(e)
        status = KILLED
    end

    wall_time = time() - t0
    _experimental_final_iterate === nothing || copyto!(_experimental_final_iterate, x)
    _report_solver_progress!(progress, data, nrows, (iteration=LPCG_COL_ITER, native=LPCG_COL_NORM_R, true_rr=LPCG_COL_RELATIVE_RESIDUAL, normwise=LPCG_COL_RELATIVE_NORMWISE_2, block=LPCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; final=true, status=string(status), policy=policy)
    total_fma = cum_base + cum_prec + cum_vector
    resize!(certification_check_iters, certification_check_count)
    resize!(true_residual_used_for_decision_iters, certification_check_count)

    # Finalize raw blocks, then scan them without concatenating the history.
    output_operation = begin_reported_phase(progress.callback, "result_write")
    try
        used = _finish_diagnostic_data!(data, nrows;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)

        # Write column metadata
        columns_path = replace(raw_data_out, r"\.[^/]*$" => "_columns.json")
        open(columns_path, "w") do f
            JSON3.write(f, LPCG_COLUMN_NAMES)
        end

        # Write tridiagonal coefficients for post-hoc Ritz analysis
        final_tridiag_path = _close_coefficients!(tridiag_stream;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)
        _clear_unwritten_next_tridiag!(residual_replacement_records, final_tridiag_path)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_tridiag_path)

        # Extract convergence thresholds post-hoc
        _write_convergence(
            convergence_data_out * "_relative_residual.csv",
            used,
            LPCG_COL_RELATIVE_RESIDUAL,
        )
        _write_convergence(
            convergence_data_out * "_relative_normwise_2.csv",
            used,
            LPCG_COL_RELATIVE_NORMWISE_2,
        )
        _write_certification_convergence(
            convergence_data_out * "_certification.csv",
            used,
            LPCG_COL_NORM_R,
            norm_b,
            certification_records,
            certification_check_iters,
            solver_policy = :scalar,
            n = n,
            base_fma = base_fma,
            correction_restart_normwise_tolerance = correction_restart_normwise_tolerance,
            solver_mode = solver_mode,
            continuation_work_factor = continuation_work_factor,
            work_limit = work_limit,
        )

        if write_solution !== nothing && status == CONVERGED
            _write_solution_vec(write_solution, x)
        end
    catch e
        finish_reported_phase(output_operation; error=e)
        rethrow()
    end
    finish_reported_phase(output_operation)

    return SolverResult(
        status,
        iter,
        wall_time,
        total_fma,
        final_relative_residual,
        residual_replacements,
        residual_replacement_records,
        correction_restarts,
        correction_restart_records,
        physical_true_residual_evals,
        physical_true_residual_evals - certification_true_residual_evals,
        certification_true_residual_evals,
        certification_check_iters,
        true_residual_used_for_decision_iters,
        false_native_crossings,
        native_candidate_iter,
        native_candidate_true_relres,
        native_candidate_normwise,
        certification_records,
        _solver_policy_metadata(policy; status=status, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance),
    )
end

# MINRES column indices
const MINRES_COL_ITER                = 1
const MINRES_COL_RELATIVE_RESIDUAL   = 2
const MINRES_COL_RELATIVE_NORMWISE_2 = 3
const MINRES_COL_ERROR               = 4
const MINRES_COL_CUM_BASE_FMA        = 5
const MINRES_COL_CUM_PREC_FMA        = 6
const MINRES_COL_CUM_VECTOR_FMA      = 7
const MINRES_COL_PHI                 = 8
const MINRES_COL_NORM_R              = 9
const MINRES_COL_NORM_R_TRUE         = 10
const MINRES_COL_ANORM               = 11
const MINRES_COL_ACOND               = 12
const MINRES_COL_ENERGY              = 13
const MINRES_COL_ARNORM              = 14
const MINRES_COL_ORTHO_LOSS          = 15
const MINRES_COL_KRYLOV_BLOCK        = 16
const MINRES_NUM_COLS                = 16

# MINRES setup vector FMA:
#   threaded_dot(r2, y, reduction_workspace) (β₁²): n
# Total setup: n
#
# Per-iteration vector FMA for MINRES:
#   v .= s .* y (scale): n
#   y .-= (beta/oldb) .* r1 (axpy): n  [skipped iter 1]
#   threaded_dot(v, y, reduction_workspace) (alfa): n
#   y .-= (alfa/beta) .* r2 (axpy): n
#   r1 .= r2, r2 .= y (2 copies): 2n
#   threaded_dot(r2, y, reduction_workspace) (β²): n
#   w1 .= w2, w2 .= w (2 copies): 2n
#   w .= (v .- oldeps.*w1 .- delta.*w2) .* denom (fused): 3n
#   x .+= phi .* w (axpy): n
# Total: 13n per iteration (12n at iteration 1, 13n thereafter)
# Scalar native crossings charge certification SpMVs separately in cum_base.
# This replacement constant is reserved for future recovery restarts.
const MINRES_VECTOR_FMA_PER_ITER = 13
const MINRES_VECTOR_FMA_SETUP = 1
const MINRES_VECTOR_FMA_RESIDUAL_REPLACEMENT = 1

const MINRES_COLUMN_NAMES = [
    "iteration",
    "relative_residual",
    "relative_normwise_2",
    "relative_error",
    "cumulative_base_fma",
    "cumulative_prec_fma",
    "cumulative_vector_fma",
    "phi",
    "norm_r",
    "norm_r_true",
    "anorm",
    "acond",
    "energy",
    "arnorm",
    "orthogonality_loss",
    "krylov_block",
]

"""
    minres(A, b, seed, spectral_norm, preconditioner,
           maxiters, raw_data_out, convergence_data_out;
           work_limit=-1)

Solves `Ax = b` using the Preconditioned Minimal Residual (MINRES) method.

MINRES minimizes the 2-norm of the residual at each step (unlike CG which
minimizes the A-norm of the error). It uses the Lanczos process to build an
orthogonal basis and applies Givens rotations to maintain the QR factorization
of the Lanczos tridiagonal.

The tridiagonal coefficients (alfa, beta) are streamed in numbered blocks
alongside the iteration data. Each correction restart has a separate manifest
for post-hoc Ritz value / spectral analysis.

# Arguments
- `A::SparseMatrixCSC`: The system matrix (symmetric, typically SPD).
- `b::Vector`: Right-hand side vector.
- `seed::Vector`: Known solution for error tracking.
- `spectral_norm::Float64`: Spectral norm of `A`, used for normwise backward error.
- `preconditioner::Preconditioner`: The preconditioner to apply.
- `maxiters::Int64`: Maximum number of iterations.
- `work_limit::Int64`: Maximum cumulative FMA before termination. -1 for no limit.
- `correction_restart_normwise_tolerance::Float64`: A failed native
  certification whose global normwise backward error is below
  `min(convergence_tolerance, correction_restart_normwise_tolerance)`
  restarts MINRES on the explicitly recomputed global residual while keeping
  all diagnostics relative to the original system. Eligibility is checked at
  every charged pending audit.
- `raw_data_out::String`: Logical .npy.zstd path; v2 blocks are written to its sibling directory.
- `convergence_data_out::String`: Path prefix for convergence threshold CSVs.

# Returns
- `SolverResult`: Contains `status::SolverStatus` and `iterations::Int`.

# References
- Paige & Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
  SIAM J. Numer. Anal. 12(4), 1975.
- Choi, Iterative Methods for Singular Linear Equations and Least-Squares
  Problems, PhD thesis, Stanford, 2006.
"""
function _minres_with_progress(
    A::SparseMatrixCSC,
    b::Vector{Float64},
    seed::Vector{Float64},
    spectral_norm::Float64,
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
    convergence_tolerance::Float64 = minimum_tolerance,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    write_solution::Union{Nothing,String} = nothing,
    _experimental_final_iterate::Union{Nothing,Vector{Float64}} = nothing,
    _diagnostic_block_rows::Int = DIAGNOSTIC_BLOCK_ROWS,
    progress = NumericalProgress(NOOP_PHASE_CALLBACK),
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    n = size(A, 1)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    spmv_plan = _spmv_plan(A)
    reduction_workspace = FusedReductionWorkspace()
    base_fma = UInt64(nnz(A))
    norm_b = threaded_norm(b)
    norm_seed = threaded_norm(seed)

    # Preallocate iteration data
    data = DiagnosticBlockWriter(raw_data_out, maxiters + 1, MINRES_COLUMN_NAMES;
        block_rows=_diagnostic_block_rows, callback=progress.callback)

    # Tridiagonal coefficients for post-hoc Ritz analysis

    # Working vectors
    x = zeros(n)
    # Before recovery, updates target x directly. Allocate correction storage lazily.
    correction_base = nothing
    update_x = x
    correction_assembly_steps = 0
    r2 = copy(b)               # r_0 = b (since x_0 = 0)
    r1 = copy(r2)
    y = zeros(n)
    v = zeros(n)
    w = zeros(n)
    w1 = zeros(n)
    w2 = zeros(n)
    Ax = zeros(n)
    r_true = similar(b)
    Mv_prev = zeros(n)         # for orthogonality loss tracking

    # Initial preconditioner application: y = M^{-1} r_0
    cum_prec = preconditioner.LinearOperator(y, r2)
    beta1_sq = threaded_dot(r2, y, reduction_workspace)      # β_1² = r_0' M^{-1} r_0
    if beta1_sq <= 0.0 || !isfinite(beta1_sq)
        return _initial_breakdown_result(data, b, seed, cum_prec,
            Float64(MINRES_VECTOR_FMA_SETUP) * n, policy, convergence_data_out,
            raw_data_out, progress, convergence_tolerance, correction_restart_normwise_tolerance)
    end
    beta1 = sqrt(beta1_sq)

    # Initialize Lanczos/Givens state
    oldb = 0.0
    beta = beta1
    dbar = 0.0
    epsln = 0.0
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0.0
    tnorm2 = 0.0
    cs = -1.0
    sn = 0.0
    gmax = 0.0
    gmin = typemax(Float64)

    # M v_1 = r_0 / β_1
    Mv_prev .= r2 ./ beta1

    # Record iteration 0
    data[1, MINRES_COL_ITER]                = 0.0
    data[1, MINRES_COL_RELATIVE_RESIDUAL]   = 1.0
    data[1, MINRES_COL_RELATIVE_NORMWISE_2] = 1.0
    data[1, MINRES_COL_ERROR]               = 1.0
    data[1, MINRES_COL_CUM_BASE_FMA]        = 0.0
    data[1, MINRES_COL_CUM_PREC_FMA]        = cum_prec
    data[1, MINRES_COL_CUM_VECTOR_FMA]      = Float64(MINRES_VECTOR_FMA_SETUP) * n
    data[1, MINRES_COL_PHI]                 = 0.0
    data[1, MINRES_COL_NORM_R]              = norm_b  # phibar = beta1 ≈ ||r||_M, but actual ||r|| = ||b||
    data[1, MINRES_COL_NORM_R_TRUE]         = norm_b
    data[1, MINRES_COL_ANORM]               = 0.0
    data[1, MINRES_COL_ACOND]               = 0.0
    data[1, MINRES_COL_ENERGY]              = 0.0
    data[1, MINRES_COL_ARNORM]              = 0.0
    data[1, MINRES_COL_ORTHO_LOSS]          = 0.0
    data[1, MINRES_COL_KRYLOV_BLOCK]        = 0.0

    iter = 0
    nrows = 1
    status = MAX_ITERS
    final_relative_residual = 1.0
    cum_base = 0.0
    cum_vector = Float64(MINRES_VECTOR_FMA_SETUP) * n
    local_iter = 0
    residual_replacements = 0
    residual_replacement_records = Dict{String,Any}[]
    correction_restarts = 0
    correction_restart_records = Dict{String,Any}[]
    tridiag_block_index = 0
    tridiag_stream = CoefficientStream(raw_data_out, maxiters, _diagnostic_block_rows;
        callback=progress.callback, source_attempt=basename(data.attempt))
    physical_true_residual_evals = 0
    certification_true_residual_evals = 0
    certification_check_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    true_residual_used_for_decision_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    certification_check_count = 0
    false_native_crossings = 0
    native_candidate_iter = nothing
    native_candidate_true_relres = nothing
    native_candidate_normwise = nothing
    certification_records = Dict{String,Any}[]
    certification_pending = false
    # Current policy allows one scalar correction restart. Additional false
    # native crossings after that continue the same correction recurrence.
    correction_restart_consumed = false

    Threads.atomic_xchg!(KILL_FLAG, false)
    t0 = time()

    try
    for i in 1:maxiters
        _prepare_diagnostic_row!(data, i + 1, maxiters + 1)
        phase_reporting_enabled(progress.callback) &&
        _report_solver_progress!(progress, data, nrows, (iteration=MINRES_COL_ITER, native=MINRES_COL_NORM_R, true_rr=MINRES_COL_RELATIVE_RESIDUAL, normwise=MINRES_COL_RELATIVE_NORMWISE_2, block=MINRES_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; policy=policy)
        iter = i
        row = i + 1
        nrows = row
        local_iter += 1

        # Lanczos step
        s = 1.0 / beta
        v .= s .* y                        # v_k = y / β_k (normalized Lanczos vector)
        threaded_spmv!(y, A', v, spmv_plan)                      # y = A v_k

        if local_iter > 1
            y .-= (beta / oldb) .* r1      # y -= (β_k / β_{k-1}) r_{k-1}
        end

        alfa = threaded_dot(v, y, reduction_workspace)                    # δ_k = v_k' A v_k

        y .-= (alfa / beta) .* r2          # y -= (δ_k / β_k) r_k
        r1 .= r2
        r2 .= y

        cum_prec += preconditioner.LinearOperator(y, r2)  # y = M^{-1} r_{k+1}
        oldb = beta
        beta_sq = threaded_dot(r2, y, reduction_workspace)               # β_{k+1}² = r_{k+1}' M^{-1} r_{k+1}

        lanczos_exhausted = iszero(beta_sq) && all(iszero, r2)
        if (beta_sq <= 0.0 && !lanczos_exhausted) || !isfinite(beta_sq)
            _append_coefficients!(tridiag_stream, alfa, beta_sq)
            status = BREAKDOWN
            # Still record this iteration's data before breaking
            threaded_spmv!(Ax, A', x, spmv_plan)
            physical_true_residual_evals += 1
            d = _fused_diagnostics!(r_true, Ax, b, seed, x, nothing, reduction_workspace, true)
            norm_r_true = d.norm_true
            norm_x = d.norm_x
            relative_residual, relative_normwise_2 =
                compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)
            error_val = d.norm_error / norm_seed
            cum_base = Float64(base_fma) * (i + residual_replacements + certification_true_residual_evals)
            cum_vector = Float64(
                MINRES_VECTOR_FMA_SETUP +
                MINRES_VECTOR_FMA_PER_ITER * i +
                TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * certification_true_residual_evals +
                MINRES_VECTOR_FMA_SETUP * correction_restarts +
                correction_assembly_steps +
                MINRES_VECTOR_FMA_RESIDUAL_REPLACEMENT * residual_replacements
            ) * n
            final_relative_residual = relative_residual
            data[row, MINRES_COL_ITER]                = Float64(i)
            data[row, MINRES_COL_RELATIVE_RESIDUAL]   = relative_residual
            data[row, MINRES_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
            data[row, MINRES_COL_ERROR]               = error_val
            data[row, MINRES_COL_CUM_BASE_FMA]        = cum_base
            data[row, MINRES_COL_CUM_PREC_FMA]        = cum_prec
            data[row, MINRES_COL_CUM_VECTOR_FMA]      = cum_vector
            data[row, MINRES_COL_PHI]                 = 0.0
            data[row, MINRES_COL_NORM_R]              = phibar
            data[row, MINRES_COL_NORM_R_TRUE]         = norm_r_true
            data[row, MINRES_COL_ANORM]               = sqrt(tnorm2)
            data[row, MINRES_COL_ACOND]               = gmax > 0 ? gmax / gmin : 0.0
            data[row, MINRES_COL_ENERGY]              = 0.5 * d.xAx - d.xb
            data[row, MINRES_COL_ARNORM]              = 0.0
            data[row, MINRES_COL_ORTHO_LOSS]          = 0.0
            data[row, MINRES_COL_KRYLOV_BLOCK]        = Float64(tridiag_block_index)
            break
        end

        beta = sqrt(beta_sq)
        _append_coefficients!(tridiag_stream, alfa, beta)

        # v_{k+1}' M v_k = threaded_dot(y, Mv_prev, reduction_workspace) / β_{k+1}
        ortho_loss = lanczos_exhausted ? 0.0 : threaded_dot(y, Mv_prev, reduction_workspace) / beta

        # Update Mv_prev for next iteration: M v_{k+1} = r2 / β_{k+1}
        if !lanczos_exhausted
            Mv_prev .= r2 ./ beta
        end

        # Givens rotation and solution update
        tnorm2 += alfa^2 + oldb^2 + beta^2
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        root = sqrt(gbar^2 + dbar^2)
        arnorm = phibar * root              # ≈ ||A' r_k||

        gamma = sqrt(gbar^2 + beta^2)
        gamma = max(gamma, eps(1.0))
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        denom = 1.0 / gamma

        w1 .= w2
        w2 .= w
        w .= (v .- oldeps .* w1 .- delta .* w2) .* denom

        updated_norm_x = _fused_update_norm!(update_x, phi, w, reduction_workspace, nothing, nothing, correction_base, x)
        if correction_base !== nothing
            correction_assembly_steps += 1
        end

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z

        anorm_est = sqrt(tnorm2)
        acond_est = gmax / gmin

        # True residual
        threaded_spmv!(Ax, A', x, spmv_plan)
        physical_true_residual_evals += 1
        d = _fused_diagnostics!(r_true, Ax, b, seed, x, nothing,
            reduction_workspace, false)
        norm_r_true = d.norm_true
        norm_x = updated_norm_x
        error_val = d.norm_error / norm_seed
        energy = 0.5 * d.xAx - d.xb
        relative_residual, relative_normwise_2 =
            compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)

        native_relative_residual = abs(phibar) / norm_b
        native_candidate = native_relative_residual <= convergence_tolerance
        new_native_candidate = native_candidate && !certification_pending
        certification_iter = certification_pending || native_candidate
        if certification_iter
            certification_true_residual_evals += 1
            certification_check_count += 1
            _grow_diagnostic_vector!(certification_check_iters, certification_check_count, maxiters)
            _grow_diagnostic_vector!(true_residual_used_for_decision_iters, certification_check_count, maxiters)
            certification_check_iters[certification_check_count] = i
            true_residual_used_for_decision_iters[certification_check_count] = i
            if native_candidate && native_candidate_iter === nothing
                native_candidate_iter = i
                native_candidate_true_relres = relative_residual
                native_candidate_normwise = relative_normwise_2
            end
        end

        cum_base = Float64(base_fma) * (
            i + residual_replacements + certification_true_residual_evals
        )
        cum_vector = Float64(
            MINRES_VECTOR_FMA_SETUP +
            MINRES_VECTOR_FMA_PER_ITER * i +
            TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA * certification_true_residual_evals +
            correction_assembly_steps +
            MINRES_VECTOR_FMA_SETUP * correction_restarts +
            MINRES_VECTOR_FMA_RESIDUAL_REPLACEMENT * residual_replacements
        ) * n
        _observe_native_candidate!(policy, native_candidate, i, cum_base + cum_prec + cum_vector)
        final_relative_residual = relative_residual

        data[row, MINRES_COL_ITER]                = Float64(i)
        data[row, MINRES_COL_RELATIVE_RESIDUAL]   = relative_residual
        data[row, MINRES_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
        data[row, MINRES_COL_ERROR]               = error_val
        data[row, MINRES_COL_CUM_BASE_FMA]        = cum_base
        data[row, MINRES_COL_CUM_PREC_FMA]        = cum_prec
        data[row, MINRES_COL_CUM_VECTOR_FMA]      = cum_vector
        data[row, MINRES_COL_PHI]                 = phi
        data[row, MINRES_COL_NORM_R]              = phibar  # MINRES scalar residual estimate
        data[row, MINRES_COL_NORM_R_TRUE]         = norm_r_true
        data[row, MINRES_COL_ANORM]               = anorm_est
        data[row, MINRES_COL_ACOND]               = acond_est
        data[row, MINRES_COL_ENERGY]              = energy
        data[row, MINRES_COL_ARNORM]              = arnorm
        data[row, MINRES_COL_ORTHO_LOSS]          = ortho_loss
        data[row, MINRES_COL_KRYLOV_BLOCK]        = Float64(tridiag_block_index)

        # NaN check on key scalars
        if isnan(alfa) || isnan(beta) || isnan(phibar) || isnan(gamma)
            status = NAN_DETECTED
            break
        end

        if KILL_FLAG[]
            status = KILLED
            break
        end

        certified_this_iter = false
        if certification_iter
            certified = relative_residual <= convergence_tolerance &&
                        relative_normwise_2 <= convergence_tolerance
            if solver_mode == :bounded_continuation
                cap = policy.continuation_work_limit
                within_budget = (work_limit <= 0 || cum_base + cum_prec + cum_vector <= work_limit) &&
                    (cap === nothing || cum_base + cum_prec + cum_vector <= cap)
                certified &= within_budget
            end
            certified_this_iter = certified
            correction_restart_this_iter =
                solver_mode == :certified &&
                !certified &&
                certification_iter &&
                !correction_restart_consumed &&
                relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)
            correction_restart_action = correction_restart_this_iter ? "correction_restart" :
                                        "continue_recurrence"
            if correction_restart_this_iter
                # Restart on r_true; successful setup freezes x as the base.
                # Later increments accumulate separately before forming global x.
                r2 .= r_true
                r1 .= r2
                cum_prec += Float64(preconditioner.LinearOperator(y, r2))
                beta1_sq = threaded_dot(r2, y, reduction_workspace)
                cum_vector += Float64(MINRES_VECTOR_FMA_SETUP) * n
                data[row, MINRES_COL_CUM_PREC_FMA] = cum_prec
                data[row, MINRES_COL_CUM_VECTOR_FMA] = cum_vector
                if work_limit > 0 && (cum_base + cum_prec + cum_vector) > work_limit
                    correction_restart_action = "correction_restart_work_limit"
                    status = WORK_LIMIT
                elseif beta1_sq <= 0.0
                    correction_restart_action = "correction_restart_breakdown"
                    status = BREAKDOWN
                else
                    restart_index = correction_restarts + 1
                    closed_tridiag_path = _close_coefficients!(tridiag_stream; next_iteration=i+1)
                    tridiag_block_index += 1
                    _record_correction_restart!(
                        correction_restart_records,
                        raw_data_out,
                        restart_index,
                        i,
                        native_relative_residual,
                        relative_residual,
                        relative_normwise_2,
                        abs(phibar),
                        norm_r_true,
                        x;
                        closed_tridiag_path = closed_tridiag_path,
                        next_tridiag_path = _tridiag_block_path(raw_data_out, tridiag_block_index),
                    )

                    beta1 = sqrt(beta1_sq)
                    oldb = 0.0
                    beta = beta1
                    dbar = 0.0
                    epsln = 0.0
                    phibar = beta1
                    rhs1 = beta1
                    rhs2 = 0.0
                    tnorm2 = 0.0
                    cs = -1.0
                    sn = 0.0
                    gmax = 0.0
                    gmin = typemax(Float64)
                    @. Mv_prev = r2 / beta1
                    fill!(w, 0.0)
                    fill!(w1, 0.0)
                    fill!(w2, 0.0)
                    local_iter = 0
                    correction_restarts += 1
                    correction_base = copy(x)
                    update_x = zeros(n)
                    correction_restart_consumed = true
                end
            end
            if new_native_candidate || certified || correction_restart_this_iter
                event = new_native_candidate ? "native_candidate" : "certification_audit"
                action = certified ? "accept_convergence" :
                         correction_restart_action
                _record_certification_check!(
                    certification_records,
                    i,
                    event,
                    native_relative_residual,
                    relative_residual,
                    relative_normwise_2,
                    abs(phibar),
                    norm_r_true;
                    certified = certified,
                    action = action,
                )
            end
            if certified
                certification_pending = false
            else
                if new_native_candidate
                    false_native_crossings += 1
                end
                certification_pending = true
            end
        end

        if status == BREAKDOWN || status == WORK_LIMIT
            break
        end

        stop_status = _policy_stop_status(policy, cum_base + cum_prec + cum_vector, certified_this_iter)
        if stop_status !== nothing
            status = stop_status
            break
        end
        # A successful correction setup resets local_iter and supplies a new
        # basis. Otherwise an exhausted space cannot take another step.
        if lanczos_exhausted && local_iter > 0
            status = BREAKDOWN
            break
        end
    end
    catch e
        e isa InterruptException || rethrow(e)
        status = KILLED
    end

    wall_time = time() - t0
    _experimental_final_iterate === nothing || copyto!(_experimental_final_iterate, x)
    _report_solver_progress!(progress, data, nrows, (iteration=MINRES_COL_ITER, native=MINRES_COL_NORM_R, true_rr=MINRES_COL_RELATIVE_RESIDUAL, normwise=MINRES_COL_RELATIVE_NORMWISE_2, block=MINRES_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; final=true, status=string(status), policy=policy)
    total_fma = cum_base + cum_prec + cum_vector
    resize!(certification_check_iters, certification_check_count)
    resize!(true_residual_used_for_decision_iters, certification_check_count)

    # Finalize raw blocks, then scan them without concatenating the history.
    output_operation = begin_reported_phase(progress.callback, "result_write")
    try
        used = _finish_diagnostic_data!(data, nrows;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)

        # Write column metadata
        columns_path = replace(raw_data_out, r"\.[^/]*$" => "_columns.json")
        open(columns_path, "w") do f
            JSON3.write(f, MINRES_COLUMN_NAMES)
        end

        # Write tridiagonal coefficients for post-hoc Ritz analysis
        final_tridiag_path = _close_coefficients!(tridiag_stream;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)
        _clear_unwritten_next_tridiag!(residual_replacement_records, final_tridiag_path)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_tridiag_path)

        # Extract convergence thresholds post-hoc
        _write_convergence(
            convergence_data_out * "_relative_residual.csv",
            used,
            MINRES_COL_RELATIVE_RESIDUAL,
        )
        _write_convergence(
            convergence_data_out * "_relative_normwise_2.csv",
            used,
            MINRES_COL_RELATIVE_NORMWISE_2,
        )
        _write_certification_convergence(
            convergence_data_out * "_certification.csv",
            used,
            MINRES_COL_NORM_R,
            norm_b,
            certification_records,
            certification_check_iters,
            solver_policy = :scalar,
            n = n,
            base_fma = base_fma,
            correction_restart_normwise_tolerance = correction_restart_normwise_tolerance,
            solver_mode = solver_mode,
            continuation_work_factor = continuation_work_factor,
            work_limit = work_limit,
        )

        if write_solution !== nothing && status == CONVERGED
            _write_solution_vec(write_solution, x)
        end
    catch e
        finish_reported_phase(output_operation; error=e)
        rethrow()
    end
    finish_reported_phase(output_operation)

    return SolverResult(
        status,
        iter,
        wall_time,
        total_fma,
        final_relative_residual,
        residual_replacements,
        residual_replacement_records,
        correction_restarts,
        correction_restart_records,
        physical_true_residual_evals,
        physical_true_residual_evals - certification_true_residual_evals,
        certification_true_residual_evals,
        certification_check_iters,
        true_residual_used_for_decision_iters,
        false_native_crossings,
        native_candidate_iter,
        native_candidate_true_relres,
        native_candidate_normwise,
        certification_records,
        _solver_policy_metadata(policy; status=status, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance),
    )
end

# ─── Preconditioned Conjugate Residual (new API) ──────────────────────

const PCR_COL_ITER                = 1
const PCR_COL_RELATIVE_RESIDUAL   = 2
const PCR_COL_RELATIVE_NORMWISE_2 = 3
const PCR_COL_ERROR               = 4
const PCR_COL_CUM_BASE_FMA        = 5
const PCR_COL_CUM_PREC_FMA        = 6
const PCR_COL_CUM_VECTOR_FMA      = 7
const PCR_COL_ALPHA               = 8
const PCR_COL_NORM_R              = 9
const PCR_COL_NORM_R_TRUE         = 10
const PCR_COL_ZAZ                 = 11
const PCR_COL_APAP                = 12
const PCR_COL_ENERGY              = 13
const PCR_COL_ORTHO_LOSS          = 14
const PCR_COL_DIAG_ABS_FLAG       = 15
const PCR_COL_KRYLOV_BLOCK       = 16
const PCR_COL_AP_MINV_AP         = 17
const PCR_NUM_COLS               = 17

# Per-iteration vector FMA for PCR:
#   threaded_dot(z, Az, reduction_workspace) (dot): n                       — zAz_new
#   p .= z .+ β .* p (axpy): n            — search direction [skipped iter 1]
#   Ap .= Az .+ β .* Ap (axpy): n         — A*search direction [skipped iter 1]
#   threaded_dot(Ap, MinvAp, reduction_workspace) (dot): n
#   x .+= α .* p (axpy): n
#   r .-= α .* Ap (axpy): n
#   z .-= α .* MinvAp (axpy): n
# Total: 7n per iteration, 4n on each block's first step (reuse z'Az and
# skip both direction updates). Setup computes z'Az once.
# Correction restart charges the extra Az = A*z SpMV, z'Az dot, and r = r_true.
# Orthogonality loss, the unweighted Ap'Ap diagnostic, and
# tridiagonal recovery are diagnostic overhead, NOT counted in vector FMA.
const PCR_VECTOR_FMA_PER_ITER = 7
const PCR_VECTOR_FMA_SETUP = 1
const PCR_VECTOR_FMA_RESIDUAL_REPLACEMENT = 2

const PCR_COLUMN_NAMES = [
    "iteration",
    "relative_residual",
    "relative_normwise_2",
    "relative_error",
    "cumulative_base_fma",
    "cumulative_prec_fma",
    "cumulative_vector_fma",
    "alpha",
    "norm_r",
    "norm_r_true",
    "zAz",
    "ApAp",
    "energy",
    "orthogonality_loss",
    "diag_abs_flag",
    "krylov_block",
    "ApMinvAp",
]

"""
    pcr(A, b, seed, spectral_norm, preconditioner;
        maxiters, work_limit=-1, raw_data_out, convergence_data_out)

Solves `Ax = b` using Preconditioned Conjugate Residual.

For fixed SPD preconditioning M, CR minimizes the M^-1-weighted residual norm.
The step is (z'Az)/(Ap'M^-1 Ap), with z = M^-1 r maintained recursively.
The unpreconditioned recursive residual still schedules true-residual certification.
With identity preconditioning this reduces to ordinary CR. Unlike MINRES, this
short recurrence can break down on symmetric indefinite systems (z'Az can vanish).

Iteration data is written in bounded, numbered compressed NPY blocks.

# Arguments
- `A::SparseMatrixCSC`: The system matrix (symmetric).
- `b::Vector`: Right-hand side vector.
- `seed::Vector`: Known solution for error tracking.
- `spectral_norm::Float64`: Spectral norm of `A`, used for normwise backward error.
- `preconditioner::Preconditioner`: The preconditioner to apply.
- `maxiters::Int64`: Maximum number of iterations.
- `work_limit::Int64`: Maximum cumulative FMA before termination. -1 for no limit.
- `convergence_tolerance::Float64`: Certified true-residual target.
- `correction_restart_normwise_tolerance::Float64`: A failed native
  certification starts one correction solve only when
  `relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)`.
- `raw_data_out::String`: Logical .npy.zstd path; v2 blocks are written to its sibling directory.
- `convergence_data_out::String`: Path prefix for convergence threshold CSVs.

# Returns
- `SolverResult`: Contains `status::SolverStatus` and `iterations::Int`.
"""
function _pcr_with_progress(
    A::SparseMatrixCSC,
    b::Vector{Float64},
    seed::Vector{Float64},
    spectral_norm::Float64,
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
    convergence_tolerance::Float64 = minimum_tolerance,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    write_solution::Union{Nothing,String} = nothing,
    _experimental_final_iterate::Union{Nothing,Vector{Float64}} = nothing,
    _diagnostic_block_rows::Int = DIAGNOSTIC_BLOCK_ROWS,
    progress = NumericalProgress(NOOP_PHASE_CALLBACK),
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    n = size(A, 1)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    spmv_plan = _spmv_plan(A)
    reduction_workspace = FusedReductionWorkspace()
    base_fma = UInt64(nnz(A))

    norm_b = threaded_norm(b)
    norm_seed = threaded_norm(seed)

    # Reuse a fixed raw-history buffer; row 1 is iteration 0.
    data = DiagnosticBlockWriter(raw_data_out, maxiters + 1, PCR_COLUMN_NAMES;
        block_rows=_diagnostic_block_rows, callback=progress.callback)

    # Working vectors
    x = zeros(n)
    # Before recovery, updates target x directly. Allocate correction storage lazily.
    correction_base = nothing
    update_x = x
    r = copy(b)
    z = zeros(n)
    Ax = zeros(n)
    r_true = similar(b)

    # CR-specific: Az (A times preconditioned residual), search directions
    Az = zeros(n)
    Az_prev = similar(b)  # for orthogonality loss diagnostic

    # Tridiagonal recovery from CR scalars (diagnostic only)
    # δ_k = 1/α_k + β_{k-1}/α_{k-1}
    # γ_k = sqrt(|zAz_k|) / (α_{k-1} * sqrt(|zAz_{k-1}|))
    # This reconstruction concerns the preconditioned operator, not A alone.
    α_prev = 0.0

    # Initial preconditioner application
    cum_prec = preconditioner.LinearOperator(z, r)

    # Az = A * z (using A' since A is symmetric stored as CSC)
    threaded_spmv!(Az, A', z, spmv_plan)

    p = copy(z)
    Ap = copy(Az)
    MinvAp = similar(b)

    zAz = threaded_dot(z, Az, reduction_workspace)
    ApAp = threaded_dot(Ap, Ap, reduction_workspace)
    # No step or weighted denominator has been computed at iteration zero.
    α = NaN

    # Record iteration 0
    norm_r = threaded_norm(r)
    data[1, PCR_COL_ITER]                = 0.0
    data[1, PCR_COL_RELATIVE_RESIDUAL]   = norm_r / norm_b
    data[1, PCR_COL_RELATIVE_NORMWISE_2] = norm_r / (norm_b + spectral_norm * threaded_norm(x))
    data[1, PCR_COL_ERROR]               = 1.0
    data[1, PCR_COL_CUM_BASE_FMA]        = Float64(base_fma)
    data[1, PCR_COL_CUM_PREC_FMA]        = cum_prec
    data[1, PCR_COL_CUM_VECTOR_FMA]      = Float64(PCR_VECTOR_FMA_SETUP) * n
    data[1, PCR_COL_ALPHA]               = α
    data[1, PCR_COL_NORM_R]              = norm_r
    data[1, PCR_COL_NORM_R_TRUE]         = norm_r  # at iteration 0, r = b - A*0 = b = r_true
    data[1, PCR_COL_ZAZ]                 = zAz
    data[1, PCR_COL_APAP]               = ApAp
    data[1, PCR_COL_ENERGY]              = 0.0  # ½x'Ax - x'b = 0 at x=0
    data[1, PCR_COL_ORTHO_LOSS]          = 0.0  # no previous iteration
    data[1, PCR_COL_DIAG_ABS_FLAG]       = 0.0

    data[1, PCR_COL_KRYLOV_BLOCK] = 0.0
    data[1, PCR_COL_AP_MINV_AP] = NaN

    iter = 0
    nrows = 1  # row 1 (iteration 0) is already filled
    status = MAX_ITERS
    final_relative_residual = norm_r / norm_b
    cum_base = Float64(base_fma)
    cum_vector = Float64(PCR_VECTOR_FMA_SETUP) * n
    local_iter = 0
    residual_replacements = 0
    residual_replacement_records = Dict{String,Any}[]
    correction_restarts = 0
    correction_restart_records = Dict{String,Any}[]
    tridiag_block_index = 0
    tridiag_stream = CoefficientStream(raw_data_out, maxiters, _diagnostic_block_rows;
        callback=progress.callback, source_attempt=basename(data.attempt))
    physical_true_residual_evals = 0
    certification_true_residual_evals = 0
    certification_check_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    true_residual_used_for_decision_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    certification_check_count = 0
    false_native_crossings = 0
    native_candidate_iter = nothing
    native_candidate_true_relres = nothing
    native_candidate_normwise = nothing
    certification_records = Dict{String,Any}[]
    certification_pending = false
    # One correction restart is allowed per solve under the current policy.
    correction_restart_consumed = false

    Threads.atomic_xchg!(KILL_FLAG, false)
    t0 = time()

    try
    for i in 1:maxiters
        _prepare_diagnostic_row!(data, i + 1, maxiters + 1)
        phase_reporting_enabled(progress.callback) &&
        _report_solver_progress!(progress, data, nrows, (iteration=PCR_COL_ITER, native=PCR_COL_NORM_R, true_rr=PCR_COL_RELATIVE_RESIDUAL, normwise=PCR_COL_RELATIVE_NORMWISE_2, block=PCR_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; policy=policy)
        iter = i
        row = i + 1
        nrows = row
        local_iter += 1

        zAz_new = local_iter == 1 ? zAz : threaded_dot(z, Az, reduction_workspace)
        local_iter > 1 && (cum_vector += Float64(n))

        if local_iter > 1
            β = zAz_new / zAz
            p .= z .+ β .* p
            Ap .= Az .+ β .* Ap
            cum_vector += 2.0 * n
        else
            β = 0.0
        end

        ApAp = threaded_dot(Ap, Ap, reduction_workspace)
        cum_prec += preconditioner.LinearOperator(MinvAp, Ap)
        ApMinvAp = threaded_dot(Ap, MinvAp, reduction_workspace)
        cum_vector += Float64(n)
        valid_step = isfinite(zAz_new) && zAz_new != 0.0 &&
            isfinite(ApMinvAp) && ApMinvAp > 0.0
        α = valid_step ? zAz_new / ApMinvAp : NaN
        valid_step &= isfinite(α)

        # --- Diagnostic: orthogonality loss and tridiagonal recovery ---
        diag_abs_flag = 0.0
        ortho_loss = 0.0
        updated_norm_x = valid_step ? 0.0 : threaded_norm(x)
        if valid_step
            # Retain the historical unweighted image-correlation diagnostic.
            # With preconditioning this is not the CR orthogonality metric.
            if local_iter > 1
                if zAz_new < 0.0 || zAz < 0.0
                    diag_abs_flag = 1.0
                end
                ortho_loss = (threaded_dot(Az, Az_prev, reduction_workspace)) / sqrt(abs(zAz_new) * abs(zAz))
            end

            # Tridiagonal recovery: δ_k = 1/α_k + β_{k-1}/α_{k-1}
            if local_iter == 1
                tridiag_diagonal = 1.0 / α
            else
                tridiag_diagonal = 1.0 / α + β / α_prev
            end

            previous_tridiag_beta = 0.0
            # Off-diagonal: -γ_k = -sqrt(|zAz_k|) / (α_{k-1} * sqrt(|zAz_{k-1}|))
            if local_iter > 1
                if zAz_new < 0.0 || zAz < 0.0
                    diag_abs_flag = 1.0
                end
                previous_tridiag_beta = -sqrt(abs(zAz_new)) / (α_prev * sqrt(abs(zAz)))
            end

            _append_delayed_tridiagonal!(tridiag_stream, tridiag_diagonal, previous_tridiag_beta)
            α_prev = α
            Az_prev .= Az  # save Az before update (diagnostic only)
            # --- End diagnostic ---

            updated_norm_x = _fused_update_norm!(update_x, α, p, reduction_workspace, r, Ap, correction_base, x)

            # Standard preconditioned CR: reuse M^-1 Ap, avoiding a second apply.
            z .-= α .* MinvAp
            cum_vector += 3.0 * n

            threaded_spmv!(Az, A', z, spmv_plan)
            cum_base += Float64(base_fma)

            zAz = zAz_new
        end

        # True residual
        threaded_spmv!(Ax, A', x, spmv_plan)
        physical_true_residual_evals += 1
        d = _fused_diagnostics!(r_true, Ax, b, seed, x, r,
            reduction_workspace, false)
        norm_r_true = d.norm_true
        norm_x = updated_norm_x
        error_val = d.norm_error / norm_seed
        energy = 0.5 * d.xAx - d.xb
        norm_r = d.norm_native
        relative_residual, relative_normwise_2 =
            compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)

        native_relative_residual = norm_r / norm_b
        native_candidate = native_relative_residual <= convergence_tolerance
        new_native_candidate = native_candidate && !certification_pending
        certification_iter = certification_pending || native_candidate
        if certification_iter
            certification_true_residual_evals += 1
            cum_base += Float64(base_fma)
            cum_vector += Float64(TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA) * n
            certification_check_count += 1
            _grow_diagnostic_vector!(certification_check_iters, certification_check_count, maxiters)
            _grow_diagnostic_vector!(true_residual_used_for_decision_iters, certification_check_count, maxiters)
            certification_check_iters[certification_check_count] = i
            true_residual_used_for_decision_iters[certification_check_count] = i
            if native_candidate && native_candidate_iter === nothing
                native_candidate_iter = i
                native_candidate_true_relres = relative_residual
                native_candidate_normwise = relative_normwise_2
            end
        end

        valid_step && correction_base !== nothing && (cum_vector += Float64(n))
        _observe_native_candidate!(policy, native_candidate, i, cum_base + cum_prec + cum_vector)
        final_relative_residual = relative_residual

        data[row, PCR_COL_KRYLOV_BLOCK] = Float64(tridiag_block_index)
        data[row, PCR_COL_ITER]                = Float64(i)
        data[row, PCR_COL_RELATIVE_RESIDUAL]   = relative_residual
        data[row, PCR_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
        data[row, PCR_COL_ERROR]               = error_val
        data[row, PCR_COL_CUM_BASE_FMA]        = cum_base
        data[row, PCR_COL_CUM_PREC_FMA]        = cum_prec
        data[row, PCR_COL_CUM_VECTOR_FMA]      = cum_vector
        data[row, PCR_COL_ALPHA]               = α
        data[row, PCR_COL_NORM_R]              = norm_r
        data[row, PCR_COL_NORM_R_TRUE]         = norm_r_true
        data[row, PCR_COL_ZAZ]                 = zAz_new
        data[row, PCR_COL_APAP]               = ApAp
        data[row, PCR_COL_ENERGY]              = energy
        data[row, PCR_COL_ORTHO_LOSS]          = abs(ortho_loss)
        data[row, PCR_COL_DIAG_ABS_FLAG]       = diag_abs_flag
        data[row, PCR_COL_AP_MINV_AP]          = ApMinvAp

        if !valid_step
            status = work_limit > 0 && cum_base + cum_prec + cum_vector > work_limit ?
                WORK_LIMIT : BREAKDOWN
            break
        end

        if KILL_FLAG[]
            status = KILLED
            break
        end

        certified_this_iter = false
        if certification_iter
            certified = relative_residual <= convergence_tolerance &&
                        relative_normwise_2 <= convergence_tolerance
            if solver_mode == :bounded_continuation
                cap = policy.continuation_work_limit
                within_budget = (work_limit <= 0 || cum_base + cum_prec + cum_vector <= work_limit) &&
                    (cap === nothing || cum_base + cum_prec + cum_vector <= cap)
                certified &= within_budget
            end
            certified_this_iter = certified
            correction_restart_this_iter =
                solver_mode == :certified &&
                !certified &&
                new_native_candidate &&
                !correction_restart_consumed &&
                relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)
            correction_restart_action = correction_restart_this_iter ? "correction_restart" : "continue_recurrence"
            if correction_restart_this_iter
                r .= r_true
                cum_prec += Float64(preconditioner.LinearOperator(z, r))
                threaded_spmv!(Az, A', z, spmv_plan)
                p .= z
                Ap .= Az
                zAz_restart = threaded_dot(z, Az, reduction_workspace)
                cum_base += Float64(base_fma)
                cum_vector += Float64(PCR_VECTOR_FMA_RESIDUAL_REPLACEMENT) * n
                data[row, PCR_COL_CUM_BASE_FMA] = cum_base
                data[row, PCR_COL_CUM_PREC_FMA] = cum_prec
                data[row, PCR_COL_CUM_VECTOR_FMA] = cum_vector
                if work_limit > 0 && (cum_base + cum_prec + cum_vector) > work_limit
                    correction_restart_action = "correction_restart_work_limit"
                    status = WORK_LIMIT
                elseif zAz_restart == 0.0 || !isfinite(zAz_restart)
                    correction_restart_action = "correction_restart_breakdown"
                    status = BREAKDOWN
                else
                    restart_index = correction_restarts + 1
                    closed_tridiag_path = _close_coefficients!(tridiag_stream; next_iteration=i+1)
                    tridiag_block_index += 1
                    _record_correction_restart!(
                        correction_restart_records,
                        raw_data_out,
                        restart_index,
                        i,
                        native_relative_residual,
                        relative_residual,
                        relative_normwise_2,
                        norm_r,
                        norm_r_true,
                        x;
                        closed_tridiag_path = closed_tridiag_path,
                        next_tridiag_path = _tridiag_block_path(raw_data_out, tridiag_block_index),
                    )
                    zAz = zAz_restart
                    α_prev = 0.0
                    local_iter = 0
                    correction_restarts += 1
                    correction_base = copy(x)
                    update_x = zeros(n)
                    correction_restart_consumed = true
                end
            end
            if new_native_candidate || certified
                event = new_native_candidate ? "native_candidate" : "certification_audit"
                action = certified ? "accept_convergence" : correction_restart_action
                _record_certification_check!(
                    certification_records,
                    i,
                    event,
                    native_relative_residual,
                    relative_residual,
                    relative_normwise_2,
                    norm_r,
                    norm_r_true;
                    certified = certified,
                    action = action,
                )
            end
            if certified
                certification_pending = false
            else
                new_native_candidate && (false_native_crossings += 1)
                certification_pending = true
            end
        end

        if status == BREAKDOWN || status == WORK_LIMIT
            break
        end

        stop_status = _policy_stop_status(policy, cum_base + cum_prec + cum_vector, certified_this_iter)
        if stop_status !== nothing
            status = stop_status
            break
        end
    end
    catch e
        e isa InterruptException || rethrow(e)
        status = KILLED
    end

    wall_time = time() - t0
    _experimental_final_iterate === nothing || copyto!(_experimental_final_iterate, x)
    _report_solver_progress!(progress, data, nrows, (iteration=PCR_COL_ITER, native=PCR_COL_NORM_R, true_rr=PCR_COL_RELATIVE_RESIDUAL, normwise=PCR_COL_RELATIVE_NORMWISE_2, block=PCR_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; final=true, status=string(status), policy=policy)
    total_fma = cum_base + cum_prec + cum_vector
    resize!(certification_check_iters, certification_check_count)
    resize!(true_residual_used_for_decision_iters, certification_check_count)

    # Finalize raw blocks, then scan them without concatenating the history.
    output_operation = begin_reported_phase(progress.callback, "result_write")
    try
        used = _finish_diagnostic_data!(data, nrows;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)

        # Write tridiagonal coefficients (diagnostic)
        final_tridiag_path = _close_coefficients!(tridiag_stream;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)
        _clear_unwritten_next_tridiag!(residual_replacement_records, final_tridiag_path)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_tridiag_path)

        # Write column metadata
        columns_path = replace(raw_data_out, r"\.[^/]*$" => "_columns.json")
        open(columns_path, "w") do f
            JSON3.write(f, PCR_COLUMN_NAMES)
        end

        # Extract convergence thresholds post-hoc
        _write_convergence(
            convergence_data_out * "_relative_residual.csv",
            used,
            PCR_COL_RELATIVE_RESIDUAL,
        )
        _write_convergence(
            convergence_data_out * "_relative_normwise_2.csv",
            used,
            PCR_COL_RELATIVE_NORMWISE_2,
        )
        _write_certification_convergence(
            convergence_data_out * "_certification.csv",
            used,
            PCR_COL_NORM_R,
            norm_b,
            certification_records,
            certification_check_iters,
            solver_policy = :explicit,
            n = n,
            base_fma = base_fma,
            correction_restart_normwise_tolerance = correction_restart_normwise_tolerance,
            solver_mode = solver_mode,
            continuation_work_factor = continuation_work_factor,
            work_limit = work_limit,
        )

        if write_solution !== nothing && status == CONVERGED
            _write_solution_vec(write_solution, x)
        end
    catch e
        finish_reported_phase(output_operation; error=e)
        rethrow()
    end
    finish_reported_phase(output_operation)

    return SolverResult(
        status,
        iter,
        wall_time,
        total_fma,
        final_relative_residual,
        residual_replacements,
        residual_replacement_records,
        correction_restarts,
        correction_restart_records,
        physical_true_residual_evals,
        physical_true_residual_evals - certification_true_residual_evals,
        certification_true_residual_evals,
        certification_check_iters,
        true_residual_used_for_decision_iters,
        false_native_crossings,
        native_candidate_iter,
        native_candidate_true_relres,
        native_candidate_normwise,
        certification_records,
        _solver_policy_metadata(policy; status=status, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance),
    )
end

# ─── Flexible PCG (Notay FCG) ─────────────────────────────────────────

# Column indices for FPCG iteration data (shared by fpcg1 and fpcg2)
const FPCG_COL_ITER                = 1
const FPCG_COL_RELATIVE_RESIDUAL   = 2
const FPCG_COL_RELATIVE_NORMWISE_2 = 3
const FPCG_COL_ERROR               = 4
const FPCG_COL_CUM_BASE_FMA        = 5
const FPCG_COL_CUM_PREC_FMA        = 6
const FPCG_COL_CUM_VECTOR_FMA      = 7
const FPCG_COL_ALPHA               = 8   # step size (di, ri) / (di, A di)
const FPCG_COL_NORM_R              = 9
const FPCG_COL_NORM_R_TRUE         = 10
const FPCG_COL_ENERGY              = 11
const FPCG_COL_ORTHO_LOSS          = 12  # |d_i' A d_old| / (||d_i||_A ||d_old||_A), beyond ortho window
const FPCG_COL_ORTHO_COEFF_1       = 13  # (w, A d_{i-1}) / (d_{i-1}, A d_{i-1})
const FPCG_COL_ORTHO_COEFF_2       = 14  # (w, A d_{i-2}) / (d_{i-2}, A d_{i-2}) — diagnostic in fpcg1, used in fpcg2
const FPCG_COL_CORRECTION_NORM     = 15  # ||w - d|| / ||w||
const FPCG_COL_PREC_FMA_ITER       = 16  # per-apply FMA returned by Preconditioner
const FPCG_COL_DIAG_ABS_FLAG       = 17
const FPCG_COL_KRYLOV_BLOCK       = 18
const FPCG_NUM_COLS               = 18

# Per-iteration vector FMA for fpcg1 (FCG(1)):
#   (d, r) dot: n                     — for α numerator
#   (d, A*d) dot: n                   — for α denominator (reuses q = A*d)
#   x += α d axpy: n
#   r -= α q axpy: n
#   (w, A*d_prev) dot: n              — ortho coefficient 1
#   d = w - β₁ d_prev fused: n        — form new search direction
# Total: 6n per iter (5n at iter 0 since no orthogonalization)
# Correction restart charges the extra q = A*d SpMV plus one vector operation
# for r = r_true.
const FPCG1_VECTOR_FMA_PER_ITER = 6
const FPCG1_VECTOR_FMA_ITER0 = 5
const FPCG1_VECTOR_FMA_RESIDUAL_REPLACEMENT = 1

# Per-iteration vector FMA for fpcg2 (FCG(2)):
#   Same as fpcg1 plus, when orthogonalizing against 2 directions:
#   (w, A*d_{i-2}) dot: n             — ortho coefficient 2
#   extra axpy in d = w - β₁ d_{i-1} - β₂ d_{i-2}: n
# Total: 8n when 2 directions, 6n when 1 direction (restart step), 5n at iter 0
# Correction restart charges the extra q = A*d SpMV plus one vector operation
# for r = r_true.
const FPCG2_VECTOR_FMA_2DIR = 8
const FPCG2_VECTOR_FMA_1DIR = 6
const FPCG2_VECTOR_FMA_ITER0 = 5
const FPCG2_VECTOR_FMA_RESIDUAL_REPLACEMENT = 1

const FPCG_COLUMN_NAMES = [
    "iteration",
    "relative_residual",
    "relative_normwise_2",
    "relative_error",
    "cumulative_base_fma",
    "cumulative_prec_fma",
    "cumulative_vector_fma",
    "alpha",
    "norm_r",
    "norm_r_true",
    "energy",
    "orthogonality_loss",
    "ortho_coeff_1",
    "ortho_coeff_2",
    "correction_norm",
    "prec_fma_iter",
    "diag_abs_flag",
    "krylov_block",
]

"""
    fpcg1(A, b, seed, spectral_norm, preconditioner,
          maxiters, raw_data_out, convergence_data_out;
          work_limit=-1)

Solves `Ax = b` using Notay's Flexible Conjugate Gradient, FCG(1).

Orthogonalizes each new search direction against 1 previous direction
(A-conjugation). When the preconditioner is fixed SPD, this reduces to
standard CG. Accepts a `Preconditioner` whose apply returns
the per-application FMA cost.

Failed native certifications can trigger one global correction solve when
`relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)`.

# References
- Notay, Flexible Conjugate Gradients, SIAM J. Sci. Comput. 22(4), 2000.
- Golub & Ye, Inexact Preconditioned Conjugate Gradient Method with
  Inner-Outer Iteration, SIAM J. Sci. Comput. 21(4), 1999.
"""
function _fpcg1_with_progress(
    A::SparseMatrixCSC,
    b::Vector{Float64},
    seed::Vector{Float64},
    spectral_norm::Float64,
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
    convergence_tolerance::Float64 = minimum_tolerance,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    write_solution::Union{Nothing,String} = nothing,
    _experimental_final_iterate::Union{Nothing,Vector{Float64}} = nothing,
    _diagnostic_block_rows::Int = DIAGNOSTIC_BLOCK_ROWS,
    progress = NumericalProgress(NOOP_PHASE_CALLBACK),
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    n = size(A, 1)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    spmv_plan = _spmv_plan(A)
    reduction_workspace = FusedReductionWorkspace()
    base_fma = UInt64(nnz(A))
    norm_b = threaded_norm(b)
    norm_seed = threaded_norm(seed)

    # Preallocate iteration data
    data = DiagnosticBlockWriter(raw_data_out, maxiters + 1, FPCG_COLUMN_NAMES;
        block_rows=_diagnostic_block_rows, callback=progress.callback)

    # Working vectors
    x = zeros(n)
    # Before recovery, updates target x directly. Allocate correction storage lazily.
    correction_base = nothing
    update_x = x
    correction_assembly_steps = 0
    r = copy(b)
    w = zeros(n)        # preconditioned residual
    Ax = zeros(n)
    r_true = similar(b)

    # Circular buffer for directions and their A-images (size 3 for loss tracking)
    # Slot 0: d_{i}, slot 1: d_{i-1}, slot 2: d_{i-2}
    d_buf = [zeros(n) for _ in 1:3]
    q_buf = [zeros(n) for _ in 1:3]   # q = A*d for each buffered direction
    dAd_buf = zeros(3)                  # (d_k, A d_k) cached per slot

    # Initial preconditioner application
    prec_fma_iter = preconditioner.LinearOperator(w, r)
    cum_prec = Float64(prec_fma_iter)

    # d_0 = w_0 (no orthogonalization)
    cur = 1  # current slot in circular buffer (1-indexed)
    d_buf[cur] .= w
    threaded_spmv!(q_buf[cur], A', d_buf[cur], spmv_plan)
    dAd_buf[cur] = threaded_dot(d_buf[cur], q_buf[cur], reduction_workspace)

    dr = threaded_dot(d_buf[cur], r, reduction_workspace)
    α = dr / dAd_buf[cur]

    # Record iteration 0
    norm_r = threaded_norm(r)
    cum_base = Float64(base_fma)
    cum_vector = Float64(FPCG1_VECTOR_FMA_ITER0) * n

    data[1, FPCG_COL_ITER]                = 0.0
    data[1, FPCG_COL_RELATIVE_RESIDUAL]   = norm_r / norm_b
    data[1, FPCG_COL_RELATIVE_NORMWISE_2] = norm_r / (norm_b + spectral_norm * threaded_norm(x))
    data[1, FPCG_COL_ERROR]               = 1.0
    data[1, FPCG_COL_CUM_BASE_FMA]        = cum_base
    data[1, FPCG_COL_CUM_PREC_FMA]        = cum_prec
    data[1, FPCG_COL_CUM_VECTOR_FMA]      = cum_vector
    data[1, FPCG_COL_ALPHA]               = α
    data[1, FPCG_COL_NORM_R]              = norm_r
    data[1, FPCG_COL_NORM_R_TRUE]         = norm_r  # r = b at x=0
    data[1, FPCG_COL_ENERGY]              = 0.0
    data[1, FPCG_COL_ORTHO_LOSS]          = 0.0
    data[1, FPCG_COL_ORTHO_COEFF_1]       = 0.0
    data[1, FPCG_COL_ORTHO_COEFF_2]       = 0.0
    data[1, FPCG_COL_CORRECTION_NORM]     = 0.0
    data[1, FPCG_COL_PREC_FMA_ITER]       = Float64(prec_fma_iter)
    data[1, FPCG_COL_DIAG_ABS_FLAG]       = 0.0

    data[1, FPCG_COL_KRYLOV_BLOCK] = 0.0

    iter = 0
    nrows = 1
    status = MAX_ITERS
    final_relative_residual = norm_r / norm_b
    coefficient_streams = FlexibleCoefficientStreams(raw_data_out, maxiters, _diagnostic_block_rows;
        callback=progress.callback, source_attempt=basename(data.attempt))
    coefficient_c1 = 0.0
    coefficient_c2 = 0.0
    local_iter = 0
    residual_replacements = 0
    residual_replacement_records = Dict{String,Any}[]
    correction_restarts = 0
    correction_restart_records = Dict{String,Any}[]
    physical_true_residual_evals = 0
    certification_true_residual_evals = 0
    certification_check_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    true_residual_used_for_decision_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    certification_check_count = 0
    false_native_crossings = 0
    native_candidate_iter = nothing
    native_candidate_true_relres = nothing
    native_candidate_normwise = nothing
    certification_records = Dict{String,Any}[]
    certification_pending = false
    # One correction restart is allowed per solve under the current policy.
    correction_restart_consumed = false

    Threads.atomic_xchg!(KILL_FLAG, false)
    t0 = time()

    try
    for i in 1:maxiters
        _prepare_diagnostic_row!(data, i + 1, maxiters + 1)
        phase_reporting_enabled(progress.callback) &&
        _report_solver_progress!(progress, data, nrows, (iteration=FPCG_COL_ITER, native=FPCG_COL_NORM_R, true_rr=FPCG_COL_RELATIVE_RESIDUAL, normwise=FPCG_COL_RELATIVE_NORMWISE_2, block=FPCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; policy=policy)
        iter = i
        row = i + 1
        nrows = row
        local_iter += 1

        _record_flexible_coefficients!(coefficient_streams, i, α, coefficient_c1, coefficient_c2,
            threaded_dot(r, w, reduction_workspace), dr, dAd_buf[cur], threaded_norm(r), threaded_norm(w))

        # Solution and residual update using previous iteration's α and d
        updated_norm_x = _fused_update_norm!(update_x, α, d_buf[cur], reduction_workspace, r, q_buf[cur], correction_base, x)
        if correction_base !== nothing
            correction_assembly_steps += 1
        end

        # Variable preconditioner application
        prec_fma_iter = preconditioner.LinearOperator(w, r)
        cum_prec += Float64(prec_fma_iter)

        # SpMV for current direction (will be overwritten into new slot below)
        # But first, rotate buffer: new direction goes into next slot
        prev = cur
        prev2 = mod1(cur - 1, 3)  # slot for d_{i-2} (only valid if i >= 2)
        cur = mod1(cur + 1, 3)

        # Orthogonalize w against d_{i-1} (slot prev)
        wAd_prev = threaded_dot(w, q_buf[prev], reduction_workspace)
        β1 = wAd_prev / dAd_buf[prev]

        # Diagnostic: ortho coefficient against d_{i-2} (not applied)
        if local_iter >= 2
            wAd_prev2 = threaded_dot(w, q_buf[prev2], reduction_workspace)
            β2_diag = wAd_prev2 / dAd_buf[prev2]
        else
            β2_diag = 0.0
        end

        coefficient_c1 = -β1
        coefficient_c2 = 0.0

        # Form new search direction: d_i = w - β1 * d_{i-1}
        norm_w = threaded_norm(w)
        d_buf[cur] .= w .- β1 .* d_buf[prev]

        # Correction norm: ||w - d|| / ||w|| = ||β1 * d_{i-1}|| / ||w||
        correction_norm = abs(β1) * threaded_norm(d_buf[prev]) / norm_w

        # A * d_i
        threaded_spmv!(q_buf[cur], A', d_buf[cur], spmv_plan)
        dAd_buf[cur] = threaded_dot(d_buf[cur], q_buf[cur], reduction_workspace)

        # Step size
        dr = threaded_dot(d_buf[cur], r, reduction_workspace)
        α = dr / dAd_buf[cur]

        # Orthogonality loss: check against d_{i-2} (one beyond ortho window)
        diag_abs_flag = 0.0
        if local_iter >= 2
            dAd_old = threaded_dot(d_buf[cur], q_buf[prev2], reduction_workspace)
            if dAd_buf[cur] < 0.0 || dAd_buf[prev2] < 0.0
                diag_abs_flag = 1.0
            end
            ortho_loss = abs(dAd_old) / sqrt(abs(dAd_buf[cur]) * abs(dAd_buf[prev2]))
        else
            ortho_loss = 0.0
        end

        # True residual
        threaded_spmv!(Ax, A', x, spmv_plan)
        physical_true_residual_evals += 1
        d = _fused_diagnostics!(r_true, Ax, b, seed, x, r,
            reduction_workspace, false)
        norm_r_true = d.norm_true
        norm_x = updated_norm_x
        error_val = d.norm_error / norm_seed
        energy = 0.5 * d.xAx - d.xb
        norm_r = d.norm_native
        relative_residual, relative_normwise_2 =
            compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)

        native_relative_residual = norm_r / norm_b
        native_candidate = native_relative_residual <= convergence_tolerance
        new_native_candidate = native_candidate && !certification_pending
        certification_iter = certification_pending || native_candidate
        if certification_iter
            certification_true_residual_evals += 1
            certification_check_count += 1
            _grow_diagnostic_vector!(certification_check_iters, certification_check_count, maxiters)
            _grow_diagnostic_vector!(true_residual_used_for_decision_iters, certification_check_count, maxiters)
            certification_check_iters[certification_check_count] = i
            true_residual_used_for_decision_iters[certification_check_count] = i
            if native_candidate && native_candidate_iter === nothing
                native_candidate_iter = i
                native_candidate_true_relres = relative_residual
                native_candidate_normwise = relative_normwise_2
            end
        end

        cum_base += Float64(base_fma)
        certification_iter && (cum_base += Float64(base_fma))
        cum_vector += Float64(FPCG1_VECTOR_FMA_PER_ITER) * n
        certification_iter &&
            (cum_vector += Float64(TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA) * n)
        correction_base === nothing || (cum_vector += Float64(n))
        _observe_native_candidate!(policy, native_candidate, i, cum_base + cum_prec + cum_vector)
        final_relative_residual = relative_residual

        data[row, FPCG_COL_KRYLOV_BLOCK] = Float64(coefficient_streams.formal.block_index)
        data[row, FPCG_COL_ITER]                = Float64(i)
        data[row, FPCG_COL_RELATIVE_RESIDUAL]   = relative_residual
        data[row, FPCG_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
        data[row, FPCG_COL_ERROR]               = error_val
        data[row, FPCG_COL_CUM_BASE_FMA]        = cum_base
        data[row, FPCG_COL_CUM_PREC_FMA]        = cum_prec
        data[row, FPCG_COL_CUM_VECTOR_FMA]      = cum_vector
        data[row, FPCG_COL_ALPHA]               = α
        data[row, FPCG_COL_NORM_R]              = norm_r
        data[row, FPCG_COL_NORM_R_TRUE]         = norm_r_true
        data[row, FPCG_COL_ENERGY]              = energy
        data[row, FPCG_COL_ORTHO_LOSS]          = ortho_loss
        data[row, FPCG_COL_ORTHO_COEFF_1]       = β1
        data[row, FPCG_COL_ORTHO_COEFF_2]       = β2_diag
        data[row, FPCG_COL_CORRECTION_NORM]     = correction_norm
        data[row, FPCG_COL_PREC_FMA_ITER]       = Float64(prec_fma_iter)
        data[row, FPCG_COL_DIAG_ABS_FLAG]       = diag_abs_flag

        if KILL_FLAG[]
            status = KILLED
            break
        end

        certified_this_iter = false
        if certification_iter
            certified = relative_residual <= convergence_tolerance &&
                        relative_normwise_2 <= convergence_tolerance
            if solver_mode == :bounded_continuation
                cap = policy.continuation_work_limit
                within_budget = (work_limit <= 0 || cum_base + cum_prec + cum_vector <= work_limit) &&
                    (cap === nothing || cum_base + cum_prec + cum_vector <= cap)
                certified &= within_budget
            end
            certified_this_iter = certified
            correction_restart_this_iter =
                solver_mode == :certified &&
                !certified &&
                new_native_candidate &&
                !correction_restart_consumed &&
                relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)
            correction_restart_action = correction_restart_this_iter ? "correction_restart" : "continue_recurrence"
            if correction_restart_this_iter
                r .= r_true
                prec_fma_iter = preconditioner.LinearOperator(w, r)
                cum_prec += Float64(prec_fma_iter)
                for j in eachindex(d_buf)
                    d_buf[j] .= 0.0
                    q_buf[j] .= 0.0
                end
                dAd_buf .= 0.0
                cur = 1
                d_buf[cur] .= w
                threaded_spmv!(q_buf[cur], A', d_buf[cur], spmv_plan)
                dAd_restart = threaded_dot(d_buf[cur], q_buf[cur], reduction_workspace)
                dAd_buf[cur] = dAd_restart
                dr = threaded_dot(d_buf[cur], r, reduction_workspace)
                cum_base += Float64(base_fma)
                cum_vector += Float64(FPCG1_VECTOR_FMA_RESIDUAL_REPLACEMENT) * n
                data[row, FPCG_COL_CUM_BASE_FMA] = cum_base
                data[row, FPCG_COL_CUM_PREC_FMA] = cum_prec
                data[row, FPCG_COL_CUM_VECTOR_FMA] = cum_vector
                if work_limit > 0 && (cum_base + cum_prec + cum_vector) > work_limit
                    correction_restart_action = "correction_restart_work_limit"
                    status = WORK_LIMIT
                elseif dAd_restart <= 0.0 || !isfinite(dAd_restart)
                    correction_restart_action = "correction_restart_breakdown"
                    status = BREAKDOWN
                else
                    restart_index = correction_restarts + 1
                    closed_tridiag_path, closed_recurrence_path =
                        _close_flexible_coefficients!(coefficient_streams; next_iteration=i+1)
                    coefficient_c1 = 0.0
                    coefficient_c2 = 0.0
                    _record_correction_restart!(
                        correction_restart_records,
                        raw_data_out,
                        restart_index,
                        i,
                        native_relative_residual,
                        relative_residual,
                        relative_normwise_2,
                        norm_r,
                        norm_r_true,
                        x;
                        closed_tridiag_path = closed_tridiag_path,
                        next_tridiag_path = _tridiag_block_path(raw_data_out, coefficient_streams.formal.block_index),
                    )
                    last(correction_restart_records)["closed_recurrence_path"] = closed_recurrence_path
                    last(correction_restart_records)["next_recurrence_path"] = _diagnostic_manifest(
                        _coefficient_path(raw_data_out, "_recurrence", coefficient_streams.actual.block_index))
                    α = dr / dAd_restart
                    local_iter = 0
                    correction_restarts += 1
                    correction_base = copy(x)
                    update_x = zeros(n)
                    correction_restart_consumed = true
                end
            end
            if new_native_candidate || certified
                event = new_native_candidate ? "native_candidate" : "certification_audit"
                action = certified ? "accept_convergence" : correction_restart_action
                _record_certification_check!(
                    certification_records,
                    i,
                    event,
                    native_relative_residual,
                    relative_residual,
                    relative_normwise_2,
                    norm_r,
                    norm_r_true;
                    certified = certified,
                    action = action,
                )
            end
            if certified
                certification_pending = false
            else
                new_native_candidate && (false_native_crossings += 1)
                certification_pending = true
            end
        end

        if status == BREAKDOWN || status == WORK_LIMIT
            break
        end

        stop_status = _policy_stop_status(policy, cum_base + cum_prec + cum_vector, certified_this_iter)
        if stop_status !== nothing
            status = stop_status
            break
        end
        # α describes the next direction, not the update just certified.
        if !isfinite(α)
            status = NAN_DETECTED
            break
        end
    end
    catch e
        e isa InterruptException || rethrow(e)
        status = KILLED
    end

    wall_time = time() - t0
    _experimental_final_iterate === nothing || copyto!(_experimental_final_iterate, x)
    _report_solver_progress!(progress, data, nrows, (iteration=FPCG_COL_ITER, native=FPCG_COL_NORM_R, true_rr=FPCG_COL_RELATIVE_RESIDUAL, normwise=FPCG_COL_RELATIVE_NORMWISE_2, block=FPCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; final=true, status=string(status), policy=policy)
    total_fma = cum_base + cum_prec + cum_vector
    resize!(certification_check_iters, certification_check_count)
    resize!(true_residual_used_for_decision_iters, certification_check_count)

    # Finalize raw blocks, then scan them without concatenating the history.
    output_operation = begin_reported_phase(progress.callback, "result_write")
    try
        used = _finish_diagnostic_data!(data, nrows;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)

        # Write column metadata
        final_tridiag_path, final_recurrence_path = _close_flexible_coefficients!(coefficient_streams; next_iteration=iter+1,
            callback=output_operation === nothing ? progress.callback : output_operation.scope)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_tridiag_path)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_recurrence_path; key="next_recurrence_path")

        columns_path = replace(raw_data_out, r"\.[^/]*$" => "_columns.json")
        open(columns_path, "w") do f
            JSON3.write(f, FPCG_COLUMN_NAMES)
        end

        # Extract convergence thresholds post-hoc
        _write_convergence(
            convergence_data_out * "_relative_residual.csv",
            used,
            FPCG_COL_RELATIVE_RESIDUAL,
        )
        _write_convergence(
            convergence_data_out * "_relative_normwise_2.csv",
            used,
            FPCG_COL_RELATIVE_NORMWISE_2,
        )
        _write_certification_convergence(
            convergence_data_out * "_certification.csv",
            used,
            FPCG_COL_NORM_R,
            norm_b,
            certification_records,
            certification_check_iters,
            solver_policy = :explicit,
            n = n,
            base_fma = base_fma,
            correction_restart_normwise_tolerance = correction_restart_normwise_tolerance,
            solver_mode = solver_mode,
            continuation_work_factor = continuation_work_factor,
            work_limit = work_limit,
        )

        if write_solution !== nothing && status == CONVERGED
            _write_solution_vec(write_solution, x)
        end
    catch e
        finish_reported_phase(output_operation; error=e)
        rethrow()
    end
    finish_reported_phase(output_operation)

    return SolverResult(
        status,
        iter,
        wall_time,
        total_fma,
        final_relative_residual,
        residual_replacements,
        residual_replacement_records,
        correction_restarts,
        correction_restart_records,
        physical_true_residual_evals,
        physical_true_residual_evals - certification_true_residual_evals,
        certification_true_residual_evals,
        certification_check_iters,
        true_residual_used_for_decision_iters,
        false_native_crossings,
        native_candidate_iter,
        native_candidate_true_relres,
        native_candidate_normwise,
        certification_records,
        _solver_policy_metadata(policy; status=status, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance),
    )
end

"""
    fpcg2(A, b, seed, spectral_norm, preconditioner,
          maxiters, raw_data_out, convergence_data_out;
          work_limit=-1)

Solves `Ax = b` using Notay's Flexible Conjugate Gradient, FCG(2).

Orthogonalizes each new search direction against up to 2 previous directions
(A-conjugation), with truncation pattern `m_i = max(1, i mod 3)`: every 3rd
iteration uses only 1 direction. Accepts a `Preconditioner` whose
apply returns the per-application FMA cost.

Failed native certifications can trigger one global correction solve when
`relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)`.

# References
- Notay, Flexible Conjugate Gradients, SIAM J. Sci. Comput. 22(4), 2000.
"""
function _fpcg2_with_progress(
    A::SparseMatrixCSC,
    b::Vector{Float64},
    seed::Vector{Float64},
    spectral_norm::Float64,
    preconditioner::Preconditioner,
    maxiters::Int64,
    raw_data_out::String,
    convergence_data_out::String;
    work_limit::Int64 = -1,
    convergence_tolerance::Float64 = minimum_tolerance,
    correction_restart_normwise_tolerance::Float64 = correction_restart_normwise_tolerance_default,
    write_solution::Union{Nothing,String} = nothing,
    _experimental_final_iterate::Union{Nothing,Vector{Float64}} = nothing,
    _diagnostic_block_rows::Int = DIAGNOSTIC_BLOCK_ROWS,
    progress = NumericalProgress(NOOP_PHASE_CALLBACK),
    solver_mode::Symbol = :certified,
    continuation_work_factor::Float64 = 2.0,
)
    policy = SolverPolicyState(solver_mode, continuation_work_factor, work_limit)
    n = size(A, 1)
    if _experimental_final_iterate !== nothing
        length(_experimental_final_iterate) == n || throw(DimensionMismatch("final iterate buffer"))
        fill!(_experimental_final_iterate, 0.0)
    end
    spmv_plan = _spmv_plan(A)
    reduction_workspace = FusedReductionWorkspace()
    base_fma = UInt64(nnz(A))
    norm_b = threaded_norm(b)
    norm_seed = threaded_norm(seed)

    # Preallocate iteration data
    data = DiagnosticBlockWriter(raw_data_out, maxiters + 1, FPCG_COLUMN_NAMES;
        block_rows=_diagnostic_block_rows, callback=progress.callback)

    # Working vectors
    x = zeros(n)
    # Before recovery, updates target x directly. Allocate correction storage lazily.
    correction_base = nothing
    update_x = x
    correction_assembly_steps = 0
    r = copy(b)
    w = zeros(n)        # preconditioned residual
    Ax = zeros(n)
    r_true = similar(b)

    # Circular buffer for directions and their A-images (size 4 for loss tracking)
    # cur: d_{i}, prev1: d_{i-1}, prev2: d_{i-2}, prev3: d_{i-3}
    d_buf = [zeros(n) for _ in 1:4]
    q_buf = [zeros(n) for _ in 1:4]   # q = A*d for each buffered direction
    dAd_buf = zeros(4)                  # (d_k, A d_k) cached per slot

    # Initial preconditioner application
    prec_fma_iter = preconditioner.LinearOperator(w, r)
    cum_prec = Float64(prec_fma_iter)

    # d_0 = w_0 (no orthogonalization)
    cur = 1
    d_buf[cur] .= w
    threaded_spmv!(q_buf[cur], A', d_buf[cur], spmv_plan)
    dAd_buf[cur] = threaded_dot(d_buf[cur], q_buf[cur], reduction_workspace)

    dr = threaded_dot(d_buf[cur], r, reduction_workspace)
    α = dr / dAd_buf[cur]

    # Record iteration 0
    norm_r = threaded_norm(r)
    cum_base = Float64(base_fma)
    cum_vector = Float64(FPCG2_VECTOR_FMA_ITER0) * n

    data[1, FPCG_COL_ITER]                = 0.0
    data[1, FPCG_COL_RELATIVE_RESIDUAL]   = norm_r / norm_b
    data[1, FPCG_COL_RELATIVE_NORMWISE_2] = norm_r / (norm_b + spectral_norm * threaded_norm(x))
    data[1, FPCG_COL_ERROR]               = 1.0
    data[1, FPCG_COL_CUM_BASE_FMA]        = cum_base
    data[1, FPCG_COL_CUM_PREC_FMA]        = cum_prec
    data[1, FPCG_COL_CUM_VECTOR_FMA]      = cum_vector
    data[1, FPCG_COL_ALPHA]               = α
    data[1, FPCG_COL_NORM_R]              = norm_r
    data[1, FPCG_COL_NORM_R_TRUE]         = norm_r  # r = b at x=0
    data[1, FPCG_COL_ENERGY]              = 0.0
    data[1, FPCG_COL_ORTHO_LOSS]          = 0.0
    data[1, FPCG_COL_ORTHO_COEFF_1]       = 0.0
    data[1, FPCG_COL_ORTHO_COEFF_2]       = 0.0
    data[1, FPCG_COL_CORRECTION_NORM]     = 0.0
    data[1, FPCG_COL_PREC_FMA_ITER]       = Float64(prec_fma_iter)
    data[1, FPCG_COL_DIAG_ABS_FLAG]       = 0.0

    data[1, FPCG_COL_KRYLOV_BLOCK] = 0.0

    iter = 0
    nrows = 1
    status = MAX_ITERS
    final_relative_residual = norm_r / norm_b
    coefficient_streams = FlexibleCoefficientStreams(raw_data_out, maxiters, _diagnostic_block_rows;
        callback=progress.callback, source_attempt=basename(data.attempt))
    coefficient_c1 = 0.0
    coefficient_c2 = 0.0
    local_iter = 0
    residual_replacements = 0
    residual_replacement_records = Dict{String,Any}[]
    correction_restarts = 0
    correction_restart_records = Dict{String,Any}[]
    physical_true_residual_evals = 0
    certification_true_residual_evals = 0
    certification_check_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    true_residual_used_for_decision_iters = Vector{Int}(undef, min(maxiters, DIAGNOSTIC_INITIAL_CAPACITY))
    certification_check_count = 0
    false_native_crossings = 0
    native_candidate_iter = nothing
    native_candidate_true_relres = nothing
    native_candidate_normwise = nothing
    certification_records = Dict{String,Any}[]
    certification_pending = false
    # One correction restart is allowed per solve under the current policy.
    correction_restart_consumed = false

    Threads.atomic_xchg!(KILL_FLAG, false)
    t0 = time()

    try
    for i in 1:maxiters
        _prepare_diagnostic_row!(data, i + 1, maxiters + 1)
        phase_reporting_enabled(progress.callback) &&
        _report_solver_progress!(progress, data, nrows, (iteration=FPCG_COL_ITER, native=FPCG_COL_NORM_R, true_rr=FPCG_COL_RELATIVE_RESIDUAL, normwise=FPCG_COL_RELATIVE_NORMWISE_2, block=FPCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; policy=policy)
        iter = i
        row = i + 1
        nrows = row
        local_iter += 1

        _record_flexible_coefficients!(coefficient_streams, i, α, coefficient_c1, coefficient_c2,
            threaded_dot(r, w, reduction_workspace), dr, dAd_buf[cur], threaded_norm(r), threaded_norm(w))

        # Solution and residual update using previous iteration's α and d
        updated_norm_x = _fused_update_norm!(update_x, α, d_buf[cur], reduction_workspace, r, q_buf[cur], correction_base, x)
        if correction_base !== nothing
            correction_assembly_steps += 1
        end

        # Variable preconditioner application
        prec_fma_iter = preconditioner.LinearOperator(w, r)
        cum_prec += Float64(prec_fma_iter)

        # Rotate buffer
        prev1 = cur                   # d_{i-1}
        prev2 = mod1(cur - 1, 4)      # d_{i-2}
        prev3 = mod1(cur - 2, 4)      # d_{i-3}
        cur = mod1(cur + 1, 4)

        # Truncation pattern: m_i = max(1, i mod 3)
        use_2dir = (local_iter >= 2) && (mod(local_iter, 3) != 0)

        # Orthogonalize w against d_{i-1} (always)
        wAd1 = threaded_dot(w, q_buf[prev1], reduction_workspace)
        β1 = wAd1 / dAd_buf[prev1]

        # Orthogonalize against d_{i-2} (when truncation allows and available)
        if use_2dir
            wAd2 = threaded_dot(w, q_buf[prev2], reduction_workspace)
            β2 = wAd2 / dAd_buf[prev2]
        else
            β2 = 0.0
            # Diagnostic: compute what β2 would have been
            if local_iter >= 2
                wAd2 = threaded_dot(w, q_buf[prev2], reduction_workspace)
                β2_diag = wAd2 / dAd_buf[prev2]
            else
                β2_diag = 0.0
            end
        end

        coefficient_c1 = -β1
        coefficient_c2 = use_2dir ? -β2 : 0.0

        # Form new search direction
        norm_w = threaded_norm(w)
        if use_2dir
            d_buf[cur] .= w .- β1 .* d_buf[prev1] .- β2 .* d_buf[prev2]
            β2_diag = β2
        else
            d_buf[cur] .= w .- β1 .* d_buf[prev1]
        end

        # Correction norm: ||w - d|| / ||w||
        # w - d = β1 * d_{prev1} + β2 * d_{prev2} (the part subtracted)
        # Reuse r_true as scratch (not yet used this iteration)
        r_true .= w .- d_buf[cur]
        correction_norm = threaded_norm(r_true) / norm_w

        # A * d_i
        threaded_spmv!(q_buf[cur], A', d_buf[cur], spmv_plan)
        dAd_buf[cur] = threaded_dot(d_buf[cur], q_buf[cur], reduction_workspace)

        # Step size
        dr = threaded_dot(d_buf[cur], r, reduction_workspace)
        α = dr / dAd_buf[cur]

        # Orthogonality loss: check against direction beyond ortho window
        # For 2-dir iterations: check against d_{i-3}; for 1-dir: check against d_{i-2}
        diag_abs_flag = 0.0
        if use_2dir && local_iter >= 3
            dAd_old = threaded_dot(d_buf[cur], q_buf[prev3], reduction_workspace)
            if dAd_buf[cur] < 0.0 || dAd_buf[prev3] < 0.0
                diag_abs_flag = 1.0
            end
            ortho_loss = abs(dAd_old) / sqrt(abs(dAd_buf[cur]) * abs(dAd_buf[prev3]))
        elseif !use_2dir && local_iter >= 2
            dAd_old = threaded_dot(d_buf[cur], q_buf[prev2], reduction_workspace)
            if dAd_buf[cur] < 0.0 || dAd_buf[prev2] < 0.0
                diag_abs_flag = 1.0
            end
            ortho_loss = abs(dAd_old) / sqrt(abs(dAd_buf[cur]) * abs(dAd_buf[prev2]))
        else
            ortho_loss = 0.0
        end

        # True residual
        threaded_spmv!(Ax, A', x, spmv_plan)
        physical_true_residual_evals += 1
        d = _fused_diagnostics!(r_true, Ax, b, seed, x, r,
            reduction_workspace, false)
        norm_r_true = d.norm_true
        norm_x = updated_norm_x
        error_val = d.norm_error / norm_seed
        energy = 0.5 * d.xAx - d.xb
        norm_r = d.norm_native
        relative_residual, relative_normwise_2 =
            compute_tolerances(norm_r_true, norm_b, norm_x, spectral_norm)

        native_relative_residual = norm_r / norm_b
        native_candidate = native_relative_residual <= convergence_tolerance
        new_native_candidate = native_candidate && !certification_pending
        certification_iter = certification_pending || native_candidate
        if certification_iter
            certification_true_residual_evals += 1
            certification_check_count += 1
            _grow_diagnostic_vector!(certification_check_iters, certification_check_count, maxiters)
            _grow_diagnostic_vector!(true_residual_used_for_decision_iters, certification_check_count, maxiters)
            certification_check_iters[certification_check_count] = i
            true_residual_used_for_decision_iters[certification_check_count] = i
            if native_candidate && native_candidate_iter === nothing
                native_candidate_iter = i
                native_candidate_true_relres = relative_residual
                native_candidate_normwise = relative_normwise_2
            end
        end

        cum_base += Float64(base_fma)
        certification_iter && (cum_base += Float64(base_fma))
        cum_vector += Float64(use_2dir ? FPCG2_VECTOR_FMA_2DIR : FPCG2_VECTOR_FMA_1DIR) * n
        certification_iter &&
            (cum_vector += Float64(TRUE_RESIDUAL_CERTIFICATION_VECTOR_FMA) * n)
        correction_base === nothing || (cum_vector += Float64(n))
        _observe_native_candidate!(policy, native_candidate, i, cum_base + cum_prec + cum_vector)
        final_relative_residual = relative_residual

        data[row, FPCG_COL_KRYLOV_BLOCK] = Float64(coefficient_streams.formal.block_index)
        data[row, FPCG_COL_ITER]                = Float64(i)
        data[row, FPCG_COL_RELATIVE_RESIDUAL]   = relative_residual
        data[row, FPCG_COL_RELATIVE_NORMWISE_2] = relative_normwise_2
        data[row, FPCG_COL_ERROR]               = error_val
        data[row, FPCG_COL_CUM_BASE_FMA]        = cum_base
        data[row, FPCG_COL_CUM_PREC_FMA]        = cum_prec
        data[row, FPCG_COL_CUM_VECTOR_FMA]      = cum_vector
        data[row, FPCG_COL_ALPHA]               = α
        data[row, FPCG_COL_NORM_R]              = norm_r
        data[row, FPCG_COL_NORM_R_TRUE]         = norm_r_true
        data[row, FPCG_COL_ENERGY]              = energy
        data[row, FPCG_COL_ORTHO_LOSS]          = ortho_loss
        data[row, FPCG_COL_ORTHO_COEFF_1]       = β1
        data[row, FPCG_COL_ORTHO_COEFF_2]       = β2_diag
        data[row, FPCG_COL_CORRECTION_NORM]     = correction_norm
        data[row, FPCG_COL_PREC_FMA_ITER]       = Float64(prec_fma_iter)
        data[row, FPCG_COL_DIAG_ABS_FLAG]       = diag_abs_flag

        if KILL_FLAG[]
            status = KILLED
            break
        end

        certified_this_iter = false
        if certification_iter
            certified = relative_residual <= convergence_tolerance &&
                        relative_normwise_2 <= convergence_tolerance
            if solver_mode == :bounded_continuation
                cap = policy.continuation_work_limit
                within_budget = (work_limit <= 0 || cum_base + cum_prec + cum_vector <= work_limit) &&
                    (cap === nothing || cum_base + cum_prec + cum_vector <= cap)
                certified &= within_budget
            end
            certified_this_iter = certified
            correction_restart_this_iter =
                solver_mode == :certified &&
                !certified &&
                new_native_candidate &&
                !correction_restart_consumed &&
                relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)
            correction_restart_action = correction_restart_this_iter ? "correction_restart" : "continue_recurrence"
            if correction_restart_this_iter
                r .= r_true
                prec_fma_iter = preconditioner.LinearOperator(w, r)
                cum_prec += Float64(prec_fma_iter)
                for j in eachindex(d_buf)
                    d_buf[j] .= 0.0
                    q_buf[j] .= 0.0
                end
                dAd_buf .= 0.0
                cur = 1
                d_buf[cur] .= w
                threaded_spmv!(q_buf[cur], A', d_buf[cur], spmv_plan)
                dAd_restart = threaded_dot(d_buf[cur], q_buf[cur], reduction_workspace)
                dAd_buf[cur] = dAd_restart
                dr = threaded_dot(d_buf[cur], r, reduction_workspace)
                cum_base += Float64(base_fma)
                cum_vector += Float64(FPCG2_VECTOR_FMA_RESIDUAL_REPLACEMENT) * n
                data[row, FPCG_COL_CUM_BASE_FMA] = cum_base
                data[row, FPCG_COL_CUM_PREC_FMA] = cum_prec
                data[row, FPCG_COL_CUM_VECTOR_FMA] = cum_vector
                if work_limit > 0 && (cum_base + cum_prec + cum_vector) > work_limit
                    correction_restart_action = "correction_restart_work_limit"
                    status = WORK_LIMIT
                elseif dAd_restart <= 0.0 || !isfinite(dAd_restart)
                    correction_restart_action = "correction_restart_breakdown"
                    status = BREAKDOWN
                else
                    restart_index = correction_restarts + 1
                    closed_tridiag_path, closed_recurrence_path =
                        _close_flexible_coefficients!(coefficient_streams; next_iteration=i+1)
                    coefficient_c1 = 0.0
                    coefficient_c2 = 0.0
                    _record_correction_restart!(
                        correction_restart_records,
                        raw_data_out,
                        restart_index,
                        i,
                        native_relative_residual,
                        relative_residual,
                        relative_normwise_2,
                        norm_r,
                        norm_r_true,
                        x;
                        closed_tridiag_path = closed_tridiag_path,
                        next_tridiag_path = _tridiag_block_path(raw_data_out, coefficient_streams.formal.block_index),
                    )
                    last(correction_restart_records)["closed_recurrence_path"] = closed_recurrence_path
                    last(correction_restart_records)["next_recurrence_path"] = _diagnostic_manifest(
                        _coefficient_path(raw_data_out, "_recurrence", coefficient_streams.actual.block_index))
                    α = dr / dAd_restart
                    local_iter = 0
                    correction_restarts += 1
                    correction_base = copy(x)
                    update_x = zeros(n)
                    correction_restart_consumed = true
                end
            end
            if new_native_candidate || certified
                event = new_native_candidate ? "native_candidate" : "certification_audit"
                action = certified ? "accept_convergence" : correction_restart_action
                _record_certification_check!(
                    certification_records,
                    i,
                    event,
                    native_relative_residual,
                    relative_residual,
                    relative_normwise_2,
                    norm_r,
                    norm_r_true;
                    certified = certified,
                    action = action,
                )
            end
            if certified
                certification_pending = false
            else
                new_native_candidate && (false_native_crossings += 1)
                certification_pending = true
            end
        end

        if status == BREAKDOWN || status == WORK_LIMIT
            break
        end

        stop_status = _policy_stop_status(policy, cum_base + cum_prec + cum_vector, certified_this_iter)
        if stop_status !== nothing
            status = stop_status
            break
        end
        # α describes the next direction, not the update just certified.
        if !isfinite(α)
            status = NAN_DETECTED
            break
        end
    end
    catch e
        e isa InterruptException || rethrow(e)
        status = KILLED
    end

    wall_time = time() - t0
    _experimental_final_iterate === nothing || copyto!(_experimental_final_iterate, x)
    _report_solver_progress!(progress, data, nrows, (iteration=FPCG_COL_ITER, native=FPCG_COL_NORM_R, true_rr=FPCG_COL_RELATIVE_RESIDUAL, normwise=FPCG_COL_RELATIVE_NORMWISE_2, block=FPCG_COL_KRYLOV_BLOCK), norm_b,
            maxiters, work_limit, cum_base + cum_prec + cum_vector,
            certification_check_iters, certification_check_count, correction_restarts; final=true, status=string(status), policy=policy)
    total_fma = cum_base + cum_prec + cum_vector
    resize!(certification_check_iters, certification_check_count)
    resize!(true_residual_used_for_decision_iters, certification_check_count)

    # Finalize raw blocks, then scan them without concatenating the history.
    output_operation = begin_reported_phase(progress.callback, "result_write")
    try
        used = _finish_diagnostic_data!(data, nrows;
            callback=output_operation === nothing ? progress.callback : output_operation.scope)

        # Write column metadata
        final_tridiag_path, final_recurrence_path = _close_flexible_coefficients!(coefficient_streams; next_iteration=iter+1,
            callback=output_operation === nothing ? progress.callback : output_operation.scope)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_tridiag_path)
        _clear_unwritten_next_tridiag!(correction_restart_records, final_recurrence_path; key="next_recurrence_path")

        columns_path = replace(raw_data_out, r"\.[^/]*$" => "_columns.json")
        open(columns_path, "w") do f
            JSON3.write(f, FPCG_COLUMN_NAMES)
        end

        # Extract convergence thresholds post-hoc
        _write_convergence(
            convergence_data_out * "_relative_residual.csv",
            used,
            FPCG_COL_RELATIVE_RESIDUAL,
        )
        _write_convergence(
            convergence_data_out * "_relative_normwise_2.csv",
            used,
            FPCG_COL_RELATIVE_NORMWISE_2,
        )
        _write_certification_convergence(
            convergence_data_out * "_certification.csv",
            used,
            FPCG_COL_NORM_R,
            norm_b,
            certification_records,
            certification_check_iters,
            solver_policy = :explicit,
            n = n,
            base_fma = base_fma,
            correction_restart_normwise_tolerance = correction_restart_normwise_tolerance,
            solver_mode = solver_mode,
            continuation_work_factor = continuation_work_factor,
            work_limit = work_limit,
        )

        if write_solution !== nothing && status == CONVERGED
            _write_solution_vec(write_solution, x)
        end
    catch e
        finish_reported_phase(output_operation; error=e)
        rethrow()
    end
    finish_reported_phase(output_operation)

    return SolverResult(
        status,
        iter,
        wall_time,
        total_fma,
        final_relative_residual,
        residual_replacements,
        residual_replacement_records,
        correction_restarts,
        correction_restart_records,
        physical_true_residual_evals,
        physical_true_residual_evals - certification_true_residual_evals,
        certification_true_residual_evals,
        certification_check_iters,
        true_residual_used_for_decision_iters,
        false_native_crossings,
        native_candidate_iter,
        native_candidate_true_relres,
        native_candidate_normwise,
        certification_records,
        _solver_policy_metadata(policy; status=status, convergence_tolerance=convergence_tolerance,
            correction_restart_normwise_tolerance=correction_restart_normwise_tolerance),
    )
end

include("zero_rhs.jl")
include("solver_progress.jl")

end
