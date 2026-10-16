module Utilities

using ..NativeLibraries

# Basic Linear Algebra
using LinearAlgebra
using SparseArrays

# Stable Random Number Generation
using StableRNGs
using Random    # shuffle
using Statistics # mean

# Orderings
import AMD
import SymRCM
import Metis

# File I/O
import CSV
import DataFrames
using CodecZstd
using TranscodingStreams
using MatrixMarket
import JSON3
import SHA

# Date/Time
using Dates

# Matrix Utilities
using Laplacians

using Libdl
using ..ThreadedUtilities: threaded_spmv!, threaded_norm, _spmv_plan,
    _sleep_julia_workers_for_blas!
using Polyester: @batch

const AMD_MAX_NNZ_DEFAULT = 1_000_000_000
const ORDERING_TIMEOUT_SECONDS_DEFAULT = 24 * 60 * 60
const EIGBOUNDS_LOWER_TOL_DEFAULT = 1e-2
const EIGBOUNDS_UPPER_TOL_DEFAULT = 1e-3
const EIGBOUNDS_MAX_RESTARTS_DEFAULT = 100
const QUADRATURE_UNSCALED_MAX_ITERS = 60
const QUADRATURE_SCALED_MAX_ITERS = 30
const PREPARE_CHECKPOINT_SCHEMA_VERSION = 1

_eigbounds_max_restarts(n::Integer) =
    max(1, min(EIGBOUNDS_MAX_RESTARTS_DEFAULT, ceil(Int, 37.5 * log(n))))

_noop_prepare_checkpoint_callback = _ -> nothing

# ─── Exports ────────────────────────────────────────────────────────

# Matrix analysis
export is_sdd, is_m_matrix, classify_component, reduce_to_laplacian
export diagonal_dominance_ratio
export estimate_eigrange, estimate_eigbounds, lanczos_quadrature, tridiag_eigvals!, tridiag_eigen!, tridiag_eigen_mrrr!
export ensure_dsterf_built, ensure_dstev_built, ensure_dstegr_built

# Matrix operations
export extract_negative, make_augmented, laplace_to_adj, adjacency_to_laplacian
export scale_symmetric!

# Component loading pipeline
export ComponentDescriptor, describe_components
export load_component_matrix, load_components, load_reordering
export generate_component_rhs, select_prolific_node
export decompose_and_cache, prepare_system
export AMD_MAX_NNZ_DEFAULT

# Tolerances
export compute_tolerances

# Resource monitoring
export ResourceSample, ResourceMonitor, start_monitor, stop_monitor!, write_resources

# Logging
export LogEntry, Logger, open_logger, log!, close_logger!
export NOOP_PHASE_CALLBACK, run_phase, run_scoped_phase, phase_callback_with_context

# Binary I/O
export read_npy_zstd_1d_f64, read_npy_zstd_1d_i64,
       write_npy_zstd_1d, write_npy_zstd_1d_f64, write_npy_zstd_1d_i64,
       read_matrixmarket_coord

include("phase_reporting.jl")
include("lanczos_workspace.jl")
include("triangle_counts.jl")

# 1-D NPY and zstd I/O
#
# These helpers produce / consume files compatible with `numpy.save` (NPY 1.0)
# wrapped in a zstd compressor. They are the canonical on-disk format for
# 1-D scalar arrays throughout the package: component CSC structures
# (colptr/rowval/nzval), reorderings, RHS vectors, reference solutions.

"""
    _parse_npy_1d_header(raw::Vector{UInt8}) -> NamedTuple

Parse the NPY 1.0 header on a 1-D array. Returns `(n, descr, data_start)`
where `n` is the array length, `descr` is the numpy dtype string
(e.g. `"<f8"`, `"<i8"`), and `data_start` is the 1-based byte offset of
the array data.
"""
function _parse_npy_1d_header(raw::AbstractVector{UInt8})
    @assert raw[1:6] == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] "Not a .npy file"
    raw[7] == 0x01 ||
        error("_parse_npy_1d_header: unsupported NPY major version $(raw[7]); expected 1")
    header_len = Int(reinterpret(UInt16, raw[9:10])[1])
    header_str = String(raw[11:10+header_len])

    shape_match = match(r"'shape':\s*\((\d+),?\s*\)", header_str)
    shape_match === nothing &&
        error("_parse_npy_1d_header: could not parse 1-D shape from header: $header_str")
    n = parse(Int, shape_match[1])

    descr_match = match(r"'descr':\s*'([^']+)'", header_str)
    descr_match === nothing &&
        error("_parse_npy_1d_header: could not parse descr from header: $header_str")
    descr = descr_match[1]

    return (n = n, descr = descr, data_start = 11 + header_len)
end

"""
    read_npy_zstd_1d_f64(path) -> Vector{Float64}

Read a 1-D Float64 array from a zstd-compressed `.npy` file. Verifies the
NPY header reports `<f8` (little-endian 8-byte float) and that the data
length matches the declared shape.
"""
read_npy_zstd_1d_f64(path::String) = _read_npy_zstd_vector(path, Float64, "<f8")

"""
    read_npy_zstd_1d_i64(path) -> Vector{Int64}

Read a 1-D Int64 array from a zstd-compressed `.npy` file. Verifies the
NPY header reports `<i8` (little-endian 8-byte signed int).
"""
read_npy_zstd_1d_i64(path::String) = _read_npy_zstd_vector(path, Int64, "<i8")

include("streaming_npy.jl")

"""
    write_npy_zstd_1d(path, v, descr::String)

Write a 1-D array as a zstd-compressed NPY 1.0 file. `descr` is the numpy
dtype string: `"<f8"` for Float64, `"<i8"` for Int64. Header is padded
to 64-byte alignment matching `numpy.save`'s convention.
"""
function write_npy_zstd_1d(path::String, v::AbstractVector, descr::String)
    n = length(v)
    header_dict = "{'descr': '$descr', 'fortran_order': False, 'shape': ($n,), }"
    prefix_len = 10                 # 6 magic + 1 major + 1 minor + 2 length
    pad_len = 64 - mod(prefix_len + length(header_dict) + 1, 64)
    if pad_len == 64; pad_len = 0; end
    padded_header = header_dict * " "^pad_len * "\n"
    header_len = length(padded_header)

    T = descr == "<f8" ? Float64 : descr == "<i8" ? Int64 :
        throw(ArgumentError("unsupported NPY dtype: $descr"))
    open(path, "w") do f
        zstd = ZstdCompressorStream(f, level = 1)
        try
        write(zstd, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])  # magic
        write(zstd, UInt8(1))                                     # major
        write(zstd, UInt8(0))                                     # minor
        write(zstd, UInt16(header_len))                           # header length
        write(zstd, padded_header)
        _write_npy_payload(zstd, v, T)
        finally
            close(zstd)
        end
    end
    return path
end

write_npy_zstd_1d_f64(path::String, v::AbstractVector{Float64}) =
    write_npy_zstd_1d(path, v, "<f8")

write_npy_zstd_1d_i64(path::String, v::AbstractVector{<:Integer}) =
    write_npy_zstd_1d(path, v, "<i8")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom MatrixMarket reader (preserves explicit zeros)
# ═══════════════════════════════════════════════════════════════════════════════
#
# `MatrixMarket.mmread` silently drops entries with value 0.0 on load, which
# corrupts series-style supersetted patterns where some edges are encoded as
# explicit zeros. This reader handles the subset of MM coordinate files that
# decompose_and_cache sees and preserves all stored entries bit-for-bit.

"""
    read_matrixmarket_coord(io::IO) -> SparseMatrixCSC{Float64,Int64}

Parse a MatrixMarket coordinate-format matrix from `io`, preserving every
stored entry (including explicit zeros) bit-for-bit. Supports `real`,
`integer` (and the `int` alias), `pattern`, and `binary` fields. All values are promoted to
`Float64`; `pattern` entries are assigned value `1.0`, and `binary` entries
must be implicit or explicit 0/1 values. Supports `general` and `symmetric`
symmetries; for `symmetric`, entries are expanded to full storage so both
`A[i,j]` and `A[j,i]` are present.

Limitations (intentional, matching what the package consumes):
- Only `coordinate` format (not `array`).
- Only `real`, `integer`/`int`, `pattern`, and `binary` fields (not `complex`).
- Only `general` and `symmetric` symmetries (not `skew-symmetric`, `hermitian`).

Throws `ErrorException` on any unsupported variant.
"""
function _matrixmarket_binary_value(token::AbstractString, line::AbstractString, entry_idx::Int)::Float64
    value = lowercase(String(token))
    value in ("0", "false") && return 0.0
    value in ("1", "true") && return 1.0
    error("read_matrixmarket_coord: bad binary value on entry line $entry_idx: $line")
end

function _matrixmarket_entry_value(
    field::String,
    ep::Vector{SubString{String}},
    line::AbstractString,
    entry_idx::Int,
)::Float64
    if field == "pattern"
        length(ep) == 2 ||
            error("read_matrixmarket_coord: pattern entry line $entry_idx must contain exactly 2 indices: $line")
        return 1.0
    elseif field == "real"
        length(ep) == 3 ||
            error("read_matrixmarket_coord: real entry line $entry_idx must contain exactly 2 indices and 1 value: $line")
        return parse(Float64, ep[3])
    elseif field == "integer"
        length(ep) == 3 ||
            error("read_matrixmarket_coord: integer entry line $entry_idx must contain exactly 2 indices and 1 value: $line")
        return Float64(parse(Int64, ep[3]))
    elseif field == "binary"
        if length(ep) == 2
            return 1.0
        elseif length(ep) == 3
            return _matrixmarket_binary_value(ep[3], line, entry_idx)
        end
        error("read_matrixmarket_coord: binary entry line $entry_idx must contain 2 indices, optionally followed by 0/1: $line")
    end
    error("read_matrixmarket_coord: unsupported field: $field")
end

function read_matrixmarket_coord(io::IO)::SparseMatrixCSC{Float64,Int64}
    header = strip(readline(io))
    startswith(header, "%%MatrixMarket") ||
        error("read_matrixmarket_coord: missing '%%MatrixMarket' banner; got: $header")
    parts = split(header)
    length(parts) >= 5 ||
        error("read_matrixmarket_coord: malformed banner: $header")
    object = lowercase(String(parts[2]))
    format = lowercase(String(parts[3]))
    field = lowercase(String(parts[4]))
    field == "int" && (field = "integer")
    symmetry = lowercase(String(parts[5]))

    object == "matrix" ||
        error("read_matrixmarket_coord: only 'matrix' object supported; got: $(parts[2])")
    format == "coordinate" ||
        error("read_matrixmarket_coord: only 'coordinate' format supported; got: $(parts[3])")
    field in ("real", "integer", "pattern", "binary") ||
        error("read_matrixmarket_coord: only 'real', 'integer'/'int', 'pattern', or 'binary' fields supported; got: $(parts[4])")
    symmetry in ("general", "symmetric") ||
        error("read_matrixmarket_coord: only 'general' or 'symmetric' symmetry supported; got: $symmetry")

    # Skip comment lines and reach the size line
    line = readline(io)
    while !eof(io) && startswith(strip(line), "%")
        line = readline(io)
    end
    size_parts = split(strip(line))
    length(size_parts) == 3 ||
        error("read_matrixmarket_coord: malformed size line: $line")
    m_rows = parse(Int, size_parts[1])
    n_cols = parse(Int, size_parts[2])
    nnz_decl = parse(Int, size_parts[3])

    estimated = symmetry == "symmetric" ? 2 * nnz_decl : nnz_decl
    I_vec = Vector{Int64}(); sizehint!(I_vec, estimated)
    J_vec = Vector{Int64}(); sizehint!(J_vec, estimated)
    V_vec = Vector{Float64}(); sizehint!(V_vec, estimated)

    for entry_idx in 1:nnz_decl
        line = readline(io)
        ep = split(strip(line))
        length(ep) >= 2 ||
            error("read_matrixmarket_coord: bad entry line $entry_idx: $line")
        i = parse(Int64, ep[1])
        j = parse(Int64, ep[2])
        v = _matrixmarket_entry_value(field, ep, line, entry_idx)
        push!(I_vec, i); push!(J_vec, j); push!(V_vec, v)
        if symmetry == "symmetric" && i != j
            push!(I_vec, j); push!(J_vec, i); push!(V_vec, v)
        end
    end

    # `sparse(I, J, V, m, n)` preserves explicit-zero entries (verified) and
    # returns a SparseMatrixCSC; suitable for our format-preserving load.
    return sparse(I_vec, J_vec, V_vec, m_rows, n_cols)
end

"""
    read_matrixmarket_coord(path::String) -> SparseMatrixCSC{Float64,Int64}

Read a `.mtx` file. Convenience wrapper that opens the file and dispatches
to the IO-based reader.
"""
function read_matrixmarket_coord(path::String)
    endswith(path, ".mtx") ||
        error("read_matrixmarket_coord: expected a `.mtx` file, got: $path")
    return open(path, "r") do f
        return read_matrixmarket_coord(f)
    end
end


"""
    compute_tolerances(residual_norm, norm_b, norm_x, spectral_norm::Float64)

Computes relative residual and normwise backward error using the spectral norm.

# Returns
- `(relative_residual, relative_normwise_2)`:
    - `relative_residual`: `‖r‖ / ‖b‖`
    - `relative_normwise_2`: `‖r‖ / (‖b‖ + ‖A‖₂ ‖x‖)`
"""
function compute_tolerances(
    residual_norm::AbstractFloat,
    norm_b::AbstractFloat,
    norm_x::AbstractFloat,
    spectral_norm::Float64,
)
    relative_residual = residual_norm / norm_b
    relative_normwise_2 = residual_norm / (norm_b + spectral_norm * norm_x)
    return (relative_residual, relative_normwise_2)
end


"""
    extract_negative(A::SparseMatrixCSC{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

Extracts the negative elements from the given sparse matrix `A`.

# Arguments
- `A::SparseMatrixCSC{Tv, Ti}`: The input sparse matrix.

# Returns
- `SparseMatrixCSC{Tv, Ti}`: A sparse matrix containing only the negative elements of `A`, with the same dimensions as `A`.

# Notes
- This function identifies and isolates negative values while preserving their original positions in the matrix.
- The resulting matrix has zeros in place of all non-negative elements.
"""
function extract_negative(A::SparseMatrixCSC{Tv,Ti})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}
    m, n = size(A)
    rows, cols, vals = findnz(A)

    neg_indices = findall(x -> x < 0, vals)
    neg_vals = vals[neg_indices]
    neg_rows = rows[neg_indices]
    neg_cols = cols[neg_indices]

    return sparse(neg_rows, neg_cols, neg_vals, m, n)
end


"""
    make_augmented(A, b) -> (augmented_system, augmented_b)

Constructs an augmented (double cover) system from a symmetric diagonally dominant (SDD)
matrix `A`, producing a graph-Laplacian of size `2m × 2m`.

Uses Kahan compensated summation for the diagonal excess computation to avoid
floating-point issues at the SDD boundary.

# Arguments
- `A::SparseMatrixCSC`: The input SDD matrix.
- `b::Vector`: The right-hand side vector.

# Returns
- `SparseMatrixCSC{Float64,Int64}`: The augmented graph-Laplacian.
- `Vector{Float64}`: The augmented RHS `[b; -b]`.

# Notes
- Decomposes `A` into positive off-diag (`Ap`), negative off-diag (`An`),
  sum of absolute off-diagonals (`D1`), and diagonal excess (`D2 = diag(A) - D1`).
- `D1` and `D2` are computed per-row using `kahan_abssum_offdiag` for accuracy.
- The augmented matrix is:
    ```[ D1 + D2/2 + An    -(D2/2) - Ap ]
    [ -(D2/2) - Ap    D1 + D2/2 + An ]```
- Errors if any row of `A` is not diagonally dominant.
"""
function make_augmented(
    A::SparseMatrixCSC{Tv,Ti},
    b::Vector{Tv},
)::Tuple{SparseMatrixCSC{Float64,Int64},Vector{Float64}} where {Tv, Ti}

    A = convert(SparseMatrixCSC{Float64,Int64}, A)
    m, _ = size(A)

    # Decompose A into: diagonal, negative off-diag (An), positive off-diag (Ap)
    An = extract_negative(A)
    An = convert(SparseMatrixCSC{Float64,Int64}, An)
    diagonal = spdiagm(diag(A))
    Ap = A - An - diagonal

    # Compute D1 and D2 per row using Kahan compensated summation.
    # D1[i] = sum of absolute off-diagonals in row i = sum(Ap[i,:]) - sum(An[i,:])
    # D2[i] = diagonal excess = A[i,i] - D1[i] = A[i,i] - sum(|off-diag|)
    # D2 >= 0 for SDD matrices.
    d1 = Vector{Float64}(undef, m)
    d2 = Vector{Float64}(undef, m)
    kbuf = Vector{Float64}(undef, maximum(A.colptr[i+1] - A.colptr[i] for i in 1:m))
    for i in 1:m
        offdiag_abs = kahan_abssum_offdiag(A, i, kbuf)
        d1[i] = offdiag_abs
        d2[i] = A[i, i] - offdiag_abs
        if d2[i] < 0
            error("make_augmented: matrix is not SDD (row $i deficit = $(d2[i]))")
        end
    end

    D1 = spdiagm(d1)
    D2 = spdiagm(d2)

    # Augmented system:
    # [ D1 + D2/2 + An    -(D2/2) - Ap ]
    # [ -(D2/2) - Ap    D1 + D2/2 + An ]
    block_diag = D1 + D2 ./ 2 + An
    block_off = -(D2 ./ 2) - Ap
    augmented_system = [block_diag block_off; block_off block_diag]

    augmented_b = [Float64.(b); -Float64.(b)]

    return augmented_system, augmented_b
end

"""
    laplace_to_adj(A::SparseMatrixCSC{Tv, Ti}) -> SparseMatrixCSC{Tv, Ti}

Converts a Laplacian matrix `A` to its corresponding adjacency matrix.

# Arguments
- `A::SparseMatrixCSC{Tv, Ti}`: The Laplacian matrix to be converted.

# Returns
- `SparseMatrixCSC{Tv, Ti}`: The adjacency matrix corresponding to `A`.

# Notes
- The adjacency matrix is computed as `B = D - A`, where `D` is the diagonal degree matrix of `A`.
- This conversion assumes `A` is a valid Laplacian, meaning its diagonal entries represent node degrees, and off-diagonal entries represent negative edge weights.
- The resulting adjacency matrix is symmetric and retains the sparsity pattern of `A`.
"""
function laplace_to_adj(A::SparseMatrixCSC{Tv,Ti})::SparseMatrixCSC{Tv,Ti} where {Tv,Ti}

    m, n = size(A)

    diagonal = diag(A)

    B = spdiagm(diagonal) - A

    return B

end

"""
    adjacency_to_laplacian(W::SparseMatrixCSC) -> SparseMatrixCSC

Converts an adjacency (weight) matrix `W` to its graph Laplacian `L = D - W`,
where `D[i] = sum(|W[i,j]|)` for `j ≠ i` (sum of absolute off-diagonal entries).
This ensures the diagonal is non-negative even for graphs with negative edge
weights. Self-loops (diagonal entries of `W`) are ignored in the degree
computation and removed in the output. Degree sums use Kahan compensated
summation for numerical accuracy.
"""
function adjacency_to_laplacian(W::SparseMatrixCSC)
    n = size(W, 1)
    d = Vector{Float64}(undef, n)
    kbuf = Vector{Float64}(undef, maximum(W.colptr[i+1] - W.colptr[i] for i in 1:n))
    for i in 1:n
        d[i] = kahan_abssum_offdiag(W, i, kbuf)
    end
    return spdiagm(d) - W + spdiagm(diag(W))
end


"""
    is_sdd(B::SparseMatrixCSC) -> Bool

Checks whether the given sparse matrix `B` is symmetric and diagonally dominant (SDD).

# Arguments
- `B::SparseMatrixCSC`: The sparse matrix to check.

# Returns
- `Bool`: `true` if `B` is symmetric and diagonally dominant, otherwise `false`.

# Notes
- A matrix `B` is considered **symmetric diagonally dominant (SDD)** if:
  - It is symmetric, meaning `B == B'`.
  - For every row `i`, the diagonal entry satisfies `2 * B[i, i] ≥ sum(abs, B[:, i])`.
- The function iterates over all columns to verify diagonal dominance.
- If any row violates the SDD condition, the function returns `false` immediately.
"""
function is_sdd(B::SparseMatrixCSC; rtol::Float64=1e-6)::Bool
    n = size(B, 2)
    kbuf = Vector{Float64}(undef, n > 0 ? maximum(B.colptr[i+1] - B.colptr[i] for i in 1:n) : 0)
    for i in axes(B, 2)
        offdiag_sum = kahan_abssum_offdiag(B, i, kbuf)
        if B[i, i] < offdiag_sum * (1 - rtol)
            return false
        end
    end

    return true
end

"""
    kahan_abssum_offdiag(A, col, [buf]) -> Float64

Compute the sum of absolute values of off-diagonal entries in column `col`
using Kahan compensated summation with values sorted by ascending magnitude
for best floating-point accuracy.

The 3-argument form reuses `buf` as workspace (resized as needed) to avoid
per-call allocation. Callers in loops should preallocate `buf` once.
"""
function kahan_abssum_offdiag(A::SparseMatrixCSC, col::Integer, buf::Vector{Float64})::Float64
    nnz_col = A.colptr[col+1] - A.colptr[col]
    resize!(buf, nnz_col)
    k = 0
    for j in A.colptr[col]:(A.colptr[col+1]-1)
        if A.rowval[j] != col
            k += 1
            buf[k] = abs(A.nzval[j])
        end
    end
    v = view(buf, 1:k)
    sort!(v)  # ascending magnitude — sum small values first
    return kahan_sum(v)
end

function kahan_abssum_offdiag(A::SparseMatrixCSC, col::Integer)::Float64
    buf = Vector{Float64}(undef, A.colptr[col+1] - A.colptr[col])
    return kahan_abssum_offdiag(A, col, buf)
end

"""
    kahan_sum(vals) -> Float64

Kahan compensated summation. Assumes `vals` is pre-sorted by ascending
magnitude for best accuracy.
"""
function kahan_sum(vals::AbstractVector{Float64})::Float64
    s = 0.0
    c = 0.0  # compensation
    for v in vals
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    end
    return s
end

"""
    kahan_rowsum(A, col, [buf]) -> Float64

Compute the full row/column sum of a symmetric matrix column using Kahan
compensated summation with values sorted by ascending absolute magnitude.
Used for checking if row sums are zero (Laplacian test).

The 3-argument form reuses `buf` as workspace to avoid per-call allocation.
"""
function kahan_rowsum(A::SparseMatrixCSC, col::Integer, buf::Vector{Float64})::Float64
    nnz_col = A.colptr[col+1] - A.colptr[col]
    resize!(buf, nnz_col)
    for j in A.colptr[col]:(A.colptr[col+1]-1)
        buf[j - A.colptr[col] + 1] = A.nzval[j]
    end
    v = view(buf, 1:nnz_col)
    sort!(v, by=abs)  # ascending absolute magnitude
    return kahan_sum(v)
end

function kahan_rowsum(A::SparseMatrixCSC, col::Integer)::Float64
    buf = Vector{Float64}(undef, A.colptr[col+1] - A.colptr[col])
    return kahan_rowsum(A, col, buf)
end

"""
    diagonal_dominance_ratio(A) -> Float64

Returns the minimum ratio of diagonal to off-diagonal absolute sum across
all rows. Values ≥ 1.0 indicate diagonal dominance.
"""
function diagonal_dominance_ratio(A::SparseMatrixCSC)::Float64
    n = size(A, 2)
    min_ratio = Inf
    kbuf = Vector{Float64}(undef, n > 0 ? maximum(A.colptr[i+1] - A.colptr[i] for i in 1:n) : 0)
    for i in axes(A, 2)
        offdiag = kahan_abssum_offdiag(A, i, kbuf)
        if offdiag > 0
            min_ratio = min(min_ratio, A[i, i] / offdiag)
        end
    end
    return min_ratio
end

"""
    is_m_matrix(A) -> Bool

Returns true if all off-diagonal entries of A are non-positive.
"""
function is_m_matrix(A::SparseMatrixCSC)::Bool
    for i in axes(A, 2)
        for j in A.colptr[i]:(A.colptr[i+1]-1)
            if A.rowval[j] != i && A.nzval[j] > 0
                return false
            end
        end
    end
    return true
end

"""
    classify_component(A) -> Symbol

Classify a symmetric component matrix as one of:
  :adjacency             — zero diagonal (adjacency / weight matrix, may have negative weights)
  :graph_laplacian       — M-matrix with zero row sums (L = D - W)
  :sddm                  — M-matrix, diagonally dominant, positive diagonal excess
  :general_sdd           — diagonally dominant, has positive off-diagonal entries
  :symmetric             — all diagonal entries positive, no stronger structure detected
  :symmetric_indefinite  — has a non-positive diagonal entry (provably not positive definite)
"""
function classify_component(A::SparseMatrixCSC)::Symbol
    # Check for adjacency matrix: zero diagonal (edge weights may be positive or negative)
    is_adj = true
    for col in axes(A, 2)
        for j in A.colptr[col]:(A.colptr[col+1]-1)
            if A.rowval[j] == col && abs(A.nzval[j]) > 1e-12
                is_adj = false
                break
            end
        end
        is_adj || break
    end
    if is_adj
        return :adjacency
    end

    m_matrix = is_m_matrix(A)

    if m_matrix
        # Check if row sums are zero (Laplacian)
        is_lap = true
        kbuf = Vector{Float64}(undef, maximum(A.colptr[i+1] - A.colptr[i] for i in axes(A, 2)))
        for i in axes(A, 2)
            if abs(kahan_rowsum(A, i, kbuf)) > 1e-12 * abs(A[i, i])
                is_lap = false
                break
            end
        end
        if is_lap
            return :graph_laplacian
        end
        # M-matrix but not Laplacian — check SDD
        if is_sdd(A)
            return :sddm
        end
    else
        if is_sdd(A)
            return :general_sdd
        end
    end

    # Check diagonal positivity
    for col in axes(A, 2)
        for j in A.colptr[col]:(A.colptr[col+1]-1)
            if A.rowval[j] == col
                if A.nzval[j] <= 0
                    return :symmetric_indefinite
                end
                break
            end
        end
    end

    return :symmetric
end

"""
    reduce_to_laplacian(A) -> (components, wrap_solver)

Reduce an SPD matrix to one or more graph Laplacian components suitable for
approximate Cholesky factorization. The reduction is determined automatically:

- **SDDM** (M-matrix, diagonally dominant): ground node reduction → (n+1) Laplacian, 1 component
- **General SDD** (has positive off-diags, diagonally dominant): double cover → 2n Laplacian, decomposed
- **General SPD** (not diagonally dominant): pad diagonal by smallest uniform shift to reach SDD, then double cover

Returns:
- `components::Vector{Tuple{SparseMatrixCSC, Vector{Int}}}` — each is
  (adjacency_matrix, indices_into_reduced_system) for one connected component
- `wrap_solver(solvers::Vector)` — takes a vector of in-place solvers `F!(y, b)`,
  one per component, and returns an in-place `LinearOperator(z, r)` with
  pre-allocated workspace.
"""
function reduce_to_laplacian(A::SparseMatrixCSC)
    n = size(A, 1)
    m_matrix = is_m_matrix(A)

    # Always check strict SDD (rtol=0) for the reduction path.
    # The classifier may have used a tolerance to label the matrix as SDD,
    # but tiny floating-point deficits must be padded before reduction
    # so the ground node / double cover gets an exactly-SDD matrix.
    strictly_sdd = is_sdd(A; rtol=0.0)

    if !strictly_sdd
        max_defect = 0.0
        kbuf = Vector{Float64}(undef, maximum(A.colptr[i+1] - A.colptr[i] for i in 1:n))
        for i in axes(A, 2)
            defect = kahan_abssum_offdiag(A, i, kbuf) - A[i, i]
            if defect > max_defect
                max_defect = defect
            end
        end
        A = A + max_defect * sparse(I, n, n)
    end

    if m_matrix
        return _reduce_sddm(A, n)
    else
        return _reduce_double_cover(A, n)
    end
end

function _reduce_sddm(A::SparseMatrixCSC, n::Int)
    # Compute diagonal excess: d_excess[i] = A[i,i] - sum(|off-diag|)
    # Since A is M-matrix, off-diags ≤ 0, so sum(|off-diag|) = -sum(off-diag) = A[i,i] - rowsum
    d_excess = Vector{Float64}(undef, n)
    kbuf = Vector{Float64}(undef, maximum(A.colptr[i+1] - A.colptr[i] for i in 1:n))
    for i in 1:n
        d_excess[i] = kahan_rowsum(A, i, kbuf)
    end

    # Build (n+1) × (n+1) Laplacian: L_aug = [A - diag(d_excess), -d_excess; -d_excess', sum(d_excess)]
    # The top-left block is A with diagonal reduced to remove the excess → a graph Laplacian
    # The ground node (n+1) connects to each node i with weight d_excess[i]
    # nnz(A) entries from the original matrix + up to 2n edges to ground + 1 ground diagonal
    hint = nnz(A) + 2 * n + 1
    I_aug = Int[]; sizehint!(I_aug, hint)
    J_aug = Int[]; sizehint!(J_aug, hint)
    V_aug = Float64[]; sizehint!(V_aug, hint)

    # Copy A entries, adjusting diagonal
    for col in 1:n
        for j in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[j]
            if row == col
                push!(I_aug, row)
                push!(J_aug, col)
                push!(V_aug, A.nzval[j] - d_excess[col])
            else
                push!(I_aug, row)
                push!(J_aug, col)
                push!(V_aug, A.nzval[j])
            end
        end
    end

    # Ground node edges
    ground = n + 1
    ground_diag = 0.0
    for i in 1:n
        if d_excess[i] > 0
            push!(I_aug, i)
            push!(J_aug, ground)
            push!(V_aug, -d_excess[i])
            push!(I_aug, ground)
            push!(J_aug, i)
            push!(V_aug, -d_excess[i])
            ground_diag += d_excess[i]
        end
    end
    push!(I_aug, ground)
    push!(J_aug, ground)
    push!(V_aug, ground_diag)

    L_aug = sparse(I_aug, J_aug, V_aug, n + 1, n + 1)
    adj = laplace_to_adj(L_aug)

    components = [(adj, collect(1:n+1))]

    # Pre-allocate workspace
    rhs = zeros(n + 1)
    sol = zeros(n + 1)

    function wrap_solver(solvers::Vector, apply_fma::UInt64)
        _build_sddm_operator(solvers[1], rhs, sol, n, apply_fma)
    end

    return (components, wrap_solver)
end

# Function barrier: Julia specializes on the concrete type of F!, eliminating
# dynamic dispatch that would otherwise occur from extracting F! out of Vector{Any}.
function _build_sddm_operator(F!, rhs, sol, n, apply_fma::UInt64)
    function LinearOperator(z::Vector{Float64}, r::Vector{Float64})::Float64
        mu = mean(r)
        @inbounds for i in 1:n
            rhs[i] = r[i] - mu
        end
        rhs[n+1] = 0.0
        F!(sol, rhs)
        ground_val = sol[n+1]
        @inbounds for i in 1:n
            z[i] = sol[i] - ground_val + mu
        end
        return apply_fma
    end
    return LinearOperator
end

function _reduce_double_cover(A::SparseMatrixCSC, n::Int)
    dummy_b = zeros(n)
    A_aug, _ = make_augmented(A, dummy_b)
    aug_n = size(A_aug, 1)  # 2n

    # Decompose into connected components
    co = Laplacians.components(A_aug)
    num_comps = maximum(co)

    if num_comps == 1
        adj = laplace_to_adj(A_aug)
        components = [(adj, collect(1:aug_n))]
    else
        comp_lists = Laplacians.vecToComps(co)
        components = Vector{Tuple{SparseMatrixCSC{Float64,Int}, Vector{Int}}}()
        for cl in comp_lists
            if length(cl) <= 1
                continue  # skip singletons
            end
            Asub = A_aug[cl, cl]
            adj = laplace_to_adj(Asub)
            push!(components, (adj, cl))
        end
    end

    # Pre-extract index arrays and allocate workspace
    comp_indices = [ind for (_, ind) in components]
    rhs = zeros(aug_n)
    sol_aug = zeros(aug_n)

    function wrap_solver(solvers::Vector, apply_fma::UInt64)
        # Build each component applier through a function barrier so Julia
        # specializes on the concrete solver type (avoids dynamic dispatch
        # from extracting F! out of Vector{Any}).
        bound = map(enumerate(solvers)) do (k, F!)
            _build_component_applier(F!, comp_indices[k])
        end
        bound_t = Tuple(bound)

        _build_double_cover_operator(bound_t, rhs, sol_aug, n, apply_fma)
    end

    return (components, wrap_solver)
end

# Function barrier: Julia specializes on the concrete type of F!, so the
# F!(ws_sol, ws_rhs) call inside the closure is a direct call, not dynamic dispatch.
function _build_component_applier(F!, ind)
    ws_rhs = zeros(length(ind))
    ws_sol = zeros(length(ind))
    function apply_component!(rhs_full, sol_full)
        @inbounds for j in eachindex(ind)
            ws_rhs[j] = rhs_full[ind[j]]
        end
        F!(ws_sol, ws_rhs)
        @inbounds for j in eachindex(ind)
            sol_full[ind[j]] = ws_sol[j]
        end
    end
    return apply_component!
end

# Function barrier for the outer LinearOperator. bound_t is a Tuple of
# concrete closure types, so iteration is unrolled at compile time.
function _build_double_cover_operator(bound_t, rhs, sol_aug, n, apply_fma::UInt64)
    function LinearOperator(z::Vector{Float64}, r::Vector{Float64})::Float64
        mu = mean(r)
        @inbounds for i in 1:n
            rhs[i] = r[i] - mu
            rhs[n+i] = -(r[i] - mu)
        end

        for apply_comp! in bound_t
            apply_comp!(rhs, sol_aug)
        end

        @inbounds for i in 1:n
            z[i] = (sol_aug[i] - sol_aug[n+i]) / 2 + mu
        end
        return apply_fma
    end
    return LinearOperator
end

# ═══════════════════════════════════════════════════════════════════════════════
# Vendored LAPACK DSTERF with flop counting — build and load
# ═══════════════════════════════════════════════════════════════════════════════

const LAPACK_SRC_DIR = normpath(joinpath(@__DIR__, "..", "External", "lapack"))
const LAPACK_LIB_DIR = joinpath(LAPACK_SRC_DIR, String(Sys.ARCH))
const LIBDSTERF_NAME = Sys.isapple() ? "libdsterf_flops.dylib" : "libdsterf_flops.so"
const LIBDSTERF = joinpath(LAPACK_LIB_DIR, LIBDSTERF_NAME)

function build_dsterf()
    NativeLibraries.allow_build()
    mkpath(LAPACK_LIB_DIR)
    src = joinpath(LAPACK_SRC_DIR, "dsterf_flops.f")
    isfile(src) || error("DSTERF source not found at $src")
    run(`gfortran -shared -fPIC -O2 -o $LIBDSTERF $src`)
    isfile(LIBDSTERF) || error("Build failed: $LIBDSTERF_NAME not created")
end

function ensure_dsterf_built()
    NativeLibraries.ensure(LIBDSTERF, build_dsterf)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Vendored LAPACK DSTEV (eigenvalues + eigenvectors) with flop counting
# ═══════════════════════════════════════════════════════════════════════════════

const LIBDSTEV_NAME = Sys.isapple() ? "libdstev_flops.dylib" : "libdstev_flops.so"
const LIBDSTEV = joinpath(LAPACK_LIB_DIR, LIBDSTEV_NAME)

function build_dstev()
    NativeLibraries.allow_build()
    mkpath(LAPACK_LIB_DIR)
    src = joinpath(LAPACK_SRC_DIR, "dstev_flops.f")
    isfile(src) || error("DSTEV source not found at $src")
    run(`gfortran -shared -fPIC -O2 -o $LIBDSTEV $src`)
    isfile(LIBDSTEV) || error("Build failed: $LIBDSTEV_NAME not created")
end

function ensure_dstev_built()
    NativeLibraries.ensure(LIBDSTEV, build_dstev)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Vendored LAPACK DSTEGR (MRRR eigensolver) with flop counting
# ═══════════════════════════════════════════════════════════════════════════════

const DSTEGR_SRC_DIR = normpath(joinpath(@__DIR__, "..", "External", "lapack", "dstegr"))
const DSTEGR_LIB_DIR = joinpath(DSTEGR_SRC_DIR, String(Sys.ARCH))
const LIBDSTEGR_NAME = Sys.isapple() ? "libdstegr.dylib" : "libdstegr.so"
const LIBDSTEGR = joinpath(DSTEGR_LIB_DIR, LIBDSTEGR_NAME)

function build_dstegr()
    NativeLibraries.allow_build()
    mkpath(DSTEGR_LIB_DIR)
    srcs = filter(f -> endswith(f, ".f"), readdir(DSTEGR_SRC_DIR; join=true))
    isempty(srcs) && error("No Fortran sources found in $DSTEGR_SRC_DIR")
    run(`gfortran -shared -fPIC -O2 -o $LIBDSTEGR $srcs`)
    isfile(LIBDSTEGR) || error("Build failed: $LIBDSTEGR_NAME not created")
end

function ensure_dstegr_built()
    NativeLibraries.ensure(LIBDSTEGR, build_dstegr)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Symmetric tridiagonal eigensolver with FMA counting
# (vendored LAPACK DSTERF with inlined helpers — see External/lapack/)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tridiag_eigvals!(d::Vector{Float64}, e::Vector{Float64})
        -> (eigenvalues::Vector{Float64}, flops::UInt64)

Compute all eigenvalues of a symmetric tridiagonal matrix in-place using the
Pal-Walker-Kahan variant of the QL/QR algorithm with implicit shift (LAPACK's
DSTERF), instrumented with FMA operation counting.

`d` is the diagonal (length n), `e` is the subdiagonal (length n-1).
Both are modified in-place. Returns sorted eigenvalues and total flop count.

Calls the vendored Fortran library `External/lapack/dsterf_flops.f` via ccall.
"""
function tridiag_eigvals!(d::Vector{Float64}, e::Vector{Float64})
    n = length(d)
    n <= 1 && return (d, UInt64(0))
    length(e) == n - 1 || error("e must have length n-1")
    info = Ref{Int32}(0)
    flops = Ref{Int64}(0)
    ccall((:dsterf_flops_, NativeLibraries.library(LIBDSTERF)), Cvoid,
          (Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ref{Int64}),
          Int32(n), d, e, info, flops)
    info[] != 0 && error("dsterf_flops failed: INFO=$(info[])")
    return (d, UInt64(flops[]))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Symmetric tridiagonal eigensolver (eigenvalues + eigenvectors) with flop counting
# (vendored LAPACK DSTEV with inlined helpers — see External/lapack/)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tridiag_eigen!(d::Vector{Float64}, e::Vector{Float64})
        -> (eigenvalues::Vector{Float64}, eigenvectors::Matrix{Float64}, flops::UInt64)

Compute all eigenvalues and eigenvectors of a symmetric tridiagonal matrix
in-place using the implicit QL/QR algorithm with Givens rotations (LAPACK's
DSTEQR), instrumented with FMA operation counting.

`d` is the diagonal (length n), `e` is the subdiagonal (length n-1).
Both are modified in-place. Returns sorted eigenvalues, orthonormal
eigenvector matrix (column i = eigenvector for eigenvalue i), and total
flop count.

Calls the vendored Fortran library `External/lapack/dstev_flops.f` via ccall.
"""
function tridiag_eigen!(d::Vector{Float64}, e::Vector{Float64})
    n = length(d)
    if n == 0
        return (d, zeros(Float64, 0, 0), UInt64(0))
    end
    if n == 1
        return (d, ones(Float64, 1, 1), UInt64(0))
    end
    length(e) == n - 1 || error("e must have length n-1")
    Z = zeros(Float64, n, n)
    work = zeros(Float64, max(1, 2*n - 2))
    info = Ref{Int32}(0)
    flops = Ref{Int64}(0)
    ccall((:dstev_flops_, NativeLibraries.library(LIBDSTEV)), Cvoid,
          (Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Int32},
           Ptr{Float64}, Ref{Int32}, Ref{Int64}),
          Int32(n), d, e, Z, Int32(n), work, info, flops)
    info[] != 0 && error("dstev_flops failed: INFO=$(info[])")
    return (d, Z, UInt64(flops[]))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Symmetric tridiagonal eigensolver via MRRR (DSTEGR) with FMA counting
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tridiag_eigen_mrrr!(d::Vector{Float64}, e::Vector{Float64})
        -> (eigenvalues::Vector{Float64}, eigenvectors::Matrix{Float64}, flops::UInt64)

Compute all eigenvalues and eigenvectors of a symmetric tridiagonal matrix
in-place using the Multiple Relatively Robust Representations (MRRR) algorithm
(LAPACK's DSTEGR/DSTEMR), instrumented with FMA operation counting.

`d` is the diagonal (length n), `e` is the subdiagonal (length n-1).
Both are modified in-place. Returns sorted eigenvalues, orthonormal
eigenvector matrix (column i = eigenvector for eigenvalue i), and total
flop count.

Calls the vendored Fortran library `External/lapack/dstegr/` via ccall.
"""
function tridiag_eigen_mrrr!(d::Vector{Float64}, e::Vector{Float64})
    n = length(d)
    if n == 0
        return (Float64[], zeros(Float64, 0, 0), UInt64(0))
    end
    if n == 1
        return (copy(d), ones(Float64, 1, 1), UInt64(0))
    end
    length(e) == n - 1 || error("e must have length n-1")
    # Save copies before MRRR (which is in-place) so we can fall back to QL/QR
    d_save = copy(d)
    e_save = copy(e)
    # DSTEGR uses E(1:N), with E(N) as workspace
    e_ext = resize!(copy(e), n)
    e_ext[n] = 0.0
    m = Ref{Int32}(0)
    w = Vector{Float64}(undef, n)
    Z = Matrix{Float64}(undef, n, n)
    isuppz = Vector{Int32}(undef, 2n)
    # Query optimal workspace
    work_query = Vector{Float64}(undef, 1)
    iwork_query = Vector{Int32}(undef, 1)
    info = Ref{Int32}(0)
    flops = Ref{Int64}(0)
    ccall((:dstegr_, NativeLibraries.library(LIBDSTEGR)), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{Int32},
           Ptr{Float64}, Ptr{Float64},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32},
           Ptr{Int32}, Ptr{Float64}, Ref{Int32}, Ptr{Int32}, Ref{Int32},
           Ref{Int32}, Ref{Int64}),
          UInt8('V'), UInt8('A'), Int32(n),
          d, e_ext,
          0.0, 0.0, Int32(0), Int32(0),
          0.0, m, w, Z, Int32(n),
          isuppz, work_query, Int32(-1), iwork_query, Int32(-1),
          info, flops)
    if info[] != 0
        # Workspace query failed — fall back to QL/QR (dstev)
        @warn "dstegr workspace query failed (INFO=$(info[])), falling back to QL/QR (dstev)"
        return tridiag_eigen!(d_save, e_save)
    end
    lwork = Int32(ceil(Int, work_query[1]))
    liwork = iwork_query[1]
    work = Vector{Float64}(undef, lwork)
    iwork = Vector{Int32}(undef, liwork)
    # Reset for actual computation
    flops[] = 0
    ccall((:dstegr_, NativeLibraries.library(LIBDSTEGR)), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{Int32},
           Ptr{Float64}, Ptr{Float64},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32},
           Ptr{Int32}, Ptr{Float64}, Ref{Int32}, Ptr{Int32}, Ref{Int32},
           Ref{Int32}, Ref{Int64}),
          UInt8('V'), UInt8('A'), Int32(n),
          d, e_ext,
          0.0, 0.0, Int32(0), Int32(0),
          0.0, m, w, Z, Int32(n),
          isuppz, work, lwork, iwork, liwork,
          info, flops)
    if info[] != 0
        # MRRR failed — fall back to QL/QR (dstev) with flop counting
        @warn "dstegr failed (INFO=$(info[])), falling back to QL/QR (dstev)"
        return tridiag_eigen!(d_save, e_save)
    end
    m_out = Int(m[])
    return (w[1:m_out], Z[:, 1:m_out], UInt64(flops[]))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Eigenvalue estimation via Lanczos
# ═══════════════════════════════════════════════════════════════════════════════

"""
    estimate_eigrange(A::SparseMatrixCSC, k::Int; seed::Int=42, method::Symbol=:ritz)
        -> (λ_min, λ_max, fma_cost, ritz_min_residual, ritz_max_residual,
            ritz_min_half_gap, ritz_max_half_gap)

Estimate the extreme eigenvalues of a symmetric positive definite matrix `A`
using `k` iterations of the Lanczos algorithm with full reorthogonalization.

`method` controls how λ_min and λ_max are derived from the Ritz values:
- `:ritz` — return raw Ritz values (default)
- `:portable_ritz` — raw Ritz values with the timing binary's deterministic start
- `:guarded` — raw Ritz values and residuals with the timing binary's start
- `:conservative` — widen bounds using Ritz residuals:
  λ_max = θ_max + ρ_max, λ_min = max(θ_min - ρ_min, √ε · λ_max).
  By the Bauer-Fike theorem (alternate formulation for symmetric A with κ₂=1),
  some true eigenvalue lies within ρ of each Ritz value. Kaniel-Paige-Saad
  theory (Saad 1980, Theorem 2) guarantees the i-th Ritz value converges to
  the i-th eigenvalue specifically, provided the starting vector has nonzero
  projection onto each eigenvector (satisfied w.p.1 for random starts). The
  half-gap condition serves as a finite-step convergence check: when satisfied,
  the Ritz value has likely converged to the correct eigenvalue. Misconvergence
  at cluster means is possible for finite k (Parlett 1987) but transient.
  Requires DSTEGR eigenvector computation (counted in fma_cost).
- `:heuristic` — λ_min = θ_min / 30 (safety margin on Ritz minimum)
- `:extrapolation` — Aitken Δ² acceleration on DSTERF prefixes k-2, k-1, k
  to estimate the limit of the geometric convergence of θ_min (costs two extra
  DSTERF calls, counted). Falls back to single geometric step if Aitken
  overshoots or there are too few Lanczos steps.

Returns `(λ_min, λ_max, fma_cost, ritz_min_residual, ritz_max_residual,
          ritz_min_half_gap, ritz_max_half_gap)` where:
- `fma_cost` is the total FMA cost (Lanczos + DSTERF; DSTEV for bounds is not counted)
- `ritz_min_residual`, `ritz_max_residual`: Ritz residual norms `|β_k * s_i[k]|`
  where β_k is the last Lanczos off-diagonal and s_i[k] is the last component
  of the i-th eigenvector of T_k. By the Bauer-Fike theorem (alternate
  formulation, symmetric case), some true eigenvalue of A lies within this
  distance of the corresponding Ritz value.
- `ritz_min_half_gap`, `ritz_max_half_gap`: half the gap between the extreme
  Ritz value and its nearest neighbor, i.e. `(θ₂ - θ₁)/2` and `(θ_k - θ_{k-1})/2`.
  When the residual is smaller than the half-gap, the Ritz interval is isolated
  from its neighbor — Parlett's "standing hypothesis" that each isolated Ritz
  interval contains exactly one eigenvalue (Parlett 1987, "Misconvergence in
  the Lanczos Algorithm"). Combined with Kaniel-Paige-Saad convergence
  guarantees, this serves as a practical finite-step convergence check.
"""
function estimate_eigrange(A::SparseMatrixCSC, k::Int; seed::Int = 42, method::Symbol = :ritz,
                           workspace::Union{Nothing,LanczosWorkspace} = nothing)
    n = size(A, 1)
    k = min(k, n)  # can't do more iterations than matrix dimension

    spmv_plan = _spmv_plan(A)
    rng = StableRNG(seed)

    # Lanczos vectors (stored for full reorthogonalization)
    ws = _lanczos_workspace(workspace, n, k)
    V = ws.basis
    # Tridiagonal entries
    α_diag = zeros(k)
    β_sub = zeros(k - 1)

    # New TNS timing-comparison modes use the same deterministic seed vector
    # as the pure C++ binary. Existing methods retain the historical StableRNG.
    v = if method in (:guarded, :portable_ritz)
        [sin(i * 0.6180339887498948) + cos(i * 0.4142135623730950)
         for i in 1:n]
    else
        randn(rng, n)
    end
    v ./= threaded_norm(v)
    V[:, 1] .= v

    # Work buffers
    w = ws.w
    h_buf = ws.h  # reorthogonalization coefficients

    fma_cost = UInt64(0)

    # Track the last β for Ritz error bounds
    β_last = 0.0

    # Lanczos iteration
    for j in 1:k
        # w = A * v_j
        threaded_spmv!(w, A, view(V, :, j), spmv_plan)
        fma_cost += nnz(A)

        # α_j = v_j' * w
        _sleep_julia_workers_for_blas!()
        α_diag[j] = dot(view(V, :, j), w)
        fma_cost += n

        # w = w - α_j * v_j
        axpy!(-α_diag[j], view(V, :, j), w)
        fma_cost += n

        # w = w - β_{j-1} * v_{j-1}
        if j > 1
            axpy!(-β_sub[j - 1], view(V, :, j - 1), w)
            fma_cost += n
        end

        # Full reorthogonalization (CGS2: classical Gram-Schmidt, twice)
        # Two BLAS-2 passes instead of j BLAS-1 pairs. CGS2 is numerically
        # as stable as modified GS for SPD operators (Giraud-Langou 2005).
        Vj = view(V, :, 1:j)
        h1 = view(h_buf, 1:j)
        mul!(h1, Vj', w)
        mul!(w, Vj, h1, -1.0, 1.0)
        mul!(h1, Vj', w)
        mul!(w, Vj, h1, -1.0, 1.0)
        fma_cost += 4 * n * j

        # β_j = ‖w‖
        if j < k
            β_sub[j] = threaded_norm(w)
            fma_cost += n

            # Breakdown check
            if β_sub[j] < 1e-14
                # Invariant subspace found; β_last = β_sub[j] ≈ 0
                β_last = β_sub[j]
                α_diag = α_diag[1:j]
                β_sub = β_sub[1:j-1]
                break
            end

            # v_{j+1} = w / β_j
            V[:, j + 1] .= w ./ β_sub[j]
            fma_cost += n
        else
            # Last iteration: compute β_k for Ritz error bounds
            β_last = threaded_norm(w)
            fma_cost += n
        end
    end

    # Save tridiagonal entries before they are destroyed
    α_diag_saved = copy(α_diag)
    β_sub_saved = copy(β_sub)

    # Compute Ritz values via DSTERF (eigenvalues only)
    ritz_vals, eigval_flops = tridiag_eigvals!(α_diag, β_sub)

    # Compute eigenvectors via DSTEGR for Ritz error bounds
    α_ev = copy(α_diag_saved)
    β_ev = copy(β_sub_saved)
    _, S, eigvec_flops = tridiag_eigen_mrrr!(α_ev, β_ev)

    # :conservative and :guarded use Ritz residuals, so count DSTEGR;
    # other methods only use DSTERF eigenvalues.
    if method == :conservative || method == :guarded
        fma_cost += eigvec_flops
    else
        fma_cost += eigval_flops
    end

    λ_max = ritz_vals[end]

    # Ritz residual norms: ρ_i = |β_k * s_i[k]| (Bauer-Fike alternate formulation)
    # Some true eigenvalue of A lies within ρ_i of θ_i.
    # Half-gap isolation (ρ_i < gap/2) suggests the nearby eigenvalue is the
    # specific extreme one, but this is a practical indicator, not a proof.
    kk = size(S, 1)
    ritz_min_residual = abs(β_last * S[kk, 1])
    ritz_max_residual = abs(β_last * S[kk, kk])
    ritz_min_half_gap = kk >= 2 ? (ritz_vals[2] - ritz_vals[1]) / 2 : Inf
    ritz_max_half_gap = kk >= 2 ? (ritz_vals[end] - ritz_vals[end-1]) / 2 : Inf

    if method == :conservative
        # Widen using Ritz residuals (Bauer-Fike): some eigenvalue is within ρ of θ.
        # Assumes extreme Ritz values have converged to λ_min/λ_max (standing hypothesis).
        λ_max = ritz_vals[end] + ritz_max_residual
        λ_min = max(ritz_vals[1] - ritz_min_residual, sqrt(eps(1.0)) * λ_max)

    elseif method in (:ritz, :guarded, :portable_ritz)
        λ_min = ritz_vals[1]

    elseif method == :heuristic
        λ_min = ritz_vals[1] / 30.0

    elseif method == :extrapolation
        λ_min_k = ritz_vals[1]
        λ_max_k = ritz_vals[end]
        kk_vals = length(ritz_vals)   # actual k (may be < input k if early breakdown)
        if kk_vals >= 4
            # Aitken Δ² acceleration: fit θ ≈ L + C·r^j using three
            # consecutive prefixes (k-2, k-1, k). Exact for geometric convergence.
            # Applied to both λ_min (converges from above) and λ_max (from below).
            α_km1 = copy(α_diag_saved[1:kk_vals-1])
            β_km1 = copy(β_sub_saved[1:kk_vals-2])
            ritz_km1, fl_km1 = tridiag_eigvals!(α_km1, β_km1)
            fma_cost += fl_km1

            α_km2 = copy(α_diag_saved[1:kk_vals-2])
            β_km2 = copy(β_sub_saved[1:kk_vals-3])
            ritz_km2, fl_km2 = tridiag_eigvals!(α_km2, β_km2)
            fma_cost += fl_km2

            # ── λ_min: converges from above (θ_min_k < θ_min_{k-1}) ──
            λ_min_km1 = ritz_km1[1]
            λ_min_km2 = ritz_km2[1]
            denom_lo = λ_min_k - 2.0 * λ_min_km1 + λ_min_km2
            if denom_lo > 0
                L_lo = λ_min_k - (λ_min_k - λ_min_km1)^2 / denom_lo
                if L_lo > 0 && L_lo < λ_min_k
                    λ_min = L_lo
                elseif λ_min_km1 > 0 && λ_min_k < λ_min_km1
                    λ_min = λ_min_k * (λ_min_k / λ_min_km1)
                else
                    λ_min = λ_min_k
                end
            elseif λ_min_km1 > 0 && λ_min_k > 0 && λ_min_k < λ_min_km1
                λ_min = λ_min_k * (λ_min_k / λ_min_km1)
            else
                λ_min = λ_min_k
            end

            # ── λ_max: converges from below (θ_max_k > θ_max_{k-1}) ──
            # Second difference is negative for convergence from below.
            λ_max_km1 = ritz_km1[end]
            λ_max_km2 = ritz_km2[end]
            denom_hi = λ_max_k - 2.0 * λ_max_km1 + λ_max_km2
            if denom_hi < 0
                L_hi = λ_max_k - (λ_max_k - λ_max_km1)^2 / denom_hi
                # Clamp: must be above the Ritz value (convergence from below)
                if L_hi > λ_max_k
                    λ_max = L_hi
                end
            end
        elseif kk_vals >= 3
            # Not enough points for Aitken; single geometric step
            α_prev = copy(α_diag_saved[1:kk_vals-1])
            β_prev = copy(β_sub_saved[1:kk_vals-2])
            ritz_prev, fl_prev = tridiag_eigvals!(α_prev, β_prev)
            fma_cost += fl_prev
            # λ_min
            λ_min_km1 = ritz_prev[1]
            if λ_min_km1 > 0 && λ_min_k > 0 && λ_min_k < λ_min_km1
                λ_min = λ_min_k * (λ_min_k / λ_min_km1)
            else
                λ_min = λ_min_k
            end
            # λ_max
            λ_max_km1 = ritz_prev[end]
            if λ_max_k > λ_max_km1 && λ_max_km1 > 0
                λ_max = λ_max_k * (λ_max_k / λ_max_km1)
            end
        else
            λ_min = λ_min_k
        end

    else
        error("Unknown method: $method. Use :ritz, :conservative, :heuristic, or :extrapolation.")
    end

    return (λ_min, λ_max, fma_cost, ritz_min_residual, ritz_max_residual,
            ritz_min_half_gap, ritz_max_half_gap)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Gauss-Lanczos quadrature for b'A⁻¹b
# ═══════════════════════════════════════════════════════════════════════════════

"""
    lanczos_quadrature(A::SparseMatrixCSC, b::Vector{Float64};
                       seed::Int=42, max_iters::Int=100)
        -> (q, q10, q30, q100, early_stop, fma_cost, k_final)

Compute the Gauss quadrature approximation to the Stieltjes integral

    b'A⁻¹b / ‖b‖² = ∫ dμ(t) / t

where μ is the spectral measure of A with respect to v₁ = b/‖b‖,
i.e. μ = Σᵢ (v₁'uᵢ)² δ(t - λᵢ) for eigenpairs (λᵢ, uᵢ) of A.

The k-step Lanczos process with starting vector v₁ produces a
tridiagonal matrix Tₖ. The k-point Gauss quadrature rule with nodes
at the Ritz values θᵢ and weights (s₁ᵢ)² (first components of the
eigenvectors of Tₖ) gives:

    qₖ = e₁ᵀ Tₖ⁻¹ e₁ = Σᵢ (s₁ᵢ)² / θᵢ  ≈  b'A⁻¹b / ‖b‖²

For SPD A, this sequence converges monotonically from below to the
true value (Golub & Meurant, "Matrices, Moments and Quadrature with
Applications", 2010). The quadrature value qₖ is computed via the
continued fraction e₁ᵀ Tₖ⁻¹ e₁ = 1 / (α₁ - β₁² / (α₂ - β₂² / ⋯)).

The iteration runs for exactly `max_iters` steps (no value-based early
termination) and records checkpoint values q₁₀, q₃₀, q₁₀₀. The only
early stop is a genuine Lanczos breakdown (invariant subspace,
β_j < 1e-14).

Returns:
- `q`: final quadrature value qₖ
- `q10`, `q30`, `q100`: quadrature at iterations 10, 30, 100
- `early_stop`: whether β-breakdown occurred before reaching max_iters
- `fma_cost`: total FMA operation count (Lanczos + quadrature evaluations)
- `k_final`: iteration at which Lanczos stopped (= max_iters unless breakdown)
"""
function _lanczos_quadrature(A::SparseMatrixCSC, b::Vector{Float64};
                            progress = NumericalProgress(NOOP_PHASE_CALLBACK),
                            seed::Int = 42, max_iters::Int = 100,
                            workspace::Union{Nothing,LanczosWorkspace} = nothing)
    n = size(A, 1)
    length(b) == n || error("b must have length n=$(n)")
    max_iters > 0 || throw(ArgumentError("max_iters must be positive"))
    spmv_plan = _spmv_plan(A)
    k_max = min(max_iters, n)

    # Lanczos vectors (stored for full reorthogonalization)
    ws = _lanczos_workspace(workspace, n, k_max)
    V = ws.basis
    α = zeros(k_max)
    β = zeros(k_max)  # β[j] = β_{j}, off-diagonal between j and j+1

    # v₁ = b / ‖b‖
    b_norm = threaded_norm(b)
    b_norm > 0 || error("b must be nonzero")
    V[:, 1] .= b ./ b_norm

    fma_cost = UInt64(n)  # norm(b) = n FMA

    w = ws.w
    h_buf = ws.h  # reorthogonalization coefficients (CGS2)

    q_hist = zeros(k_max)
    breakdown_k = 0

    for j in 1:k_max
        if progress_due!(progress, j - 1)
            emit_numerical_progress!(progress; iteration=j-1, maxiters=k_max,
                quadrature_value=q_hist[j-1], total_fma=fma_cost)
        end
        # w = A * v_j
        threaded_spmv!(w, A, view(V, :, j), spmv_plan)
        fma_cost += nnz(A)

        # α_j = v_j' * w
        _sleep_julia_workers_for_blas!()
        α[j] = dot(view(V, :, j), w)
        fma_cost += n

        # w = w - α_j * v_j
        axpy!(-α[j], view(V, :, j), w)
        fma_cost += n

        # w = w - β_{j-1} * v_{j-1}
        if j > 1
            axpy!(-β[j-1], view(V, :, j - 1), w)
            fma_cost += n
        end

        # Full reorthogonalization (CGS2: two BLAS-2 passes)
        Vj = view(V, :, 1:j)
        h1 = view(h_buf, 1:j)
        mul!(h1, Vj', w)
        mul!(w, Vj, h1, -1.0, 1.0)
        mul!(h1, Vj', w)
        mul!(w, Vj, h1, -1.0, 1.0)
        fma_cost += 4 * n * j

        # Evaluate q_j = e₁ᵀ T_j⁻¹ e₁ via continued fraction (bottom-up)
        # δ_j = α_j, δ_i = α_i - β_i² / δ_{i+1}
        # q_j = 1 / δ_1
        δ = α[j]
        for i in (j-1):-1:1
            δ = α[i] - β[i]^2 / δ
        end
        q_hist[j] = 1.0 / δ
        fma_cost += 3 * (j - 1) + 1

        # β_j and next Lanczos vector
        if j < k_max
            β[j] = threaded_norm(w)
            fma_cost += n

            if β[j] < 1e-14
                breakdown_k = j
                break
            end

            V[:, j + 1] .= w ./ β[j]
            fma_cost += n
        end
    end

    k_final = breakdown_k > 0 ? breakdown_k : k_max
    q = q_hist[k_final]
    early_stop = breakdown_k > 0

    q10  = q_hist[min(10, k_final)]
    q30  = q_hist[min(30, k_final)]
    q100 = q_hist[min(100, k_final)]

    # If Lanczos broke down early, fill checkpoints beyond breakdown with final value
    if breakdown_k > 0
        breakdown_k < 10  && (q10 = q)
        breakdown_k < 30  && (q30 = q)
        breakdown_k < 100 && (q100 = q)
    end

    return (q, q10, q30, q100, early_stop, fma_cost, k_final)
end

"""
    estimate_eigbounds(A::SparseMatrixCSC; seed=42, m=100, target=:both,
                       lower_tol=1e-2, upper_tol=1e-3,
                       max_restarts=min(100, ceil(Int, 37.5*log(n))))
        -> (λ_min, λ_max, fma_cost, total_iters,
            ritz_min_residual, ritz_max_residual, ritz_min_half_gap, ritz_max_half_gap)

Estimate extreme eigenvalues of a real symmetric matrix `A` using
restarted Lanczos with full reorthogonalization.

With `target=:upper`, stop once the upper Ritz estimate passes the diagnostic
tests and restart from its Ritz vector alone. The three lower-extreme return
fields are `nothing`: no minimum eigenvalue or condition estimate is claimed.
Preparation uses this mode. `target=:both` retains the two-extreme search.
With `target=:radius`, target the Ritz value of largest magnitude, including
negative extremes, for a spectral-norm estimate. It does not pursue a tiny
positive minimum eigenvalue. Return fields retain their algebraic ordering.
These are Ritz estimates, not certified spectral bounds; exhaustion returns
the last estimates and their residuals, which callers must inspect.

Each restart cycle runs `m` Lanczos steps, computes Ritz values and eigenvectors
via DSTEGR (flop-counted), and checks convergence. Each targeted extreme requires:
  1. Residual is small: `ρ_i < tol_i * |θ_i|`
  2. Residual isolates the Ritz value: `ρ_i < (θ₂ - θ₁)/2` (half-gap condition)

where `ρ_i = |β_m * s_i[m]|` is the Ritz residual norm, β_m is the last Lanczos
off-diagonal, and s_i[m] is the last component of the i-th eigenvector of T_m.

By the Bauer-Fike theorem (alternate formulation, symmetric case), some true
eigenvalue lies within ρ_i of θ_i. Kaniel-Paige-Saad theory (Saad 1980,
Theorem 2) guarantees that the i-th Ritz value converges to the i-th eigenvalue
specifically (given nonzero starting vector projection, satisfied by random
init). Condition (2) is Parlett's "standing hypothesis" — a finite-step check
that the Ritz interval is isolated from its neighbor, indicating convergence
to the correct eigenvalue has likely occurred. Misconvergence at cluster means
is possible but transient (Parlett 1987).

The upper extreme retains the tighter `upper_tol = 1e-3` because it is used
for downstream polynomial-preconditioner scaling. The lower extreme uses
`lower_tol = 1e-2`; it is much slower on difficult graph Laplacians and its
primary downstream role is condition estimation. The default restart cap of
100 bounds pathological preparation time while still allowing up to 10,000
Lanczos steps with the default `m = 100`.

Returns `(λ_min, λ_max, fma_cost, total_iters, ritz_min_residual, ritz_max_residual,
          ritz_min_half_gap, ritz_max_half_gap)`.
"""
function _estimate_eigbounds(A::SparseMatrixCSC;
                             progress = NumericalProgress(NOOP_PHASE_CALLBACK),
                             seed::Int = 42, m::Int = 100,
                             target::Symbol = :both,
                             workspace::Union{Nothing,LanczosWorkspace} = nothing,
                             lower_tol::Float64 = EIGBOUNDS_LOWER_TOL_DEFAULT,
                             upper_tol::Float64 = EIGBOUNDS_UPPER_TOL_DEFAULT,
                             max_restarts::Int = _eigbounds_max_restarts(size(A, 1)))
    n = size(A, 1)

    target in (:both, :upper, :radius) || throw(ArgumentError("target must be :both, :upper, or :radius"))
    n > 0 && size(A, 2) == n || throw(ArgumentError("A must be nonempty and square"))
    m > 0 || throw(ArgumentError("m must be positive"))
    lower_tol > 0.0 || throw(ArgumentError("lower_tol must be positive"))
    upper_tol > 0.0 || throw(ArgumentError("upper_tol must be positive"))
    max_restarts > 0 || throw(ArgumentError("max_restarts must be positive"))

    # Trivial case: 1×1 matrix — eigenvalue is the single diagonal entry
    if n == 1
        λ = A[1, 1]
        return target == :upper ? (nothing, λ, UInt64(0), 0, nothing, 0.0, nothing, Inf) :
                                 (λ, λ, UInt64(0), 0, 0.0, 0.0, 0.0, 0.0)
    end

    m = min(m, n)
    nnz_A = nnz(A)

    spmv_plan = _spmv_plan(A)
    rng = StableRNG(seed)

    fma_cost = UInt64(0)
    total_iters = 0

    # Initial random starting vector
    v_start = randn(rng, n)
    v_start ./= threaded_norm(v_start)

    # Work arrays for Lanczos
    ws = _lanczos_workspace(workspace, n, m; both=target == :both, restart=true)
    V = ws.basis
    α_diag = zeros(m)
    β_sub = zeros(m - 1)
    w = ws.w
    y_min = ws.y_min
    y_max = ws.y_max
    h_buf = ws.h  # reorthogonalization coefficients (CGS2)

    # Results from latest cycle
    θ_min = 0.0
    θ_max = 0.0
    ritz_min_residual = Inf
    ritz_max_residual = Inf
    ritz_min_half_gap = Inf
    ritz_max_half_gap = Inf

    estimate_iteration = 0
    for restart in 1:max_restarts
        # Run m-step Lanczos with full reorthogonalization
        V[:, 1] .= v_start
        actual_m = m
        β_last = 0.0

        for j in 1:m
            threaded_spmv!(w, A, view(V, :, j), spmv_plan)
            fma_cost += nnz_A

            _sleep_julia_workers_for_blas!()
            α_diag[j] = dot(view(V, :, j), w)
            fma_cost += n

            axpy!(-α_diag[j], view(V, :, j), w)
            fma_cost += n

            if j > 1
                axpy!(-β_sub[j-1], view(V, :, j-1), w)
                fma_cost += n
            end

            # Full reorthogonalization (CGS2: two BLAS-2 passes)
            Vj = view(V, :, 1:j)
            h1 = view(h_buf, 1:j)
            mul!(h1, Vj', w)
            mul!(w, Vj, h1, -1.0, 1.0)
            mul!(h1, Vj', w)
            mul!(w, Vj, h1, -1.0, 1.0)
            fma_cost += 4 * n * j

            b = threaded_norm(w)
            fma_cost += n
            total_iters += 1
            if progress_due!(progress, total_iters)
                emit_numerical_progress!(progress; iteration=total_iters, cycle=restart,
                    cycle_budget=max_restarts, maxiters=m*max_restarts,
                    upper_estimate=estimate_iteration == 0 ? nothing : θ_max,
                    upper_ritz_residual=estimate_iteration == 0 ? nothing : ritz_max_residual,
                    estimate_iteration=estimate_iteration, total_fma=fma_cost)
            end

            if b < 1e-14
                actual_m = j
                β_last = b
                break
            end

            if j < m
                β_sub[j] = b
                V[:, j+1] .= w ./ b
            else
                β_last = b
            end
        end

        # Compute Ritz values and eigenvectors via DSTEGR (flop-counted).
        # Slicing already allocates a fresh Vector, so no extra copy is needed
        # before passing to the destructive tridiag_eigen_mrrr!.
        α_ev = α_diag[1:actual_m]
        β_ev = β_sub[1:actual_m-1]
        ritz, S, ev_flops = tridiag_eigen_mrrr!(α_ev, β_ev)
        fma_cost += ev_flops

        θ_min = ritz[1]
        θ_max = ritz[end]
        estimate_iteration = total_iters

        # Ritz residual norms (Bauer-Fike) and half-gaps (isolation check)
        ritz_min_residual = abs(β_last * S[actual_m, 1])
        ritz_max_residual = abs(β_last * S[actual_m, actual_m])
        ritz_min_half_gap = actual_m >= 2 ? (ritz[2] - ritz[1]) / 2 : Inf
        ritz_max_half_gap = actual_m >= 2 ? (ritz[end] - ritz[end-1]) / 2 : Inf

        # Convergence requires:
        # 1. Residual is small relative to eigenvalue estimate
        # 2. Ritz interval is isolated from neighbor (standing hypothesis)
        min_converged = ritz_min_residual <= lower_tol * abs(θ_min) && ritz_min_residual < ritz_min_half_gap
        max_converged = ritz_max_residual <= upper_tol * abs(θ_max) && ritz_max_residual < ritz_max_half_gap

        radius_index = abs(θ_min) > abs(θ_max) ? 1 : actual_m
        radius_converged = radius_index == actual_m ? max_converged :
            ritz_min_residual <= upper_tol * abs(θ_min) && ritz_min_residual < ritz_min_half_gap
        if target == :radius ? radius_converged : (target == :upper || min_converged) && max_converged
            break
        end

        # Restart from the requested extreme Ritz vector(s).
        # Reuse preallocated y_min/y_max across restarts via mul!.
        V_actual = view(V, :, 1:actual_m)
        _sleep_julia_workers_for_blas!()
        mul!(y_max, V_actual, view(S, :, target == :radius ? radius_index : actual_m))
        if target == :both
            mul!(y_min, V_actual, view(S, :, 1))
            v_start .= 0.9 .* y_min .+ 0.1 .* y_max
        else
            v_start .= y_max
        end
        v_start ./= threaded_norm(v_start)
    end

    # Return the last cycle, including when the finite work budget is exhausted.
    if target == :upper
        return (nothing, θ_max, fma_cost, total_iters,
                nothing, ritz_max_residual, nothing, ritz_max_half_gap)
    end
    return (θ_min, θ_max, fma_cost, total_iters,
            ritz_min_residual, ritz_max_residual, ritz_min_half_gap, ritz_max_half_gap)
end


function _quadrature_policy()
    Dict{String,Any}("unscaled_max_iters" => QUADRATURE_UNSCALED_MAX_ITERS,
                    "scaled_max_iters" => QUADRATURE_SCALED_MAX_ITERS,
                    "termination" => "fixed_budget_or_breakdown")
end


function _rhs_quadrature_metadata(A::SparseMatrixCSC, b::Vector{Float64},
                                  lambda_max::Real; scaled::Bool = false,
                                  workspace = nothing,
                                  progress_callback = NOOP_PHASE_CALLBACK)
    budget = scaled ? QUADRATURE_SCALED_MAX_ITERS : QUADRATURE_UNSCALED_MAX_ITERS
    q, q10, q30, _, early, _, k = lanczos_quadrature(A, b; max_iters = budget, workspace=workspace,
        progress_callback = phase_callback_with_context(progress_callback; scaling = scaled ? "scaled" : "unscaled"))
    prefix = scaled ? "scaled_quadrature" : "quadrature"
    # Do not label a shorter budget's final estimate as a 100-step checkpoint.
    Dict{String,Any}(
        prefix => q * lambda_max,
        prefix * "_k10" => k >= 10 ? q10 * lambda_max : nothing,
        prefix * "_k30" => k >= 30 ? q30 * lambda_max : nothing,
        prefix * "_k60" => k == 60 ? q * lambda_max : nothing,
        prefix * "_k100" => nothing,
        prefix * "_early_stop" => early,
        prefix * "_k_final" => k,
        prefix * "_max_iters" => budget,
    )
end


function _upper_eigenvalue_policy(n::Int)
    Dict{String,Any}("target" => "upper", "lower_tolerance" => nothing,
                    "upper_tolerance" => EIGBOUNDS_UPPER_TOL_DEFAULT,
                    "max_restarts" => _eigbounds_max_restarts(n),
                    "subspace_dimension" => min(100, n))
end


function _upper_eigenvalue_metadata(A::SparseMatrixCSC; prefix::String = "", workspace=nothing,
                                    progress_callback = NOOP_PHASE_CALLBACK)
    _, hi, _, iters, _, residual, _, gap = estimate_eigbounds(A; target = :upper, workspace=workspace, progress_callback =
        phase_callback_with_context(progress_callback; scaling = isempty(prefix) ? "unscaled" : "scaled"))
    converged = isfinite(hi) && hi > 0 && isfinite(residual) &&
                residual <= EIGBOUNDS_UPPER_TOL_DEFAULT * abs(hi) && residual < gap
    converged || @warn "Upper eigenvalue estimate exhausted its budget without passing diagnostics" estimate=hi residual=residual iterations=iters
    Dict{String,Any}(
        prefix * "lambda_min" => nothing,
        prefix * "lambda_max" => hi,
        prefix * "ritz_min_residual" => nothing,
        prefix * "ritz_min_half_gap" => nothing,
        prefix * "ritz_max_residual" => _json_safe(residual),
        prefix * "ritz_max_half_gap" => _json_safe(gap),
        prefix * "lambda_min_status" => "not_estimated",
        prefix * "lambda_max_status" => converged ? "ritz_converged" : "budget_exhausted",
        prefix * "eigenvalue_iterations" => iters,
    )
end

"""
    generate_component_rhs(A, component_index) -> (seed, b)

Generate a deterministic seed and RHS for a component. The seed is
generated from a StableRNG seeded by (123456789 + component_index)
to ensure reproducibility and independence across components.
"""
function generate_component_rhs(A::SparseMatrixCSC, component_index::Int)
    m = size(A, 1)
    rng = StableRNG(123456789 + component_index)

    seed = Vector{Float64}(undef, m)
    for _ in 1:max(1, round(Int, log2(m)))
        seed .= rand(rng, m)
    end

    amt = max(2, round(Int, log2(m)))
    amt % 2 == 1 && (amt += 1)
    amt = min(amt, m)  # don't exceed dimension

    top_indices = partialsortperm(seed, 1:amt, rev=true)
    seed .= 0.0
    seed[top_indices] .= 1.0
    half = shuffle(rng, collect(top_indices))[1:div(amt, 2)]
    seed[half] .= -1.0

    b = A * seed
    return seed, b
end

"""
    generate_component_rhs(A, component_index, num_rhs) -> (seeds, B)

Generate `num_rhs` deterministic RHS vectors for a component. The RNG is
seeded by (123456789 + component_index) and warmed up with `log2(m)-1`
draws. Each RHS then consumes one draw from the same stream and undergoes
the same sparsification process (top `2⌈log2(m)⌉` entries set to ±1).

The first RHS is identical to the one produced by the single-vector
`generate_component_rhs(A, component_index)`. Returns `seeds` (n × num_rhs)
and `B` (n × num_rhs) where `B[:,j] = A * seeds[:,j]`.
"""
function generate_component_rhs(A::SparseMatrixCSC, component_index::Int, num_rhs::Int)
    m = size(A, 1)
    rng = StableRNG(123456789 + component_index)

    warmup = max(1, round(Int, log2(m))) - 1
    buf = Vector{Float64}(undef, m)
    for _ in 1:warmup
        buf .= rand(rng, m)
    end

    amt = max(2, round(Int, log2(m)))
    amt % 2 == 1 && (amt += 1)
    amt = min(amt, m)

    seeds = Matrix{Float64}(undef, m, num_rhs)
    B = Matrix{Float64}(undef, m, num_rhs)

    for j in 1:num_rhs
        seeds[:, j] .= rand(rng, m)

        top_indices = partialsortperm(view(seeds, :, j), 1:amt, rev=true)
        seeds[:, j] .= 0.0
        seeds[top_indices, j] .= 1.0
        half = shuffle(rng, collect(top_indices))[1:div(amt, 2)]
        seeds[half, j] .= -1.0

        B[:, j] .= A * view(seeds, :, j)
    end

    return seeds, B
end

"""
    select_prolific_node(A) -> (node_index, tiebreak_criterion)

Select the node to pin (remove) when converting a graph Laplacian to SPD.
Tiebreaking hierarchy (all criteria are ordering-agnostic):
  1. Highest unweighted degree (number of off-diagonal nonzeros)
  2. Highest weighted degree (sum of absolute off-diagonal values)
  3. Highest second-neighbor degree (sum of neighbors' unweighted degrees)

If all three criteria tie (e.g. vertex-transitive graphs), falls back to the
lowest node index, which is ordering-dependent. The criterion string will
report "second_neighbor_degree_tied" in this case.

Returns the 1-based index of the selected node and a string indicating
which criterion resolved the selection.
"""
function select_prolific_node(A::SparseMatrixCSC)
    n = size(A, 1)

    # Level 1: unweighted degree (off-diagonal nonzeros per column)
    degrees = Vector{Int}(undef, n)
    for i in 1:n
        count = 0
        for j in A.colptr[i]:(A.colptr[i+1]-1)
            if A.rowval[j] != i
                count += 1
            end
        end
        degrees[i] = count
    end

    max_deg = maximum(degrees)
    candidates = findall(==(max_deg), degrees)

    if length(candidates) == 1
        return candidates[1], "degree"
    end

    # Level 2: weighted degree (sum of absolute off-diagonal values)
    kbuf = Vector{Float64}(undef, maximum(A.colptr[i+1] - A.colptr[i] for i in candidates))
    wdegrees = [kahan_abssum_offdiag(A, i, kbuf) for i in candidates]
    max_wdeg = maximum(wdegrees)
    tied = findall(i -> wdegrees[i] == max_wdeg, eachindex(wdegrees))
    candidates = candidates[tied]

    if length(candidates) == 1
        return candidates[1], "weighted_degree"
    end

    # Level 3: second-neighbor degree (sum of neighbors' unweighted degrees)
    neighbor_deg_sums = Vector{Int}(undef, length(candidates))
    for (ci, node) in enumerate(candidates)
        s = 0
        for j in A.colptr[node]:(A.colptr[node+1]-1)
            nbr = A.rowval[j]
            if nbr != node
                s += degrees[nbr]
            end
        end
        neighbor_deg_sums[ci] = s
    end

    max_nbr_deg = maximum(neighbor_deg_sums)
    tied = findall(==(max_nbr_deg), neighbor_deg_sums)

    # If still tied, falls back to lowest node index (ordering-dependent)
    return candidates[tied[1]], length(tied) == 1 ? "second_neighbor_degree" : "second_neighbor_degree_tied"
end

"""Replace Inf/NaN with null-safe sentinels for JSON serialization."""
_json_safe(x::Float64) = isfinite(x) ? x : (isinf(x) ? (x > 0 ? 1.0e308 : -1.0e308) : 0.0)

_plain_json_value(x) = x
_plain_json_value(x::AbstractVector) = Any[_plain_json_value(v) for v in x]
_plain_json_value(x::AbstractDict) =
    Dict{String,Any}(String(k) => _plain_json_value(v) for (k, v) in pairs(x))
_plain_json_value(x::JSON3.Object) =
    Dict{String,Any}(String(k) => _plain_json_value(v) for (k, v) in pairs(x))
_plain_json_value(x::JSON3.Array) = Any[_plain_json_value(v) for v in x]

function _atomic_output_path(path::String)
    return path * ".tmp.$(gethostname()).$(getpid()).$(time_ns())"
end

function _atomic_write_json(path::String, value)
    mkpath(dirname(path))
    tmp = _atomic_output_path(path)
    try
        open(tmp, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            flush(io)
            systemerror("fsync", ccall(:fsync, Cint, (Cint,), Base.fd(io)) != 0)
        end
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end

function _atomic_write_npy_zstd_1d_i64(path::String, values)
    tmp = _atomic_output_path(path)
    try
        write_npy_zstd_1d_i64(tmp, values)
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end


function _atomic_write_npy_zstd_1d_f64(path::String, values)
    tmp = _atomic_output_path(path)
    try
        write_npy_zstd_1d_f64(tmp, values)
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end


include("ordering_process.jl")

function _component_prepare_fingerprint(A::SparseMatrixCSC)
    ctx = SHA.SHA2_256_CTX()
    dims = Int64[size(A, 1), size(A, 2), nnz(A)]
    SHA.update!(ctx, reinterpret(UInt8, dims))
    SHA.update!(ctx, reinterpret(UInt8, A.colptr))
    SHA.update!(ctx, reinterpret(UInt8, A.rowval))
    SHA.update!(ctx, reinterpret(UInt8, A.nzval))
    return bytes2hex(SHA.digest!(ctx))
end


function _prepare_checkpoint_path(comp_dir::String, ci::Int)
    return joinpath(comp_dir, "prepare_state", "c$ci.json")
end


function _new_prepare_checkpoint(name::String, ci::Int, A::SparseMatrixCSC,
                                 fingerprint::String, amd_max_nnz::Int)
    return Dict{String,Any}(
        "schema_version" => PREPARE_CHECKPOINT_SCHEMA_VERSION,
        "source" => name,
        "component_index" => ci,
        "dimension" => size(A, 1),
        "nnz" => nnz(A),
        "fingerprint_sha256" => fingerprint,
        "amd_max_nnz" => amd_max_nnz,
        "eigenbounds_policy" => _upper_eigenvalue_policy(size(A, 1)),
        "quadrature_policy" => _quadrature_policy(),
        "phases" => Dict{String,Any}(),
        "updated_at" => time(),
    )
end


function _load_prepare_checkpoint(path::String, name::String, ci::Int,
                                  A::SparseMatrixCSC, fingerprint::String,
                                  amd_max_nnz::Int)
    if !isfile(path)
        return _new_prepare_checkpoint(name, ci, A, fingerprint, amd_max_nnz)
    end

    checkpoint = try
        _plain_json_value(JSON3.read(read(path, String)))
    catch e
        error("prepare checkpoint is unreadable at $path: $(sprint(showerror, e)); " *
              "remove the checkpoint before retrying")
    end

    expected = _new_prepare_checkpoint(name, ci, A, fingerprint, amd_max_nnz)
    identity_keys = (
        "schema_version", "source", "component_index", "dimension", "nnz",
        "fingerprint_sha256", "amd_max_nnz",
    )
    mismatches = String[]
    for key in identity_keys
        get(checkpoint, key, nothing) == expected[key] || push!(mismatches, key)
    end
    isempty(mismatches) ||
        error("prepare checkpoint identity mismatch at $path for " *
              join(mismatches, ", ") * "; remove the checkpoint before retrying")
    get!(checkpoint, "phases", Dict{String,Any}())
    if get(checkpoint, "eigenbounds_policy", nothing) != expected["eigenbounds_policy"]
        # Retain expensive structural work when only the spectral policy changes.
        for phase in ("unscaled_eigenbounds", "scaled_eigenbounds")
            delete!(checkpoint["phases"], phase)
            _invalidate_prepare_dependents!(checkpoint["phases"], phase)
        end
        checkpoint["eigenbounds_policy"] = expected["eigenbounds_policy"]
        checkpoint["updated_at"] = time()
        _atomic_write_json(path, checkpoint)
    end
    if get(checkpoint, "quadrature_policy", nothing) != expected["quadrature_policy"]
        delete!(checkpoint["phases"], "quadrature")
        _invalidate_prepare_dependents!(checkpoint["phases"], "quadrature")
        checkpoint["quadrature_policy"] = expected["quadrature_policy"]
        checkpoint["updated_at"] = time()
        _atomic_write_json(path, checkpoint)
    end
    return checkpoint
end


function _checkpoint_artifacts_exist(entry::AbstractDict, checkpoint_path::String)
    artifacts = get(entry, "artifacts", Any[])
    checkpoint_dir = dirname(abspath(checkpoint_path))
    return all(artifacts) do path
        path isa AbstractString || return false
        artifact_path = isabspath(path) ? path : joinpath(checkpoint_dir, path)
        return isfile(artifact_path)
    end
end


const _PREPARE_PHASE_DEPENDENTS = Dict{String,Vector{String}}(
    "component_storage" => ["component_metadata"],
    "structural_summary" => ["component_metadata"],
    "unscaled_eigenbounds" => ["quadrature", "component_metadata"],
    "triangles" => ["component_metadata"],
    "diameter" => ["component_metadata"],
    "ordering_natural" => ["component_metadata"],
    "ordering_rcm" => ["component_metadata"],
    "ordering_amd" => ["component_metadata"],
    "ordering_metis" => ["component_metadata"],
    "scaled_analysis" => ["component_metadata"],
    "scaled_eigenbounds" => ["quadrature", "component_metadata"],
    "quadrature" => ["component_metadata"],
)


function _invalidate_prepare_dependents!(phases::AbstractDict, phase::String)
    for dependent in get(_PREPARE_PHASE_DEPENDENTS, phase, String[])
        if haskey(phases, dependent)
            delete!(phases, dependent)
            _invalidate_prepare_dependents!(phases, dependent)
        end
    end
    return nothing
end


function _run_checkpointed_prepare_phase!(f::Function,
                                          checkpoint::Dict{String,Any},
                                          checkpoint_path::String,
                                          phase::String;
                                          artifacts::Vector{String} = String[],
                                          callback = _noop_prepare_checkpoint_callback,
                                          progress_callback = NOOP_PHASE_CALLBACK)
    phases = checkpoint["phases"]
    ci = checkpoint["component_index"]
    function report_progress(subphase_event::String; details...)
        event = Dict{String,Any}(
            "event" => "phase_progress",
            "phase" => "prepare",
            "subphase" => phase,
            "subphase_event" => subphase_event,
            "component" => ci,
            "checkpoint_path" => checkpoint_path,
            "timestamp" => time(),
        )
        for (key, value) in details
            event[String(key)] = value
        end
        progress_callback(event)
    end

    if haskey(phases, phase)
        entry = phases[phase]
        if _checkpoint_artifacts_exist(entry, checkpoint_path)
            println("    [c$ci] Resuming after completed $phase checkpoint")
            flush(stdout)
            report_progress(
                "resumed";
                checkpoint_started_at = get(entry, "started_at", nothing),
                historical_elapsed_seconds = get(entry, "elapsed_seconds", nothing),
                skipped = get(entry["payload"], "skipped", false),
                skip_reason = get(entry["payload"], "reason", nothing),
            )
            return _plain_json_value(entry["payload"])
        end
        println("    [c$ci] Invalidating $phase checkpoint: artifact missing")
        flush(stdout)
        delete!(phases, phase)
        _invalidate_prepare_dependents!(phases, phase)
        checkpoint["updated_at"] = time()
        _atomic_write_json(checkpoint_path, checkpoint)
        report_progress("invalidated"; reason = "artifact_missing")
    end

    # A missing phase can occur after a crash between artifact creation and
    # checkpoint commit or after manual cache repair. Its consumers must be
    # rebuilt even if their own payloads still exist.
    _invalidate_prepare_dependents!(phases, phase)

    started_at = time()
    report_progress("start"; subphase_started_at = started_at)
    payload = try
        _plain_json_value(run_scoped_phase(child -> (applicable(f, child) ? f(child) : f()), progress_callback, phase;
            component = ci, checkpoint_path = checkpoint_path))
    catch e
        failed_at = time()
        report_progress(
            "error";
            subphase_started_at = started_at,
            elapsed_seconds = failed_at - started_at,
            error = sprint(showerror, e),
        )
        rethrow()
    end
    completed_at = time()
    checkpoint_dir = dirname(abspath(checkpoint_path))
    # An intentional skip is a durable decision, not a missing artifact to retry.
    recorded_artifacts = get(payload, "skipped", false) ? String[] :
        [relpath(abspath(path), checkpoint_dir) for path in artifacts]
    phases[phase] = Dict{String,Any}(
        "payload" => payload,
        "artifacts" => recorded_artifacts,
        "started_at" => started_at,
        "completed_at" => completed_at,
        "elapsed_seconds" => completed_at - started_at,
    )
    checkpoint["updated_at"] = completed_at
    _atomic_write_json(checkpoint_path, checkpoint)
    report_progress(
        "complete";
        subphase_started_at = started_at,
        elapsed_seconds = completed_at - started_at,
        skipped = get(payload, "skipped", false),
        skip_reason = get(payload, "reason", nothing),
    )
    callback(Dict{String,Any}(
        "event" => "prepare_checkpoint_written",
        "phase" => phase,
        "component" => checkpoint["component_index"],
        "checkpoint_path" => checkpoint_path,
        "timestamp" => completed_at,
        "elapsed_seconds" => completed_at - started_at,
    ))
    return payload
end

"""
    _compute_component_metadata(Asub, name, ci, comp_dir, reorder_dir) -> Dict{String,Any}

Compute and cache all metadata for a single component matrix. Writes the
component CSC structure as 3 .npy.zstd files (colptr/rowval/nzval — preserves
explicit zeros), reordering permutations as .npy.zstd, structural histograms
as .npy.zstd, computes spectral norm, structural properties, and SDD status.
Returns the metadata dict.

This helper processes whatever matrix it receives — any pinning or
conversion must be done by the caller before invoking this function.
"""
function _compute_component_metadata(Asub::SparseMatrixCSC, name::String, ci::Int,
                                     comp_dir::String, reorder_dir::String;
                                     amd_max_nnz::Int=AMD_MAX_NNZ_DEFAULT,
                                     ordering_timeout_seconds::Real=ORDERING_TIMEOUT_SECONDS_DEFAULT,
                                     checkpoint_callback=_noop_prepare_checkpoint_callback,
                                     progress_callback=NOOP_PHASE_CALLBACK)
    dim = size(Asub, 1)
    nnz_val = nnz(Asub)
    fingerprint = _component_prepare_fingerprint(Asub)
    checkpoint_path = _prepare_checkpoint_path(comp_dir, ci)
    checkpoint = _load_prepare_checkpoint(
        checkpoint_path, name, ci, Asub, fingerprint, amd_max_nnz)

    # Write the component CSC as 3 .npy.zstd files. Replaces the legacy
    # `c{i}.mtx.zstd` (MatrixMarket coordinate, lossy on round-trip via mmread).
    component_files = Dict{String,String}(
        "colptr" => "c$ci.colptr.npy.zstd",
        "rowval" => "c$ci.rowval.npy.zstd",
        "nzval"  => "c$ci.nzval.npy.zstd",
    )
    component_paths = [joinpath(comp_dir, component_files[key])
                       for key in ("colptr", "rowval", "nzval")]
    storage = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "component_storage";
        artifacts = component_paths,
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        println("    [c$ci] Writing component CSC (dim=$dim, nnz=$nnz_val) as .npy.zstd..."); flush(stdout)
        _atomic_write_npy_zstd_1d_i64(component_paths[1], Asub.colptr)
        _atomic_write_npy_zstd_1d_i64(component_paths[2], Asub.rowval)
        _atomic_write_npy_zstd_1d_f64(component_paths[3], Asub.nzval)
        Dict{String,Any}("files" => component_files)
    end
    component_files = Dict{String,String}(
        String(k) => String(v) for (k, v) in pairs(storage["files"]))

    structural = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "structural_summary";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        classification = classify_component(Asub)
        nnz_per_row = [Asub.colptr[i+1] - Asub.colptr[i] for i in 1:dim]
        sorted_nnz = sort(nnz_per_row)
        median_nnz_per_row = dim % 2 == 1 ? Float64(sorted_nnz[div(dim+1,2)]) :
                             (sorted_nnz[div(dim,2)] + sorted_nnz[div(dim,2)+1]) / 2.0
        mean_nnz_per_row = mean(nnz_per_row)
        std_nnz_per_row = sqrt(sum((x - mean_nnz_per_row)^2 for x in nnz_per_row) / dim)
        dd_ratio_raw = diagonal_dominance_ratio(Asub)
        Dict{String,Any}(
            "classification" => string(classification),
            "min_nnz_per_row" => minimum(nnz_per_row),
            "max_nnz_per_row" => maximum(nnz_per_row),
            "avg_nnz_per_row" => mean_nnz_per_row,
            "median_nnz_per_row" => median_nnz_per_row,
            "std_nnz_per_row" => std_nnz_per_row,
            "is_m_matrix" => is_m_matrix(Asub),
            "diagonal_dominance_ratio" =>
                isfinite(dd_ratio_raw) ? dd_ratio_raw : -1.0,
        )
    end
    classification = Symbol(structural["classification"])
    analysis_workspace = LanczosWorkspace()

    # Eigenvalue bounds via adaptive Lanczos
    eigbound_max_restarts = _eigbounds_max_restarts(dim)
    unscaled_eigenbounds = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "unscaled_eigenbounds";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        println("    [c$ci] Estimating upper eigenvalue (unscaled, upper_tol=$EIGBOUNDS_UPPER_TOL_DEFAULT, max_restarts=$eigbound_max_restarts)..."); flush(stdout)
        t0 = time()
        fields = _upper_eigenvalue_metadata(Asub; workspace=analysis_workspace, progress_callback = child_progress)
        println("    [c$ci] Upper estimate done in $(round(time()-t0; digits=1))s: λ_max=$(fields["lambda_max"])"); flush(stdout)
        fields
    end
    lambda_max = Float64(unscaled_eigenbounds["lambda_max"])

    # Do not retain a potentially enormous basis across graph analysis and
    # ordering subprocesses. Reuse resumes for scaled analysis + all RHS.
    _release_lanczos_workspace!(analysis_workspace)

    # Per-vertex triangle count: for each vertex i, count pairs (j,k) in N(i)
    # that are also connected. Each triangle is counted once per vertex, so
    # total_triangles = sum(per_vertex_counts) / 3.
    triangle_hist_filename = "triangle_hist_c$(ci).npy.zstd"
    triangle_hist_path = joinpath(reorder_dir, triangle_hist_filename)
    triangles = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "triangles";
        artifacts = [triangle_hist_path],
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        println("    [c$ci] Counting triangles..."); flush(stdout)
        t0 = time()
        vertex_triangles = _triangle_vertex_counts(Asub)
        total_triangles = sum(vertex_triangles) ÷ 3
        max_vertex_triangles = maximum(vertex_triangles)
        triangle_hist = zeros(Int, max_vertex_triangles + 1)
        for count in vertex_triangles
            triangle_hist[count + 1] += 1
        end
        _atomic_write_npy_zstd_1d_i64(triangle_hist_path, triangle_hist)
        println("    [c$ci] Triangle counting done in $(round(time()-t0; digits=1))s: total=$total_triangles"); flush(stdout)
        Dict{String,Any}(
            "total_triangles" => total_triangles,
            "max_vertex_triangles" => max_vertex_triangles,
            "triangle_hist_file" => triangle_hist_filename,
        )
    end
    total_triangles = Int(triangles["total_triangles"])
    max_vertex_triangles = Int(triangles["max_vertex_triangles"])

    # Graph diameter lower bound via double BFS (pseudo-peripheral node search).
    # Start from node 1, BFS to find farthest node, BFS again from there.
    # The eccentricity of the second BFS is a lower bound on the diameter.
    function _diameter_lower_bound(A)
        n = size(A, 1)
        dist = fill(-1, n)
        queue = Vector{Int}(undef, n)

        function _bfs!(start)
            dist .= -1
            dist[start] = 0
            head, tail = 1, 1
            queue[1] = start
            farthest = start
            max_dist = 0
            while head <= tail
                u = queue[head]; head += 1
                for j in A.colptr[u]:(A.colptr[u+1]-1)
                    v = A.rowval[j]
                    if dist[v] == -1
                        dist[v] = dist[u] + 1
                        tail += 1
                        queue[tail] = v
                        if dist[v] > max_dist
                            max_dist = dist[v]
                            farthest = v
                        end
                    end
                end
            end
            return farthest, max_dist
        end

        far1, _ = _bfs!(1)
        _, diam_lb = _bfs!(far1)
        return diam_lb
    end
    diameter = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "diameter";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        println("    [c$ci] Computing diameter lower bound..."); flush(stdout)
        t0 = time()
        diameter_lower_bound = _diameter_lower_bound(Asub)
        println("    [c$ci] Diameter bound done in $(round(time()-t0; digits=1))s: lb=$diameter_lower_bound"); flush(stdout)
        Dict{String,Any}("diameter_lower_bound" => diameter_lower_bound)
    end
    diameter_lower_bound = Int(diameter["diameter_lower_bound"])

    # Compute ordering statistics in a single pass over nonzeros.
    # Bandwidth of A[perm,perm] without forming it: iterate original nonzeros
    # (row,col) and measure max |inv_perm[row] - inv_perm[col]| in new indices.
    # Also computes profile (envelope), RMS bandwidth, and edge length histogram.
    function _ordering_stats(A, inv_perm)
        n = size(A, 1)
        row_bw = zeros(Int, n)       # per-row bandwidth
        hist = zeros(Int, n)         # edge length histogram: hist[l] = count of edges with length l
        for col in 1:n
            for j in A.colptr[col]:(A.colptr[col+1]-1)
                row = A.rowval[j]
                l = abs(inv_perm[row] - inv_perm[col])
                row_bw[inv_perm[col]] = max(row_bw[inv_perm[col]], l)
                if l > 0
                    hist[l] += 1
                end
            end
        end
        bandwidth = maximum(row_bw)
        profile = sum(Int128, row_bw)
        rms_bw = Float64(sqrt(BigFloat(sum(x -> Int128(x)^2, row_bw)) / n))
        # Percentiles from upper-triangular edge lengths (hist double-counts symmetric entries)
        total_edges = sum(hist) ÷ 2  # each off-diag pair counted twice
        p90_thresh = ceil(Int, 0.90 * total_edges)
        p99_thresh = ceil(Int, 0.99 * total_edges)
        cum = 0
        p90 = bandwidth
        p99 = bandwidth
        for l in 1:bandwidth
            cum += hist[l] ÷ 2
            if cum >= p90_thresh && p90 == bandwidth
                p90 = l
            end
            if cum >= p99_thresh
                p99 = l
                break
            end
        end
        return (bandwidth=bandwidth, profile=profile, rms_bandwidth=rms_bw,
                edge_length_90pct=p90, edge_length_99pct=p99, edge_length_hist=hist[1:bandwidth])
    end

    natural_inv = collect(1:dim)
    ordering_stats = Dict{String,Any}()
    natural_hist_filename = "natural_edge_hist_c$(ci).npy.zstd"
    natural_hist_path = joinpath(reorder_dir, natural_hist_filename)
    natural_ordering = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "ordering_natural";
        artifacts = [natural_hist_path],
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        ns = _ordering_stats(Asub, natural_inv)
        _atomic_write_npy_zstd_1d_i64(natural_hist_path, ns.edge_length_hist)
        Dict{String,Any}(
            "bandwidth" => ns.bandwidth,
            "profile" => ns.profile,
            "rms_bandwidth" => ns.rms_bandwidth,
            "edge_length_90th_percentile" => ns.edge_length_90pct,
            "edge_length_99th_percentile" => ns.edge_length_99pct,
            "edge_length_hist_file" => natural_hist_filename,
        )
    end
    ordering_stats["natural"] = natural_ordering

    # Compute and cache reorderings
    println("    [c$ci] Computing reorderings (RCM, AMD, METIS)..."); flush(stdout)
    t0 = time()
    reorderings = Dict{String,String}()
    reordering_skips = Dict{String,Any}()
    for ord_name in ("rcm", "amd", "metis")
        phase = "ordering_$ord_name"
        if ord_name == "amd" && nnz_val > amd_max_nnz
            skipped = _run_checkpointed_prepare_phase!(
                checkpoint, checkpoint_path, phase;
                callback = checkpoint_callback,
                progress_callback = progress_callback,
            ) do child_progress
                println("    [c$ci] Skipping AMD: nnz=$nnz_val exceeds limit $amd_max_nnz"); flush(stdout)
                Dict{String,Any}(
                    "skipped" => true,
                    "reason" => "nnz_limit",
                    "nnz" => nnz_val,
                    "max_nnz" => amd_max_nnz,
                    "skipped_at" => time(),
                )
            end
            reordering_skips[ord_name] = Dict{String,Any}(
                key => value for (key, value) in pairs(skipped) if key != "skipped")
            continue
        end
        hist_filename = "$(ord_name)_edge_hist_c$(ci).npy.zstd"
        perm_filename = "$(ord_name)_c$(ci).npy.zstd"
        hist_path = joinpath(reorder_dir, hist_filename)
        perm_path = joinpath(reorder_dir, perm_filename)
        ordering = _run_checkpointed_prepare_phase!(
            checkpoint, checkpoint_path, phase;
            artifacts = [hist_path, perm_path],
            callback = checkpoint_callback,
            progress_callback = progress_callback,
        ) do child_progress
            ordering_t0 = time()
            println("    [c$ci] Computing $(uppercase(ord_name)) ordering and statistics..."); flush(stdout)
            outcome = _bounded_ordering(
                comp_dir, component_files, dim, ord_name, reorder_dir, ci;
                timeout_seconds = ordering_timeout_seconds)
            if outcome.skipped
                println("    [c$ci] Skipping $(uppercase(ord_name)): time limit reached"); flush(stdout)
                return Dict{String,Any}(
                    "skipped" => true, "reason" => "time_limit",
                    "timeout_seconds" => Float64(ordering_timeout_seconds),
                    "elapsed_seconds" => outcome.elapsed_seconds,
                    "skipped_at" => time(), "worker_log" => outcome.worker_log,
                )
            end
            perm = outcome.permutation

            # Ordering stats: perm maps new→old, so inv_perm maps old→new.
            inv_perm = Vector{Int}(undef, dim)
            for i in 1:dim
                inv_perm[perm[i]] = i
            end
            os = _ordering_stats(Asub, inv_perm)
            _atomic_write_npy_zstd_1d_i64(hist_path, os.edge_length_hist)
            _atomic_write_npy_zstd_1d_i64(perm_path, perm)
            println("    [c$ci] $(uppercase(ord_name)) done in $(round(time()-ordering_t0; digits=1))s"); flush(stdout)
            Dict{String,Any}(
                "bandwidth" => os.bandwidth,
                "profile" => os.profile,
                "rms_bandwidth" => os.rms_bandwidth,
                "edge_length_90th_percentile" => os.edge_length_90pct,
                "edge_length_99th_percentile" => os.edge_length_99pct,
                "edge_length_hist_file" => hist_filename,
                "permutation_file" => perm_filename,
                "ordering_elapsed_seconds" => outcome.elapsed_seconds,
                "ordering_timeout_seconds" => Float64(ordering_timeout_seconds),
            )
        end
        if get(ordering, "skipped", false)
            reordering_skips[ord_name] = Dict{String,Any}(
                key => value for (key, value) in pairs(ordering) if key != "skipped")
            continue
        end
        ordering_stats[ord_name] = Dict{String,Any}(
            key => value for (key, value) in pairs(ordering)
            if key != "permutation_file")
        reorderings[ord_name] = perm_filename
    end

    println("    [c$ci] Reorderings done in $(round(time()-t0; digits=1))s"); flush(stdout)

    scaled_matrix = Ref{Any}(nothing)
    function _get_scaled_matrix()
        if scaled_matrix[] === nothing
            scaled_matrix[] = _copy_csc_values(Asub)
            scale_symmetric!(scaled_matrix[])
        end
        return scaled_matrix[]
    end

    scaled_analysis = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "scaled_analysis";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        A_test = _get_scaled_matrix()
        scaled_dd_ratio_raw = diagonal_dominance_ratio(A_test)
        Dict{String,Any}(
            "unscaled_is_sdd" => is_sdd(Asub),
            "scaled_is_sdd" => is_sdd(A_test),
            "scaled_diagonal_dominance_ratio" =>
                isfinite(scaled_dd_ratio_raw) ? scaled_dd_ratio_raw : -1.0,
        )
    end
    unscaled_sdd = Bool(scaled_analysis["unscaled_is_sdd"])
    scaled_sdd = Bool(scaled_analysis["scaled_is_sdd"])
    scaled_dd_ratio = Float64(scaled_analysis["scaled_diagonal_dominance_ratio"])

    # Eigenvalue bounds of the scaled system
    scaled_eigenbounds = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "scaled_eigenbounds";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        println("    [c$ci] Estimating upper eigenvalue (scaled, upper_tol=$EIGBOUNDS_UPPER_TOL_DEFAULT, max_restarts=$eigbound_max_restarts)..."); flush(stdout)
        t0 = time()
        fields = _upper_eigenvalue_metadata(_get_scaled_matrix(); prefix = "scaled_", workspace=analysis_workspace, progress_callback = child_progress)
        println("    [c$ci] Scaled upper estimate done in $(round(time()-t0; digits=1))s: λ_max=$(fields["scaled_lambda_max"])"); flush(stdout)
        fields
    end
    scaled_lambda_max = Float64(scaled_eigenbounds["scaled_lambda_max"])

    # Gauss-Lanczos quadrature: b'A⁻¹b / ‖b‖² scaled by λ_max (dimensionless)
    quadrature_phase = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "quadrature";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        println("    [c$ci] Computing Lanczos quadrature (3 RHS; unscaled cap=$QUADRATURE_UNSCALED_MAX_ITERS, scaled cap=$QUADRATURE_SCALED_MAX_ITERS)..."); flush(stdout)
        t0 = time()
        num_rhs = 3
        _, B = generate_component_rhs(Asub, ci, num_rhs)

        # Compute D^{-1/2} for scaling RHS to match the scaled system.
        d_diag = Vector{Float64}(undef, dim)
        for col in 1:dim
            for j in Asub.colptr[col]:(Asub.colptr[col+1]-1)
                if Asub.rowval[j] == col
                    d_diag[col] = Asub.nzval[j]
                    break
                end
            end
        end
        inv_sqrt_d = 1.0 ./ sqrt.(d_diag)

        quadrature_results = Vector{Dict{String,Any}}(undef, num_rhs)
        for j in 1:num_rhs
            b_j = B[:, j]

            fields = _rhs_quadrature_metadata(Asub, b_j, lambda_max; workspace=analysis_workspace,
                progress_callback = phase_callback_with_context(child_progress; rhs_index=j))
            # Scaled system: b-hat = D^{-1/2} b
            b_scaled = inv_sqrt_d .* b_j
            merge!(fields, _rhs_quadrature_metadata(_get_scaled_matrix(), b_scaled,
                                                   scaled_lambda_max; scaled = true, workspace=analysis_workspace,
                progress_callback = phase_callback_with_context(child_progress; rhs_index=j)))
            quadrature_results[j] = fields
        end

        println("    [c$ci] Quadrature done in $(round(time()-t0; digits=1))s"); flush(stdout)
        Dict{String,Any}("quadrature" => quadrature_results)
    end
    quadrature_results = quadrature_phase["quadrature"]

    comp_meta = _run_checkpointed_prepare_phase!(
        checkpoint, checkpoint_path, "component_metadata";
        callback = checkpoint_callback,
        progress_callback = progress_callback,
    ) do child_progress
        Dict{String,Any}(
            "index" => ci,
            "files" => component_files,
            "dimension" => dim,
            "nnz" => nnz_val,
            "classification" => string(classification),
            unscaled_eigenbounds...,
            scaled_eigenbounds...,
            "eigenbounds_policy" => _upper_eigenvalue_policy(dim),
            "quadrature_policy" => _quadrature_policy(),
            "min_nnz_per_row" => Int(structural["min_nnz_per_row"]),
            "max_nnz_per_row" => Int(structural["max_nnz_per_row"]),
            "avg_nnz_per_row" => Float64(structural["avg_nnz_per_row"]),
            "median_nnz_per_row" => Float64(structural["median_nnz_per_row"]),
            "std_nnz_per_row" => Float64(structural["std_nnz_per_row"]),
            "total_triangles" => total_triangles,
            "max_vertex_triangles" => max_vertex_triangles,
            "triangle_hist_file" => triangle_hist_filename,
            "diameter_lower_bound" => diameter_lower_bound,
            "is_m_matrix" => Bool(structural["is_m_matrix"]),
            "diagonal_dominance_ratio" => Float64(structural["diagonal_dominance_ratio"]),
            "scaled_diagonal_dominance_ratio" => scaled_dd_ratio,
            "ordering_stats" => ordering_stats,
            "reorderings" => reorderings,
            "reordering_skips" => reordering_skips,
            "unscaled_is_sdd" => unscaled_sdd,
            "scaled_is_sdd" => scaled_sdd,
            "quadrature" => quadrature_results,
        )
    end

    sdd_str = " [unscaled_sdd=$unscaled_sdd, scaled_sdd=$scaled_sdd]"
    println("  Component $ci: dim=$dim, nnz=$nnz_val, class=$classification, λ_max=$lambda_max, λ_min not estimated$sdd_str"); flush(stdout)

    return comp_meta
end

"""
    decompose_and_cache(A, name, base_dir; delete_original=false, matrix_type=:auto,
                        amd_max_nnz=AMD_MAX_NNZ_DEFAULT,
                        ordering_timeout_seconds=86400,
                        prepare_checkpoint_callback=_ -> nothing,
                        prepare_progress_callback=_ -> nothing)

Decompose a sparse matrix into connected components, classify each, and
write ready-to-solve SPD systems to disk as compressed .mtx.zstd files
with a metadata.json.

For graph Laplacians and adjacency matrices, the prolific node is pinned
at ingest time and the resulting (possibly disconnected) SPD sub-components
are stored as separate components. This means all on-disk components are
SPD and require no further transformation at load time.

`matrix_type` controls how each component is interpreted:
  - `:auto` (default) — classify each component automatically
  - `:adjacency` — treat all components as adjacency matrices (convert to
    Laplacian, pin, re-decompose). Useful for graphs with self-loops or
    other structure that would confuse automatic classification.
  - `:laplacian` — treat all components as graph Laplacians (pin,
    re-decompose, skip the adjacency-to-Laplacian conversion)

AMD is skipped for components with more than `amd_max_nnz` stored nonzeros.
Each RCM/AMD/METIS worker has a separate `ordering_timeout_seconds` deadline
(default 24 hours), including startup, structure loading, and permutation
output. Statistics are computed afterward as part of the checkpointed stage.
Timeouts persist in `reordering_skips`; remove that ordering's phase checkpoint
explicitly to retry during a subsequent decompose_and_cache call.
Each skip and its applicable limit are recorded in `metadata.json`; the other
orderings are still attempted independently.

Each expensive component-analysis phase is committed atomically under
`components/prepare_state/`. If preparation is interrupted, rerunning the same
input validates its SHA-256 fingerprint and resumes from the last completed
phase. `prepare_checkpoint_callback` is invoked after each durable phase commit.
`prepare_progress_callback` receives start, completion, resume, invalidation,
and error events for live phase monitoring.

If `delete_original` is true, the original .mtx file is removed after
decomposition completes successfully. This is useful when the matrix is
too large to carry both the original and the cached components.

On-disk layout:
  {base_dir}/{name}/components/{name}_c{i}_{dim}.mtx.zstd
  {base_dir}/{name}/components/metadata.json
  {base_dir}/{name}/components/reorderings/{ordering}_c{i}.vec.zstd
"""
function decompose_and_cache(A::SparseMatrixCSC, name::String, base_dir::String;
                             delete_original::Bool=false, matrix_type::Symbol=:auto,
                             amd_max_nnz::Int=AMD_MAX_NNZ_DEFAULT,
                             ordering_timeout_seconds::Real=ORDERING_TIMEOUT_SECONDS_DEFAULT,
                             prepare_checkpoint_callback=_noop_prepare_checkpoint_callback,
                             prepare_progress_callback=NOOP_PHASE_CALLBACK)
    @assert matrix_type ∈ (:auto, :adjacency, :laplacian) "matrix_type must be :auto, :adjacency, or :laplacian"
    amd_max_nnz >= 0 || throw(ArgumentError("amd_max_nnz must be non-negative"))
    isfinite(ordering_timeout_seconds) && ordering_timeout_seconds > 0 ||
        throw(ArgumentError("ordering_timeout_seconds must be finite and positive"))
    comp_dir = joinpath(base_dir, name, "components")
    reorder_dir = joinpath(comp_dir, "reorderings")
    mkpath(comp_dir)
    mkpath(reorder_dir)

    # Symmetrize the full matrix before decomposition
    println("  Symmetrizing (n=$(size(A,1)), nnz=$(nnz(A)))..."); flush(stdout)
    run_phase(prepare_progress_callback, "symmetrize") do
        symmetrize!(A)
    end

    println("  Finding connected components..."); flush(stdout)
    comps = run_phase(prepare_progress_callback, "connected_components") do
        Laplacians.vecToComps(Laplacians.components(A))
    end

    # Sort components by descending size
    sort!(comps, by=length, rev=true)
    println("  Found $(length(comps)) components (largest=$(length(comps[1])))"); flush(stdout)

    metadata = Dict{String,Any}(
        "source" => name,
        "matrix_type" => string(matrix_type),
        "amd_max_nnz" => amd_max_nnz,
        "ordering_timeout_seconds" => Float64(ordering_timeout_seconds),
        "prepare_checkpoint_schema_version" => PREPARE_CHECKPOINT_SCHEMA_VERSION,
        "components" => []
    )

    global_ci = 0

    for (raw_ci, indices) in enumerate(comps)
        # Components are sorted descending by size, so once we hit a trivial
        # component all remaining ones are also trivial.  Batch them into a
        # single metadata entry to avoid millions of allocations / log lines.
        if length(indices) <= 1
            trivial_raw_count = length(comps) - raw_ci + 1
            global_ci += trivial_raw_count
            println("  Batched $trivial_raw_count trivial raw components (dim <= 1)"); flush(stdout)
            push!(metadata["components"], Dict{String,Any}(
                "batch" => true,
                "count" => trivial_raw_count,
                "dimension" => 0,
                "skipped" => true,
                "skip_reason" => "trivial (dim <= 1), batched",
            ))
            break
        end

        Asub = run_phase(prepare_progress_callback, "component_extract"; component=raw_ci, raw_component_total=length(comps)) do
            A[indices, indices]
        end

        if matrix_type == :auto
            classification = classify_component(Asub)
        elseif matrix_type == :adjacency
            classification = :adjacency
        elseif matrix_type == :laplacian
            classification = :graph_laplacian
        end

        # Adjacency matrix: convert to Laplacian first
        if classification == :adjacency
            println("  Raw component $raw_ci: adjacency matrix (dim=$(size(Asub,1))), converting to Laplacian"); flush(stdout)
            Asub = run_phase(prepare_progress_callback, "adjacency_to_laplacian"; component=raw_ci) do
                adjacency_to_laplacian(Asub)
            end
            classification = :graph_laplacian
        end

        if classification == :graph_laplacian
            # Pin prolific node to produce SPD system(s)
            prolific_node, prolific_criterion = run_phase(prepare_progress_callback, "pin_node_selection"; component=raw_ci) do
                select_prolific_node(Asub)
            end
            dim = size(Asub, 1)
            keep = [i for i in 1:dim if i != prolific_node]
            A_pinned = run_phase(prepare_progress_callback, "graph_pin"; component=raw_ci) do
                Asub[keep, keep]
            end

            println("  Raw component $raw_ci: graph_laplacian (dim=$dim), pinned node $prolific_node via $prolific_criterion"); flush(stdout)

            # Re-decompose: pinning may disconnect the graph
            co_sub = run_phase(prepare_progress_callback, "pinned_connected_components"; component=raw_ci) do
                Laplacians.components(A_pinned)
            end
            if isempty(co_sub)
                global_ci += 1
                println("    Pinning produced empty graph, recording metadata only"); flush(stdout)
                push!(metadata["components"], Dict{String,Any}(
                    "index" => global_ci,
                    "dimension" => 0,
                    "nnz" => 0,
                    "origin_component" => raw_ci,
                    "origin_classification" => "graph_laplacian",
                    "pinned_node" => prolific_node,
                    "pinned_criterion" => prolific_criterion,
                    "skipped" => true,
                    "skip_reason" => "empty after pinning",
                ))
                continue
            end
            sub_comps = Laplacians.vecToComps(co_sub)
            sort!(sub_comps, by=length, rev=true)

            if length(sub_comps) > 1
                println("    Pinning produced $(length(sub_comps)) sub-components"); flush(stdout)
            end

            # Sub-components are sorted descending by size, so once we hit
            # a trivial one all remaining are also trivial.
            for (sub_ci, sub_indices) in enumerate(sub_comps)
                if length(sub_indices) <= 1
                    trivial_sub_count = length(sub_comps) - sub_ci + 1
                    global_ci += trivial_sub_count
                    println("    Batched $trivial_sub_count trivial sub-components (dim <= 1)"); flush(stdout)
                    push!(metadata["components"], Dict{String,Any}(
                        "batch" => true,
                        "count" => trivial_sub_count,
                        "dimension" => 0,
                        "origin_component" => raw_ci,
                        "origin_classification" => "graph_laplacian",
                        "pinned_node" => prolific_node,
                        "pinned_criterion" => prolific_criterion,
                        "sub_component_count" => length(sub_comps),
                        "skipped" => true,
                        "skip_reason" => "trivial (dim <= 1) sub-components, batched",
                    ))
                    break
                end
                global_ci += 1
                A_final = A_pinned[sub_indices, sub_indices]
                comp_meta = _compute_component_metadata(
                    A_final, name, global_ci, comp_dir, reorder_dir;
                    amd_max_nnz = amd_max_nnz,
                    ordering_timeout_seconds = ordering_timeout_seconds,
                    checkpoint_callback = prepare_checkpoint_callback,
                    progress_callback = prepare_progress_callback,
                )

                # Provenance fields
                comp_meta["origin_component"] = raw_ci
                comp_meta["origin_classification"] = "graph_laplacian"
                comp_meta["pinned_node"] = prolific_node
                comp_meta["pinned_criterion"] = prolific_criterion
                comp_meta["sub_component_index"] = sub_ci
                comp_meta["sub_component_count"] = length(sub_comps)

                push!(metadata["components"], comp_meta)
            end
        else
            # SDDM, general SDD, general SPD — store directly
            global_ci += 1
            comp_meta = _compute_component_metadata(
                Asub, name, global_ci, comp_dir, reorder_dir;
                amd_max_nnz = amd_max_nnz,
                ordering_timeout_seconds = ordering_timeout_seconds,
                checkpoint_callback = prepare_checkpoint_callback,
                progress_callback = prepare_progress_callback,
            )
            push!(metadata["components"], comp_meta)
        end
    end

    metadata["num_components"] = global_ci

    # Only publish the canonical manifest after every component is complete.
    run_phase(prepare_progress_callback, "cache_manifest_write") do
        _atomic_write_json(joinpath(comp_dir, "metadata.json"), metadata)
    end

    if delete_original
        orig_mtx = joinpath(base_dir, name, name * ".mtx")
        if isfile(orig_mtx)
            rm(orig_mtx)
            println("Deleted original: $orig_mtx"); flush(stdout)
        end
    end

    return metadata
end

"""
    load_components(name, base_dir) -> (Vector{SparseMatrixCSC}, metadata)

Load cached components from .npy.zstd files and metadata from metadata.json.
"""
function load_components(name::String, base_dir::String)
    comp_dir = joinpath(base_dir, name, "components")
    meta_path = joinpath(comp_dir, "metadata.json")

    metadata = open(meta_path, "r") do io
        JSON3.read(io, Dict{String,Any})
    end

    matrices = SparseMatrixCSC[]
    for comp_meta in metadata["components"]
        if get(comp_meta, "skipped", false) || Int(comp_meta["dimension"]) <= 1
            continue
        end
        if !haskey(comp_meta, "files")
            ci_label = get(comp_meta, "index", "?")
            error("load_components: component $ci_label lacks 'files' field. " *
                  "Unsupported legacy cache. Preserve the existing data and prepare a fresh cache from the original input in a separate directory.")
        end
        files = comp_meta["files"]
        colptr = read_npy_zstd_1d_i64(joinpath(comp_dir, String(files["colptr"])))
        rowval = read_npy_zstd_1d_i64(joinpath(comp_dir, String(files["rowval"])))
        nzval  = read_npy_zstd_1d_f64(joinpath(comp_dir, String(files["nzval"])))
        dim = Int(comp_meta["dimension"])
        push!(matrices, SparseMatrixCSC(dim, dim, colptr, rowval, nzval))
    end

    return matrices, metadata
end

"""
    load_reordering(name, base_dir, component_index, ordering) -> Vector{Int64}

Load a cached reordering permutation for a specific component. Reads the
1-D Int64 NPY file declared in the component's `reorderings` dict.
"""
function load_reordering(name::String, base_dir::String, component_index::Int, ordering::String)
    comp_dir = joinpath(base_dir, name, "components")
    meta_path = joinpath(comp_dir, "metadata.json")

    metadata = open(meta_path, "r") do io
        JSON3.read(io, Dict{String,Any})
    end

    comp_meta = nothing
    for cm in metadata["components"]
        if get(cm, "index", -1) == component_index
            comp_meta = cm
            break
        end
    end
    comp_meta === nothing && error("Component $component_index not found in metadata for $name")
    reorderings = comp_meta["reorderings"]
    if !haskey(reorderings, ordering)
        skips = get(comp_meta, "reordering_skips", Dict{String,Any}())
        if haskey(skips, ordering)
            skip = skips[ordering]
            error("load_reordering: '$ordering' was skipped for component $component_index of $name " *
                  "($(repr(skip)))")
        end
        error("load_reordering: '$ordering' is unavailable for component $component_index of $name")
    end
    perm_filename = String(reorderings[ordering])
    perm_path = joinpath(comp_dir, "reorderings", perm_filename)
    if !endswith(perm_filename, ".npy.zstd")
        error("load_reordering: $perm_path uses an unsupported extension. " *
              "Unsupported legacy cache. Preserve the existing data and prepare a fresh cache from the original input in a separate directory.")
    end
    return read_npy_zstd_1d_i64(perm_path)
end

"""
    ComponentDescriptor

Lightweight descriptor for a matrix component. Holds metadata and paths
for lazy loading — no matrix data in memory. Created by `describe_components`.

The matrix is loaded from disk on demand via `load_component_matrix`.
Lower eigenvalue estimates and their diagnostics are `nothing` for upper-only
preparation; legacy caches can still contain numerical lower Ritz estimates.
"""
@kwdef struct ComponentDescriptor
    name::String
    component_index::Int
    base_dir::String
    dimension::Int
    nnz::Int
    classification::Symbol
    lambda_max::Float64
    lambda_min::Union{Nothing,Float64}
    ritz_min_residual::Union{Nothing,Float64}
    ritz_max_residual::Float64
    ritz_min_half_gap::Union{Nothing,Float64}
    ritz_max_half_gap::Float64
    scaled_lambda_max::Float64
    scaled_lambda_min::Union{Nothing,Float64}
    scaled_ritz_min_residual::Union{Nothing,Float64}
    scaled_ritz_max_residual::Float64
    scaled_ritz_min_half_gap::Union{Nothing,Float64}
    scaled_ritz_max_half_gap::Float64
    unscaled_is_sdd::Bool
    scaled_is_sdd::Bool
    reordering::String = "natural"
    # Provenance: tracks how this component was derived from the original matrix
    prolific_node::Union{Int, Nothing} = nothing
    prolific_criterion::Union{String, Nothing} = nothing
    origin_component::Union{Int, Nothing} = nothing
    origin_classification::Union{Symbol, Nothing} = nothing
    sub_component_index::Union{Int, Nothing} = nothing
    sub_component_count::Union{Int, Nothing} = nothing
    metadata::Dict{String, Any} = Dict{String, Any}()
end

"""
    describe_components(name; base_dir="./matrices/", reordering="natural",
                        delete_original=false, matrix_type=:auto,
                        amd_max_nnz=AMD_MAX_NNZ_DEFAULT,
                        prepare_progress_callback=_ -> nothing) -> Vector{ComponentDescriptor}

Return lightweight descriptors for each connected component of a matrix.
No matrix data is held in memory.

`matrix_type` controls how each component is interpreted during decomposition:
  - `:auto` (default) — classify each component automatically
  - `:adjacency` — treat all components as adjacency matrices (useful for
    graphs with self-loops or other structure that confuses auto-detection)
  - `:laplacian` — treat all components as graph Laplacians

Lookup order:
1. If cached components exist (`{base_dir}/{name}/components/metadata.json`),
   descriptors are built from the cache immediately.
2. Otherwise, if the original `.mtx` file exists, it is decomposed via
   `decompose_and_cache` and the resulting metadata is used. Set
   `delete_original=true` to remove the `.mtx` after successful decomposition
   (useful when the matrix is too large to keep both copies).
3. If neither cached components nor a `.mtx` file are found, an error is raised.

Components with an explicitly skipped requested ordering are logged and omitted
before loading numeric data. Missing orderings without a skip record are errors.
"""
function describe_components(name::String; base_dir::String="./matrices/",
                             reordering::String="natural",
                             delete_original::Bool=false,
                             matrix_type::Symbol=:auto,
                             amd_max_nnz::Int=AMD_MAX_NNZ_DEFAULT,
                             ordering_timeout_seconds::Real=ORDERING_TIMEOUT_SECONDS_DEFAULT,
                             prepare_progress_callback=NOOP_PHASE_CALLBACK)
    reordering = lowercase(reordering)
    @assert reordering ∈ ["natural", "rcm", "amd", "metis"]

    name_stripped = contains(name, "/") ? String(split(name, "/")[end]) : name
    matrix_dir = joinpath(base_dir, name)
    comp_dir = joinpath(matrix_dir, "components")
    meta_path = joinpath(comp_dir, "metadata.json")
    orig_mtx = joinpath(matrix_dir, name_stripped * ".mtx")

    # Check for cached components first
    cache_hit = isfile(meta_path)
    if !cache_hit
        # No cache — try to build from the original .mtx
        if isfile(orig_mtx)
            println("Decomposing $name into components..."); flush(stdout)
            println("  Reading $orig_mtx..."); flush(stdout)
            t_read = time()
            # Use the structural-zero-preserving custom reader instead of
            # MatrixMarket.mmread, which silently drops explicit zeros.
            A_orig = run_phase(prepare_progress_callback, "input_read"; path=orig_mtx) do
                read_matrixmarket_coord(orig_mtx)
            end
            println("  Read complete in $(round(time()-t_read; digits=1))s: $(size(A_orig,1))×$(size(A_orig,2)), nnz=$(nnz(A_orig))"); flush(stdout)
            decompose_and_cache(
                A_orig, name, base_dir;
                delete_original = delete_original,
                matrix_type = matrix_type,
                amd_max_nnz = amd_max_nnz,
                ordering_timeout_seconds = ordering_timeout_seconds,
                prepare_progress_callback = prepare_progress_callback,
            )
        else
            error("No cached components and no .mtx file found for '$name' in $base_dir")
        end
    end

    metadata = run_phase(prepare_progress_callback, "cache_read"; cached=cache_hit, path=meta_path) do
        open(meta_path, "r") do io
            JSON3.read(io, Dict{String,Any})
        end
    end

    descriptors = ComponentDescriptor[]
    for comp_meta in metadata["components"]
        if get(comp_meta, "skipped", false) || Int(comp_meta["dimension"]) <= 1
            continue
        end
        ci = Int(comp_meta["index"])

        if reordering != "natural" && !haskey(comp_meta["reorderings"], reordering)
            skips = get(comp_meta, "reordering_skips", Dict{String,Any}())
            if haskey(skips, reordering)
                @info "Skipping component with unavailable ordering" name component=ci reordering details=skips[reordering]
                continue
            end
            error("describe_components: '$reordering' is unavailable for component $ci of $name")
        end

        # Provenance fields (present for components derived from graph Laplacians)
        prolific = haskey(comp_meta, "pinned_node") && comp_meta["pinned_node"] !== nothing ?
            Int(comp_meta["pinned_node"]) : nothing
        prolific_crit = get(comp_meta, "pinned_criterion", nothing)
        origin_comp = haskey(comp_meta, "origin_component") ? Int(comp_meta["origin_component"]) : nothing
        origin_class = haskey(comp_meta, "origin_classification") ? Symbol(comp_meta["origin_classification"]) : nothing
        sub_ci = haskey(comp_meta, "sub_component_index") ? Int(comp_meta["sub_component_index"]) : nothing
        sub_count = haskey(comp_meta, "sub_component_count") ? Int(comp_meta["sub_component_count"]) : nothing

        push!(descriptors, ComponentDescriptor(
            name = name_stripped,
            component_index = ci,
            base_dir = base_dir,
            dimension = Int(comp_meta["dimension"]),
            nnz = Int(comp_meta["nnz"]),
            classification = Symbol(comp_meta["classification"]),
            lambda_max = Float64(comp_meta["lambda_max"]),
            lambda_min = comp_meta["lambda_min"],
            ritz_min_residual = comp_meta["ritz_min_residual"],
            ritz_max_residual = Float64(comp_meta["ritz_max_residual"]),
            ritz_min_half_gap = comp_meta["ritz_min_half_gap"],
            ritz_max_half_gap = Float64(comp_meta["ritz_max_half_gap"]),
            scaled_lambda_max = Float64(comp_meta["scaled_lambda_max"]),
            scaled_lambda_min = comp_meta["scaled_lambda_min"],
            scaled_ritz_min_residual = comp_meta["scaled_ritz_min_residual"],
            scaled_ritz_max_residual = Float64(comp_meta["scaled_ritz_max_residual"]),
            scaled_ritz_min_half_gap = comp_meta["scaled_ritz_min_half_gap"],
            scaled_ritz_max_half_gap = Float64(comp_meta["scaled_ritz_max_half_gap"]),
            unscaled_is_sdd = Bool(comp_meta["unscaled_is_sdd"]),
            scaled_is_sdd = Bool(comp_meta["scaled_is_sdd"]),
            reordering = reordering,
            prolific_node = prolific,
            prolific_criterion = prolific_crit,
            origin_component = origin_comp,
            origin_classification = origin_class,
            sub_component_index = sub_ci,
            sub_component_count = sub_count,
            metadata = comp_meta,
        ))
    end

    return descriptors
end

"""
    load_component_matrix(desc::ComponentDescriptor) -> SparseMatrixCSC

Load a single component matrix by reading its three `.npy.zstd` files
(colptr, rowval, nzval) and assembling a `SparseMatrixCSC`. Preserves
explicit zeros bit-for-bit.
"""
function load_component_matrix(desc::ComponentDescriptor)
    comp_dir = joinpath(desc.base_dir, desc.name, "components")
    haskey(desc.metadata, "files") ||
        error("load_component_matrix: component $(desc.component_index) for $(desc.name) " *
              "lacks 'files' field in metadata. Unsupported legacy cache. " *
              "Preserve the existing data and prepare a fresh cache from the original input in a separate directory.")
    files = desc.metadata["files"]
    colptr = read_npy_zstd_1d_i64(joinpath(comp_dir, String(files["colptr"])))
    rowval = read_npy_zstd_1d_i64(joinpath(comp_dir, String(files["rowval"])))
    nzval  = read_npy_zstd_1d_f64(joinpath(comp_dir, String(files["nzval"])))
    return SparseMatrixCSC(desc.dimension, desc.dimension, colptr, rowval, nzval)
end

"""
    symmetrize!(A::SparseMatrixCSC)

In-place enforcement of bitwise symmetry for a structurally symmetric
sparse matrix. For every pair of stored entries `A[i,j]` and `A[j,i]`
with `i < j`, overwrite both slots with `0.5 * (A[i,j] + A[j,i])` so
that the two positions hold identical bits.

Precondition: `A` is structurally symmetric (every `(i,j)` stored entry
has a corresponding `(j,i)` stored entry). Throws if violated.

Motivation: SuiteSparse matrices are mathematically symmetric but some
are truncated such that `A[i,j] != A[j,i]` at the bit level, which some
downstream tooling (ordering libraries, asserts) rejects. This is the
fast in-place equivalent of `A = (A + A') .* 0.5` without the O(nnz)
allocations.
"""
function symmetrize!(A::SparseMatrixCSC)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    rowval = A.rowval
    nzval = A.nzval
    colptr = A.colptr
    @batch per=core for col in 1:n
        for j in colptr[col]:(colptr[col+1]-1)
            row = rowval[j]
            row >= col && continue
            t_range = colptr[row]:(colptr[row+1]-1)
            t_local = searchsortedfirst(view(rowval, t_range), col)
            (t_local > length(t_range) || rowval[t_range[t_local]] != col) &&
                error("symmetrize!: structural asymmetry at ($row, $col)")
            t_idx = t_range[t_local]
            avg = 0.5 * (nzval[j] + nzval[t_idx])
            nzval[j] = avg
            nzval[t_idx] = avg
        end
    end
    return A
end

# Internal analysis-only copy. Callers must not alter the shared structure.
_copy_csc_values(A::SparseMatrixCSC{T,I}) where {T,I} =
    SparseMatrixCSC{T,I}(size(A,1), size(A,2), A.colptr, A.rowval, copy(A.nzval))

"""
    scale_symmetric!(A::SparseMatrixCSC)

In-place symmetric diagonal scaling: A_ij = A_ij / sqrt(d_i * d_j), with
diagonal forced to 1.0. Single pass over nzval, preserves bit-symmetry.

Returns the diagonal vector d (original diagonal values) for use in
scaling/unscaling RHS and solution vectors.
"""
function scale_symmetric!(A::SparseMatrixCSC)
    d, _ = _scale_symmetric_with_factors!(A)
    return d
end

function _diagonal_positions(A::SparseMatrixCSC)
    positions = Vector{Int64}(undef, size(A, 1))
    for col in eachindex(positions)
        range = nzrange(A, col)
        offset = searchsortedfirst(view(A.rowval, range), col)
        offset <= length(range) && A.rowval[first(range) + offset - 1] == col ||
            throw(ArgumentError("symmetric scaling requires a stored diagonal at column $col"))
        positions[col] = first(range) + offset - 1
    end
    return positions
end

# The caller owns cached positions and keeps the CSC structure immutable.
# Factors are step-local: retaining them across solves would increase live RAM.
function _scale_symmetric_with_factors!(A::SparseMatrixCSC, positions=nothing)
    n = size(A, 1)
    d = Vector{Float64}(undef, n)
    if positions === nothing
        for col in 1:n
            found = false
            for p in nzrange(A, col)
                if A.rowval[p] == col
                    d[col] = A.nzval[p]
                    found = true
                    break
                end
            end
            found || throw(ArgumentError("symmetric scaling requires a stored diagonal at column $col"))
        end
    else
        length(positions) == n || throw(DimensionMismatch("scaling diagonal positions"))
        for col in 1:n
            d[col] = A.nzval[positions[col]]
        end
    end
    inv_sqrt_d = 1.0 ./ sqrt.(abs.(d))
    for col in 1:n
        s_col = inv_sqrt_d[col]
        for j in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[j]
            if row == col
                A.nzval[j] = 1.0
            else
                A.nzval[j] *= s_col * inv_sqrt_d[row]
            end
        end
    end
    return d, inv_sqrt_d
end

"""
    prepare_system(desc::ComponentDescriptor; wants_sdd=false) -> (A, seed, b)

Load a component from disk, apply reordering, scale, and prepare for solving.

All components are stored as ready-to-solve SPD systems (graph Laplacians
and adjacency matrices are pinned at ingest time by `decompose_and_cache`).

When `wants_sdd=false` (default), the matrix is always scaled.
When `wants_sdd=true`, the SDD-preserving path is used:
  - If scaled is SDD: returns the scaled matrix
  - If unscaled is SDD and scaled is not: returns the unscaled matrix
  - If neither is SDD: returns the scaled matrix (preconditioner pads internally)
"""
function prepare_system(desc::ComponentDescriptor; wants_sdd::Bool=false,
                        spectral_info=nothing, progress_callback=NOOP_PHASE_CALLBACK)
    A = load_component_matrix(desc)
    if spectral_info !== nothing
        spectral_info[] = component_spectral_norm(desc, A; wants_sdd, progress_callback)
    end

    # Apply reordering if needed
    if desc.reordering != "natural"
        perm = load_reordering(desc.name, desc.base_dir, desc.component_index, desc.reordering)
        A = A[perm, perm]
    end

    # Generate RHS
    seed, b = generate_component_rhs(A, desc.component_index)

    # Decide whether to scale
    if wants_sdd && desc.unscaled_is_sdd && !desc.scaled_is_sdd
        # Unscaled is SDD, scaled is not: use unscaled
        return A, seed, b
    else
        # Default: always scale
        d = scale_symmetric!(A)
        b .= b .* (1.0 ./ sqrt.(abs.(d)))
        seed .= seed .* sqrt.(abs.(d))
        return A, seed, b
    end
end

# ─── Structured Logging ──────────────────────────────────────────────

const _VALID_LOG_LEVELS = (:info, :warn, :error)

struct LogEntry
    timestamp::String
    level::Symbol
    event::String
    message::String
    data::Dict{String,Any}
end

mutable struct Logger
    entries::Vector{LogEntry}
    io::Union{IO,Nothing}
end

"""
    open_logger(path::String) -> Logger

Create a logger that writes JSON Lines to `path` (append mode) and keeps entries in memory.
"""
function open_logger(path::String)
    io = open(path, "a")
    return Logger(LogEntry[], io)
end

"""
    open_logger() -> Logger

Create a memory-only logger (no file output).
"""
function open_logger()
    return Logger(LogEntry[], nothing)
end

"""
    log!(logger, level, event, message; kwargs...)

Append a structured log entry. `level` must be one of `:info`, `:warn`, `:error`.
Keyword arguments are stored as structured data and merged into the top-level JSON object.
"""
function log!(logger::Logger, level::Symbol, event::String, message::String; kwargs...)
    level in _VALID_LOG_LEVELS || throw(ArgumentError("Invalid log level: $level. Must be one of $_VALID_LOG_LEVELS"))

    ts = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS.sss")
    data = Dict{String,Any}(string(k) => v for (k, v) in kwargs)
    entry = LogEntry(ts, level, event, message, data)
    push!(logger.entries, entry)

    if logger.io !== nothing
        obj = Dict{String,Any}(
            "timestamp" => ts,
            "level" => string(level),
            "event" => event,
            "message" => message,
        )
        merge!(obj, data)
        JSON3.write(logger.io, obj)
        write(logger.io, "\n")
        flush(logger.io)
    end

    return nothing
end

"""
    close_logger!(logger) -> Vector{LogEntry}

Close the file handle (if any) and return all collected log entries.
"""
function close_logger!(logger::Logger)
    if logger.io !== nothing
        close(logger.io)
        logger.io = nothing
    end
    return logger.entries
end

# ─── Resource Monitoring ─────────────────────────────────────────────

const _CLK_TCK = Int(ccall(:sysconf, Clong, (Cint,), 2))  # _SC_CLK_TCK = 2
const _PAGE_SIZE = Int(ccall(:sysconf, Clong, (Cint,), 30))  # _SC_PAGESIZE = 30

struct ResourceSample
    elapsed_seconds::Float64
    rss_bytes::Int64
    vm_bytes::Int64
    cpu_user_seconds::Float64
    cpu_sys_seconds::Float64
    gc_live_bytes::Int64
    gc_total_allocated::Int64
end

"""
    _read_proc_stat() -> (rss_bytes, vm_bytes, cpu_user_s, cpu_sys_s)

Read current process stats from /proc/self/stat.
"""
function _read_proc_stat()
    stat = read("/proc/self/stat", String)
    # Field 2 is (comm) which may contain spaces — skip past the closing paren
    close_paren = findlast(')', stat)
    fields = split(@view(stat[close_paren+2:end]))
    # After (comm), fields are 1-indexed: state=1, ppid=2, ..., utime=12, stime=13, ..., vsize=21, rss=22
    utime = parse(Int64, fields[12])
    stime = parse(Int64, fields[13])
    vsize = parse(Int64, fields[21])
    rss_pages = parse(Int64, fields[22])
    return (
        rss_pages * _PAGE_SIZE,
        vsize,
        utime / _CLK_TCK,
        stime / _CLK_TCK,
    )
end

mutable struct ResourceMonitor
    samples::Vector{ResourceSample}
    timer::Union{Timer, Nothing}
    t0::Float64
    lock::ReentrantLock
    paths::Vector{String}
    flush_samples::Int
    stopped::Bool
end

struct ResourceTrace
    path::String
end

include("execution_provenance.jl")

function _flush_resources!(mon::ResourceMonitor)
    isempty(mon.paths) && return
    isempty(mon.samples) && return
    for path in mon.paths
        open(path,"a") do io
            _write_resource_rows(io,mon.samples)
        end
    end
    empty!(mon.samples)
end

"""
    start_monitor(; interval=1.0, paths=String[], flush_samples=60, context=Dict())

Start periodic resource sampling. With output paths, publish samples in bounded
batches and return a disk-backed ResourceTrace on stop. Without paths, retain
the legacy in-memory API. Production runners supply paths. Call stop_monitor!
to flush the final samples and stop the timer.
"""
function start_monitor(; interval::Float64=1.0, paths::Vector{String}=String[],
                         flush_samples::Int=60, context=Dict{String,Any}())
    interval > 0 && isfinite(interval) || throw(ArgumentError("invalid sample interval"))
    flush_samples > 0 || throw(ArgumentError("invalid resource buffer size"))
    samples = ResourceSample[]
    t0 = time()
    lk = ReentrantLock()

    # Take initial sample
    sample = _take_sample(t0)
    push!(samples, sample)

    destinations=unique(abspath.(paths))
    for dir in unique(dirname.(destinations))
        _write_execution_provenance!(dir,context)
    end
    for path in destinations
        open(path,"w") do io
            write(io,_RESOURCE_HEADER)
        end
    end
    mon=ResourceMonitor(samples,nothing,t0,lk,destinations,flush_samples,false)
    _flush_resources!(mon) # Publish the initial sample immediately.
    mon.timer = Timer(interval; interval=interval) do _
        lock(lk) do
            mon.stopped && return
            push!(samples,_take_sample(t0))
            length(samples) >= flush_samples && _flush_resources!(mon)
        end
    end
    return mon
end

"""
    stop_monitor!(mon::ResourceMonitor) -> Vector{ResourceSample}

Stop the resource monitor and return all collected samples (with a final sample appended).
"""
function stop_monitor!(mon::ResourceMonitor)
    if mon.timer !== nothing
        close(mon.timer)
        mon.timer = nothing
    end
    # Take one final sample
    lock(mon.lock) do
        if !mon.stopped
            push!(mon.samples, _take_sample(mon.t0))
            _flush_resources!(mon)
            mon.stopped = true
        end
    end
    return isempty(mon.paths) ? mon.samples : ResourceTrace(first(mon.paths))
end

function _take_sample(t0::Float64)
    rss, vm, utime, stime = _read_proc_stat()
    gc = Base.gc_num()
    return ResourceSample(
        time() - t0,
        rss,
        vm,
        utime,
        stime,
        Base.gc_live_bytes(),
        gc.allocd + gc.deferred_alloc,
    )
end

const _RESOURCE_HEADER = "elapsed_seconds,rss_bytes,vm_bytes,cpu_user_seconds,cpu_sys_seconds,gc_live_bytes,gc_total_allocated\n"

"""
    write_resources(path::String, samples::Vector{ResourceSample})

Write resource samples to a CSV file.
"""
function write_resources(path::String, samples::Vector{ResourceSample})
    open(path, "w") do f
        write(f, _RESOURCE_HEADER)
        _write_resource_rows(f,samples)
    end
end

function _write_resource_rows(f,samples)
    for s in samples
        write(f, string(
                s.elapsed_seconds, ",",
                s.rss_bytes, ",",
                s.vm_bytes, ",",
                s.cpu_user_seconds, ",",
                s.cpu_sys_seconds, ",",
                s.gc_live_bytes, ",",
                s.gc_total_allocated, "\n",
            ))
    end
end

function write_resources(path::String, trace::ResourceTrace)
    abspath(path) == trace.path && return nothing
    cp(trace.path,path; force=true)
    return nothing
end

function __init__()
    empty!(_provenance_repositories) # Never reuse checkout observations from a precompile process.
    ensure_dsterf_built()
    ensure_dstev_built()
    ensure_dstegr_built()
end

include("numerical_reporting.jl")
include("spectral_norm_cache.jl")

end
