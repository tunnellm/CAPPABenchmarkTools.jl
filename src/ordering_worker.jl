# Launched only by Utilities._bounded_ordering; no numeric matrix/RHS is loaded.
using CAPPABenchmarkTools
using SparseArrays, LinearAlgebra
const U = CAPPABenchmarkTools.Utilities

function ordering_worker(args)
    length(args) == 5 || error("Expected ordering, dimension, colptr, rowval, output")
    ordering, dim, colptr_path, rowval_path, output = args
    n = parse(Int, dim)
    BLAS.set_num_threads(1)
    colptr = U.read_npy_zstd_1d_i64(colptr_path)
    rowval = U.read_npy_zstd_1d_i64(rowval_path)
    length(colptr) == n + 1 || error("Invalid colptr length")
    first(colptr) == 1 && last(colptr) == length(rowval) + 1 || error("Invalid colptr endpoints")
    issorted(colptr) || error("Unsorted colptr")
    all(i -> 1 <= i <= n, rowval) || error("Invalid row index")
    A = SparseMatrixCSC(n, n, colptr, rowval, fill(true, length(rowval)))
    perm = U._validate_ordering_permutation(Int64.(U._ordering_permutation(A, ordering)), n)
    U._atomic_write_npy_zstd_1d_i64(output, perm)
end

ordering_worker(ARGS)
