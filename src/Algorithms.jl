module Algorithms

using SparseArrays

export sparse_triangular_solve, sparse_triangular_solve!


"""
    sparse_triangular_solve(L::SparseMatrixCSC, U::SparseMatrixCSC, b::Vector)

Solves the sparse triangular system `L * U * x = b` and returns the solution.

# Arguments
- `L::SparseMatrixCSC`: The lower triangular matrix in transposed form.
- `U::SparseMatrixCSC`: The upper triangular matrix in transposed form.
- `b::Vector`: The right-hand side vector.

# Returns
- `Vector{Float64}`: The solution vector `x`.
"""
function sparse_triangular_solve(L::SparseMatrixCSC, U::SparseMatrixCSC, b::Vector)
    x = Vector{Float64}(undef, size(b, 1))
    sparse_triangular_solve!(x, L, U, b)
    return x
end

"""
    sparse_triangular_solve!(out::Vector, L::SparseMatrixCSC, U::SparseMatrixCSC, b::Vector)

Solves a sparse triangular system `L * U * x = b` in-place, storing the result in `out`.

# Arguments
- `out::Vector`: The output vector, where the solution `x` will be stored.
- `L::SparseMatrixCSC`: The lower triangular matrix in transposed form.
- `U::SparseMatrixCSC`: The upper triangular matrix in transposed form.
- `b::Vector`: The right-hand side vector.
"""
function sparse_triangular_solve!(out::Vector, L::SparseMatrixCSC, U::SparseMatrixCSC, b::Vector)

    out .= b
    n = size(out, 1)

    # Forward Solve
    for i in 1:n
        for j in L.colptr[i]:L.colptr[i + 1] - 2
            out[i] -= L.nzval[j] * out[L.rowval[j]]
        end
        out[i] /= L.nzval[L.colptr[i + 1] - 1]
    end
    
    # Backward Solve
    for i in n:-1:1
        for j in U.colptr[i] + 1:U.colptr[i + 1] - 1
            out[i] -= U.nzval[j] * out[U.rowval[j]]
        end
        out[i] /= U.nzval[U.colptr[i]]
    end

    
end


"""
    sparse_triangular_solve_permutation(L, U, b, left_perm, right_perm)

Solves a sparse triangular system `L * U * x = b` with given row and column permutations.

# Arguments
- `L::SparseMatrixCSC`: The lower triangular matrix in transposed form.
- `U::SparseMatrixCSC`: The upper triangular matrix in transposed form.
- `b::Vector`: The right-hand side vector.
- `left_perm::Vector`: The left permutation vector applied to `b` before solving.
- `right_perm::Vector`: The right permutation vector applied to `x` after solving.

# Returns
- `Vector`: The solution vector `x`, reordered according to `right_perm`.

"""
function sparse_triangular_solve_permutation(L::SparseMatrixCSC, U::SparseMatrixCSC, b::Vector, left_perm::Vector, right_perm::Vector)
    y = Vector{Float64}(undef, size(b, 1))
    x = Vector{Float64}(undef, size(b, 1))
    y .= b[left_perm]
    sparse_triangle_solve!(x, L, U, y)
    return x[right_perm]
end

end