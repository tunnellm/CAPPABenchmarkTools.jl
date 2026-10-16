# ---------------------------------------------------------------------------
# Julia SparseMatrixCSC ↔ HYPRE IJ conversion utilities
# ---------------------------------------------------------------------------

using SparseArrays

"""
    julia_to_hypre_matrix(A::SparseMatrixCSC{Float64,Int}) -> (ij_A, parcsr_A)

Convert a Julia sparse matrix to HYPRE IJ/ParCSR format.
Returns the IJ matrix handle and the extracted ParCSR object pointer.
HYPRE uses 0-based indexing; Julia uses 1-based.
"""
function julia_to_hypre_matrix(A::SparseMatrixCSC{Float64,Int})
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"

    _sleep_julia_workers!()

    # Create IJ matrix with 0-based row/col range
    ij_A = Ref{HYPREIJMatrix}(C_NULL)
    HYPRE_IJMatrixCreate(HYPRE_MPI_COMM_WORLD, Cint(0), Cint(n - 1),
                         Cint(0), Cint(n - 1), ij_A)
    HYPRE_IJMatrixSetObjectType(ij_A[], HYPRE_PARCSR)
    HYPRE_IJMatrixInitialize(ij_A[])

    # For symmetric matrices, CSC columns are CSR rows — no transpose needed.
    # Column j of CSC stores the nonzeros in row j (since A = A^T).
    # Preallocate buffers sized to max nonzeros per row to avoid per-row allocation.
    max_nnz_row = maximum(A.colptr[i+1] - A.colptr[i] for i in 1:n)
    row_idx = Vector{Cint}(undef, 1)
    ncols = Vector{Cint}(undef, 1)
    cols = Vector{Cint}(undef, max_nnz_row)
    vals = Vector{Cdouble}(undef, max_nnz_row)
    for row in 1:n
        col_start = A.colptr[row]
        col_end = A.colptr[row + 1] - 1
        nnz_row = col_end - col_start + 1

        if nnz_row > 0
            # 0-based indices for HYPRE
            row_idx[1] = Cint(row - 1)
            ncols[1] = Cint(nnz_row)
            for j in col_start:col_end
                idx = j - col_start + 1
                cols[idx] = Cint(A.rowval[j] - 1)
                vals[idx] = A.nzval[j]
            end

            HYPRE_IJMatrixSetValues(ij_A[], Cint(1), ncols, row_idx, cols, vals)
        end
    end

    HYPRE_IJMatrixAssemble(ij_A[])

    # Extract ParCSR object
    parcsr_A = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJMatrixGetObject(ij_A[], parcsr_A)

    return ij_A[], reinterpret(HYPREParCSRMatrix, parcsr_A[])
end

"""
    julia_to_hypre_vectors(n::Int) -> (ij_b, par_b, ij_x, par_x)

Create HYPRE IJ/Par vectors for RHS (b) and solution (x) of size n.
Both vectors are initialized to zero.
"""
function julia_to_hypre_vectors(n::Int)
    # Create RHS vector
    ij_b = Ref{HYPREIJVector}(C_NULL)
    HYPRE_IJVectorCreate(HYPRE_MPI_COMM_WORLD, Cint(0), Cint(n - 1), ij_b)
    HYPRE_IJVectorSetObjectType(ij_b[], HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(ij_b[])
    HYPRE_IJVectorAssemble(ij_b[])

    par_b = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(ij_b[], par_b)

    # Create solution vector
    ij_x = Ref{HYPREIJVector}(C_NULL)
    HYPRE_IJVectorCreate(HYPRE_MPI_COMM_WORLD, Cint(0), Cint(n - 1), ij_x)
    HYPRE_IJVectorSetObjectType(ij_x[], HYPRE_PARCSR)
    HYPRE_IJVectorInitialize(ij_x[])
    HYPRE_IJVectorAssemble(ij_x[])

    par_x = Ref{Ptr{Cvoid}}(C_NULL)
    HYPRE_IJVectorGetObject(ij_x[], par_x)

    return ij_b[], reinterpret(HYPREParVector, par_b[]),
           ij_x[], reinterpret(HYPREParVector, par_x[])
end
