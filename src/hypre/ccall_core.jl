# Error checking helper
function hypre_check(err::Cint)
    err == 0 || error("HYPRE error code: $err")
    nothing
end

# Track initialization state
const _hypre_initialized = Ref(false)

function ensure_initialized()
    if !_hypre_initialized[]
        hypre_check(ccall((:HYPRE_Initialize, NativeLibraries.library(LIBHYPRE)), HYPREInt, ()))
        _hypre_initialized[] = true
    end
end

function hypre_finalize()
    if _hypre_initialized[]
        ccall((:HYPRE_Finalize, NativeLibraries.library(LIBHYPRE)), HYPREInt, ())
        _hypre_initialized[] = false
    end
end


# ---------------------------------------------------------------------------
# IJ Matrix interface
# ---------------------------------------------------------------------------

function HYPRE_IJMatrixCreate(comm::HYPREMPIComm, ilower::Integer, iupper::Integer,
                              jlower::Integer, jupper::Integer, matrix::Ref{HYPREIJMatrix})
    hypre_check(ccall((:HYPRE_IJMatrixCreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREMPIComm, HYPREBigInt, HYPREBigInt, HYPREBigInt, HYPREBigInt, Ptr{HYPREIJMatrix}),
                      comm, Cint(ilower), Cint(iupper), Cint(jlower), Cint(jupper), matrix))
end

function HYPRE_IJMatrixDestroy(matrix::HYPREIJMatrix)
    hypre_check(ccall((:HYPRE_IJMatrixDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJMatrix,), matrix))
end

function HYPRE_IJMatrixSetObjectType(matrix::HYPREIJMatrix, type::Integer)
    hypre_check(ccall((:HYPRE_IJMatrixSetObjectType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJMatrix, HYPREInt), matrix, Cint(type)))
end

function HYPRE_IJMatrixInitialize(matrix::HYPREIJMatrix)
    hypre_check(ccall((:HYPRE_IJMatrixInitialize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJMatrix,), matrix))
end

function HYPRE_IJMatrixSetValues(matrix::HYPREIJMatrix, nrows::Integer,
                                 ncols::Vector{Cint}, rows::Vector{Cint},
                                 cols::Vector{Cint}, values::Vector{Cdouble})
    hypre_check(ccall((:HYPRE_IJMatrixSetValues, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJMatrix, HYPREInt, Ptr{HYPREInt}, Ptr{HYPREBigInt},
                       Ptr{HYPREBigInt}, Ptr{HYPREComplex}),
                      matrix, Cint(nrows), ncols, rows, cols, values))
end

function HYPRE_IJMatrixAssemble(matrix::HYPREIJMatrix)
    hypre_check(ccall((:HYPRE_IJMatrixAssemble, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJMatrix,), matrix))
end

function HYPRE_IJMatrixGetObject(matrix::HYPREIJMatrix, object::Ref{Ptr{Cvoid}})
    hypre_check(ccall((:HYPRE_IJMatrixGetObject, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJMatrix, Ptr{Ptr{Cvoid}}), matrix, object))
end

# ---------------------------------------------------------------------------
# IJ Vector interface
# ---------------------------------------------------------------------------

function HYPRE_IJVectorCreate(comm::HYPREMPIComm, jlower::Integer, jupper::Integer,
                              vector::Ref{HYPREIJVector})
    hypre_check(ccall((:HYPRE_IJVectorCreate, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREMPIComm, HYPREBigInt, HYPREBigInt, Ptr{HYPREIJVector}),
                      comm, Cint(jlower), Cint(jupper), vector))
end

function HYPRE_IJVectorDestroy(vector::HYPREIJVector)
    hypre_check(ccall((:HYPRE_IJVectorDestroy, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector,), vector))
end

function HYPRE_IJVectorSetObjectType(vector::HYPREIJVector, type::Integer)
    hypre_check(ccall((:HYPRE_IJVectorSetObjectType, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector, HYPREInt), vector, Cint(type)))
end

function HYPRE_IJVectorInitialize(vector::HYPREIJVector)
    hypre_check(ccall((:HYPRE_IJVectorInitialize, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector,), vector))
end

function HYPRE_IJVectorSetValues(vector::HYPREIJVector, nvalues::Integer,
                                 indices::Vector{Cint}, values::Vector{Cdouble})
    hypre_check(ccall((:HYPRE_IJVectorSetValues, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector, HYPREInt, Ptr{HYPREBigInt}, Ptr{HYPREComplex}),
                      vector, Cint(nvalues), indices, values))
end

function HYPRE_IJVectorAssemble(vector::HYPREIJVector)
    hypre_check(ccall((:HYPRE_IJVectorAssemble, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector,), vector))
end

function HYPRE_IJVectorGetObject(vector::HYPREIJVector, object::Ref{Ptr{Cvoid}})
    hypre_check(ccall((:HYPRE_IJVectorGetObject, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector, Ptr{Ptr{Cvoid}}), vector, object))
end

function HYPRE_IJVectorGetValues(vector::HYPREIJVector, nvalues::Integer,
                                 indices::Vector{Cint}, values::Vector{Cdouble})
    hypre_check(ccall((:HYPRE_IJVectorGetValues, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREIJVector, HYPREInt, Ptr{HYPREBigInt}, Ptr{HYPREComplex}),
                      vector, Cint(nvalues), indices, values))
end

# ---------------------------------------------------------------------------
# ParVector operations
# ---------------------------------------------------------------------------

function HYPRE_ParVectorSetConstantValues(vector::HYPREParVector, value::Float64)
    hypre_check(ccall((:HYPRE_ParVectorSetConstantValues, NativeLibraries.library(LIBHYPRE)), HYPREInt,
                      (HYPREParVector, HYPREComplex), vector, Cdouble(value)))
end
