# CSC rows are sorted and structural symmetry is established by preparation.
# Values (including explicit zeros) do not participate in the graph definition.
function _triangle_vertex_counts(A::SparseMatrixCSC)
    n = size(A,1)
    degree = zeros(Int,n)
    for u in 1:n, p in nzrange(A,u)
        degree[u] += A.rowval[p] != u
    end
    before(u,v) = degree[u] < degree[v] || (degree[u] == degree[v] && u < v)
    ptr = ones(Int,n+1)
    for u in 1:n
        count = 0
        for p in nzrange(A,u)
            count += before(u,A.rowval[p])
        end
        ptr[u+1] = ptr[u] + count
    end
    neighbors = Vector{Int}(undef,ptr[end]-1)
    for u in 1:n
        q = ptr[u]
        for p in nzrange(A,u)
            v = A.rowval[p]
            if before(u,v)
                neighbors[q] = v
                q += 1
            end
        end
    end
    counts = zeros(Int,n)
    GC.@preserve counts neighbors ptr begin
    @batch per=core for u in 1:n
        u_count = 0
        for edge in ptr[u]:ptr[u+1]-1
            v = neighbors[edge]
            i, j = ptr[u], ptr[v]
            v_count = 0
            while i < ptr[u+1] && j < ptr[v+1]
                a, b = neighbors[i], neighbors[j]
                if a == b
                    u_count += 1
                    v_count += 1
                    _add_triangle_count!(counts,a,1)
                    i += 1; j += 1
                elseif a < b
                    i += 1
                else
                    j += 1
                end
            end
            v_count == 0 || _add_triangle_count!(counts,v,v_count)
        end
        u_count == 0 || _add_triangle_count!(counts,u,u_count)
    end
    end
    return counts
end

# One flat counter vector, all concurrent writes atomic. Bounds validation and
# GC preservation keep pointer-based atomic updates local and auditable.
@inline function _add_triangle_count!(counts::AbstractVector{Int}, vertex::Int, amount::Int)
    @boundscheck checkbounds(counts,vertex)
    GC.@preserve counts Base.unsafe_modify!(pointer(counts,vertex),+,amount,:monotonic)
    return nothing
end
