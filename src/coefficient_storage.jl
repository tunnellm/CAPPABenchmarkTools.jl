# Storage chunks and recurrence restarts are independent: each restarted
# recurrence owns a manifest, whose numbered files contain bounded row blocks.
mutable struct CoefficientStream
    raw_path::String
    suffix::String
    columns::Vector{String}
    schema::Dict{String,Any}
    limit::Int
    block_rows::Int
    callback::Any
    block_index::Int
    first_iteration::Int
    rows::Int
    writer::Union{Nothing,DiagnosticBlockWriter}
    pending_diagonal::Union{Nothing,Float64}
end

function _coefficient_path(raw, suffix, block_index)
    restart = block_index == 0 ? "" : "_r$(block_index)"
    return replace(raw, r"\.[^/]*$" => "$(suffix)$(restart).npy.zstd")
end

_tridiag_block_path(raw::String, index::Int) =
    _diagnostic_manifest(_coefficient_path(raw, "_tridiag", index))

function CoefficientStream(raw, limit, block_rows; suffix="_tridiag",
                           columns=["diagonal", "offdiagonal_to_next"],
                           interpretation="fixed_operator_reconstruction",
                           callback=NOOP_PHASE_CALLBACK, source_attempt=nothing)
    schema = Dict{String,Any}("version"=>2, "kind"=>"recurrence_coefficients",
        "interpretation"=>interpretation, "row_iteration_offset"=>1,
        "ritz_guarantee"=>false, "source_attempt"=>source_attempt)
    return CoefficientStream(raw, suffix, columns, schema, max(1,limit), block_rows,
        callback, 0, 1, 0, nothing, nothing)
end

function _append_coefficients!(stream::CoefficientStream, values::Vararg{Any,N}) where {N}
    if stream.writer === nothing
        schema = merge(stream.schema, Dict("krylov_block"=>stream.block_index,
            "first_iteration"=>stream.first_iteration))
        stream.writer = DiagnosticBlockWriter(
            _coefficient_path(stream.raw_path, stream.suffix, stream.block_index),
            stream.limit, stream.columns; block_rows=stream.block_rows,
            callback=stream.callback, schema=schema)
    end
    writer = stream.writer::DiagnosticBlockWriter
    length(values) == size(writer,2) || throw(DimensionMismatch("coefficient columns"))
    row = stream.rows + 1
    _prepare_diagnostic_row!(writer, row, stream.limit)
    for (col,value) in enumerate(values)
        writer[row,col] = value
    end
    stream.rows = row
    return nothing
end

# PCG-like recovery obtains beta linking rows k-1 and k only at step k.
# Keep that one unresolved row out of the durable prefix until beta arrives.
function _append_delayed_tridiagonal!(stream::CoefficientStream, diagonal, previous_beta)
    if stream.pending_diagonal !== nothing
        _append_coefficients!(stream, stream.pending_diagonal, previous_beta)
    end
    stream.pending_diagonal = diagonal
    return nothing
end

function _close_coefficients!(stream::CoefficientStream;
        next_iteration::Int=stream.first_iteration+stream.rows+Int(stream.pending_diagonal !== nothing),
        callback=stream.callback)
    stream.callback = callback
    stream.writer === nothing || (stream.writer.callback = callback)
    if stream.pending_diagonal !== nothing
        _append_coefficients!(stream, stream.pending_diagonal, 0.0)
        stream.pending_diagonal = nothing
    end
    path = nothing
    if stream.writer !== nothing
        writer = stream.writer::DiagnosticBlockWriter
        _finish_diagnostic_data!(writer, stream.rows)
        path = joinpath(writer.directory, "manifest.json")
    end
    stream.writer = nothing
    stream.rows = 0
    stream.block_index += 1
    stream.first_iteration = next_iteration
    return path
end

mutable struct FlexibleCoefficientStreams
    formal::CoefficientStream
    actual::CoefficientStream
    previous_alpha::Float64
    previous_previous_alpha::Float64
    local_step::Int
end

function FlexibleCoefficientStreams(raw, limit, block_rows;
                                    callback=NOOP_PHASE_CALLBACK, source_attempt=nothing)
    formal = CoefficientStream(raw, limit, block_rows;
        interpretation="formal_flexible_cg_orthogonal_polynomial_reconstruction",
        callback=callback, source_attempt=source_attempt)
    merge!(formal.schema, Dict(
        "diagonal_formula"=>"1/alpha_k + c1_k/alpha_previous",
        "previous_offdiagonal_formula"=>"sqrt(abs(c1_k))/alpha_previous",
        "longer_direction_terms_omitted"=>true,
        "negative_c1_handling"=>"absolute_value; flagged in companion recurrence",
        "abs_flag_alignment"=>"offdiagonal row k-1 uses recurrence row k formal_beta_abs_flag",
        "fixed_operator_assumed"=>false))
    columns = ["iteration", "local_step", "alpha_used", "c1_applied", "c2_applied",
        "r_dot_z", "p_dot_r", "p_dot_Ap", "norm_r_before", "norm_z_before",
        "residual_previous_previous", "residual_previous", "residual_current",
        "residual_next", "formal_beta_abs_flag"]
    actual = CoefficientStream(raw, limit, block_rows; suffix="_recurrence", columns=columns,
        interpretation="Az_k = sum(h_jk*r_j); p_k=z_k+c1*p_previous+c2*p_previous_previous",
        callback=callback, source_attempt=source_attempt)
    actual.schema["identity_scope"] = "exact_arithmetic; floating_point_recurrence_defects_not_stored"
    return FlexibleCoefficientStreams(formal, actual, 0.0, 0.0, 0)
end

function _record_flexible_coefficients!(s::FlexibleCoefficientStreams, iteration,
        alpha, c1, c2, rho, pr, pAp, norm_r, norm_z)
    s.local_step += 1
    t1 = s.local_step > 1 ? c1 / s.previous_alpha : 0.0
    t2 = s.local_step > 2 ? c2 / s.previous_previous_alpha : 0.0
    diagonal = inv(alpha) + t1
    previous_beta = s.local_step > 1 ? sqrt(abs(c1)) / s.previous_alpha : 0.0
    _append_delayed_tridiagonal!(s.formal, diagonal, previous_beta)
    _append_coefficients!(s.actual, Float64(iteration), Float64(s.local_step), alpha, c1, c2,
        rho, pr, pAp, norm_r, norm_z, -t2, -t1+t2, diagonal, -inv(alpha),
        c1 < 0.0 ? 1.0 : 0.0)
    s.previous_previous_alpha = s.previous_alpha
    s.previous_alpha = alpha
    return nothing
end

function _close_flexible_coefficients!(s::FlexibleCoefficientStreams;
                                      next_iteration::Int, callback=s.formal.callback)
    formal_path = _close_coefficients!(s.formal; next_iteration=next_iteration, callback=callback)
    actual_path = _close_coefficients!(s.actual; next_iteration=next_iteration, callback=callback)
    s.previous_alpha = 0.0
    s.previous_previous_alpha = 0.0
    s.local_step = 0
    return formal_path, actual_path
end
