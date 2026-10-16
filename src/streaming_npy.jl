# The header is bounded by NPY 1.0's UInt16 length. Only the final payload
# vector is proportional to input size; checking EOF also validates the frame.
function _read_npy_zstd_vector(path::String, ::Type{T}, descr::String) where T
    open(path, "r") do io
        stream = ZstdDecompressorStream(io)
        try
            prefix = read(stream, 10)
            length(prefix) == 10 || error("Truncated NPY prefix in $path")
            prefix[1:6] == UInt8[0x93,0x4e,0x55,0x4d,0x50,0x59] || error("Invalid NPY magic in $path")
            prefix[7:8] == UInt8[1,0] || error("Unsupported NPY version in $path")
            header_len = Int(prefix[9]) | (Int(prefix[10]) << 8)
            header = read(stream, header_len)
            length(header) == header_len || error("Truncated NPY header in $path")
            h = _parse_npy_1d_header(vcat(prefix, header))
            h.descr == descr || error("Expected $descr dtype, got $(h.descr) in $path")
            Base.checked_mul(h.n, sizeof(T))
            values = Vector{T}(undef, h.n)
            read!(stream, reinterpret(UInt8, values))
            eof(stream) || error("Extra NPY payload in $path")
            if ENDIAN_BOM != 0x04030201
                bits = reinterpret(UInt64, values)
                bits .= ltoh.(bits)
            end
            return values
        finally
            close(stream)
        end
    end
end

function _write_npy_payload(io, values, ::Type{T}) where T
    if values isa Vector{T} && ENDIAN_BOM == 0x04030201
        write(io, reinterpret(UInt8, values))
        return nothing
    end
    # Conversion/noncontiguous inputs never materialize a second full vector.
    buffer = Vector{T}(undef, min(length(values), 131072))
    used = 0
    for value in values
        used += 1
        buffer[used] = value
        if used == length(buffer)
            bits = reinterpret(UInt64, buffer)
            ENDIAN_BOM == 0x04030201 || (bits .= htol.(bits))
            write(io, reinterpret(UInt8, buffer))
            used = 0
        end
    end
    if used > 0
        part = view(buffer, 1:used)
        bits = reinterpret(UInt64, part)
        ENDIAN_BOM == 0x04030201 || (bits .= htol.(bits))
        write(io, reinterpret(UInt8, part))
    end
    return nothing
end
