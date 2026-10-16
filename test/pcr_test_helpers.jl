# PCR applies M^-1 to Ap, not to r, inside the loop. Retain the initial
# residual reference so the injected fault still targets the native residual.
function pcr_false_crossing_once()
    calls = Ref(0)
    residual = Ref(Float64[])
    initial_z = Ref(Float64[])
    apply! = function(out, input)
        calls[] += 1
        copyto!(out, input)
        if calls[] == 1
            residual[] = input
            initial_z[] = copy(out)
        elseif calls[] == 2
            alpha = dot(initial_z[], input) / dot(input, input)
            residual[] .= alpha .* input
        end
        return UInt64(0)
    end
    return Preconditioner(apply!, UInt64(0), UInt64(0))
end
