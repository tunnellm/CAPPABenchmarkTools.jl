# Capacity follows observed iterations, never the potentially enormous 100n cap.
const DIAGNOSTIC_INITIAL_CAPACITY = 256

function _diagnostic_capacity(current::Int, required::Int, limit::Int)
    0 <= required <= limit || throw(ArgumentError("diagnostic capacity exceeds iteration limit"))
    required <= current && return current
    # Geometric chunks give amortized linear copying without integer overflow.
    grown = current + min(current, limit - current)
    return min(limit, max(required, DIAGNOSTIC_INITIAL_CAPACITY, grown))
end

function _grow_diagnostic_vector!(buffer::Vector, required::Int, limit::Int)
    required <= length(buffer) && return buffer
    resize!(buffer,_diagnostic_capacity(length(buffer),required,limit))
end
