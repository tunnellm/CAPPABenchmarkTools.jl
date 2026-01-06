"""
    Dependencies

Wrapper module for external/vendored dependencies.
Each submodule is included from the External/ directory.
"""
module Dependencies

const EXTERNAL_DIR = joinpath(@__DIR__, "..", "External")

# LimitedLDLFactorizations (track-flops branch)
# Adds force_posdef option and FLOP counting
module LimitedLDLFactorizationsFlops
    import ..EXTERNAL_DIR
    include(joinpath(EXTERNAL_DIR, "LimitedLDLFactorizations", "src", "LimitedLDLFactorizations.jl"))
    using .LimitedLDLFactorizations: lldl, lldl_spd, lldl_factorize!, LimitedLDLFactorization, factorized
    export lldl, lldl_spd, lldl_factorize!, LimitedLDLFactorization, factorized
end

# FastSSAI3 - Optimized Julia implementation of SSAI3 preconditioner
module FastSSAI
    import ..EXTERNAL_DIR
    include(joinpath(EXTERNAL_DIR, "FastSSAI3", "src", "FastSSAI3.jl"))
    using .FastSSAI3: ssai3
    export ssai3
end

# ILUK - Incomplete LU/LDL^T with level-of-fill k (track-flops branch)
module ILUKFlops
    import ..EXTERNAL_DIR
    include(joinpath(EXTERNAL_DIR, "ILUK", "src", "ILUK.jl"))
    using .ILUK: ldlt_k, LDLFactorization
    export ldlt_k, LDLFactorization
end

end # module Dependencies
