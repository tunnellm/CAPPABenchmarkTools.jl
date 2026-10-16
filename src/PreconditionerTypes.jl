module PreconditionerTypes

export Preconditioner, OptionedPreconditioner

"""
    Preconditioner(LinearOperator, generation_fma, generation_graph)

Encapsulates a preconditioner for iterative solvers.

# Fields
- `LinearOperator::F`: `LinearOperator(z, r) -> Float64` — applies the preconditioner
  (solving `Mz = r` into `z`) and returns the fused multiply-add count for that application.
  For fixed-cost preconditioners this is a constant; for variable-cost (e.g. K-cycle with
  early exit) it may differ per call.
- `generation_fma::UInt64`: Fused multiply-add count for constructing the preconditioner.
- `generation_graph::UInt64`: Graph operations count (edge traversals, union-find, etc.) during construction.
"""
struct Preconditioner{F}
    LinearOperator::F
    generation_fma::UInt64
    generation_graph::UInt64
end

"""
    OptionedPreconditioner(configure)

A preconditioner that can be reconfigured with different apply-time options
without rebuilding the underlying hierarchy/factorization.

Callable: `op(; kwargs...) -> (Preconditioner, Dict)` reconfigures apply-time
parameters (e.g. cycle type, smoothers) and returns a fresh `Preconditioner`
plus its metadata dict. The hierarchy is shared across calls; only the closure
and cost accounting change. Cold-path only — not called in the solve loop.

# Fields
- `configure::F`: The reconfiguration function. Access via calling the struct directly.
"""
struct OptionedPreconditioner{F}
    configure::F
end

(op::OptionedPreconditioner)(; kwargs...) = op.configure(; kwargs...)

end # module PreconditionerTypes
