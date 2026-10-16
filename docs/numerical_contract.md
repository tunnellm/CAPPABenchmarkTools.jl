# Numerical and output contract

## Target system

Diagnostics and certification refer to the matrix and RHS supplied to the solver.
In the matrix pipeline these are the prepared, possibly scaled system. A small
relative residual for that system does not imply the same relative residual for
the original unscaled input. The supplied seed is a known-solution diagnostic,
not an oracle used for stopping.

## Residual certification

An exactly zero RHS is accepted at iteration zero with `x = 0`, zero relative
residual/backward error by convention, and zero modeled work. This algebraic
acceptance is recorded as `certification_check_source = "zero_rhs_identity"`
with `certification_charged = false`; no matrix or preconditioner application
is performed. Normal diagnostic and convergence files are still written, but
no recurrence coefficients exist. Actual and simulated certification tables
both report iteration zero with zero evaluations and cost. The test is on the
RHS entries themselves, not a norm that could underflow.

Each iteration records an explicitly recomputed residual `b - A*x`. These
diagnostics are not automatically algorithmically visible. A solver-native
crossing schedules a charged certification check. Acceptance requires both the
true relative residual and normwise backward-error diagnostic to meet the
requested tolerance. The operator norm used in the backward-error denominator
is supplied/estimated, so this is not a rigorous interval certificate.

The matrix runner estimates the largest-magnitude Ritz value, not merely the
largest algebraic eigenvalue, so negative eigenvalues contribute to the norm
of indefinite systems. The bounded search uses 30 Lanczos steps per cycle and
at most 20 cycles. Its estimate, Ritz residual, convergence/budget status, and
policy are recorded under `spectral_norm` in solver metadata. The cache is keyed
by component values and scaling, shared across all symmetric orderings, and
serialized across concurrent jobs. Existing component caches remain readable;
missing norm entries are computed lazily without redefining `lambda_max`.

Norm reductions retain the ordinary fast path and use scaled accumulation
when squared magnitudes underflow or overflow. Exact Lanczos termination still
performs the final solution update and charged certification; it is not by
itself a breakdown. Invalid initial preconditioner norms produce finalized
iteration-zero failure output with the attempted setup work recorded.

Default `solver_mode=:certified` allows at most one correction restart after a
failed native crossing, subject to
`relative_normwise_2 <= min(convergence_tolerance, correction_restart_normwise_tolerance)`.
The latter parameter defaults to `sqrt(eps(Float64))`; at the default solver
tolerance `1e-10`, the effective gate is `1e-10`. Correction solves retain a base
solution and accumulate a separate correction. Diagnostics remain relative to
the global solution. A false crossing alone is not evidence of stagnation.

`solver_mode=:bounded_continuation` disables correction restarts. The first
native crossing arms a cumulative-work cap of `continuation_work_factor` times
the work already expended (default factor two). Certification is audited from
that crossing onward. Certification, continuation cap, overall work limit,
iteration limit, or failure can end the solve. An overshot work budget takes
precedence over convergence; convergence exactly at the budget is allowed.
Budgets are checked at defined algorithmic boundaries, not between individual
scalar operations.

## Work versus runtime

PCR requires a fixed linear SPD preconditioner. It uses `z'Az / (Ap' M^-1 Ap)`
and recursively updates both `r` and `z = M^-1 r`, with one preconditioner
application per iteration after setup. Its native test uses the unpreconditioned
recursive `r`. The `ApAp` column remains an unweighted diagnostic; `ApMinvAp`
is the actual denominator. Both it and `alpha` are undefined (`NaN`) at row zero.
PCR's `orthogonality_loss` column records an unweighted image correlation,
not a preconditioned-CR orthogonality test.
The weighted denominator and recursive preconditioned-residual update also
appear in [PETSc's CR implementation](https://petsc.org/main/src/ksp/ksp/impls/cr/cr.c.html).

Modeled work is split into operator, preconditioner, and vector work. A charged
true-residual certification adds one matrix application and `3n` vector-work
units. Physical diagnostic work is recorded separately from the modeled policy
cost. FMA work is a comparison model, not elapsed time or a hardware instruction
counter. Consult setup/application metadata for backend-specific work models.

The matrix runner can return `SKIPPED_COST_GUARD` without executing a solver
when an instrumented application cost already exceeds that solver's entire work
budget. Admission is evaluated separately for each configuration and solver,
not for the whole shared hierarchy. The guard accepts declared fixed costs or
fixed-inner-iteration costs; it does not substitute an arbitrary upper bound
for a variable-cost application. Its metadata records the cost, budget, and
whether exact-zero shortcuts are excluded from the model. Such a record is not
a numerical convergence or failure result and need not have raw diagnostics.

## Storage

Passing `raw_data.npy.zstd` as the logical output path writes current v2 output
to `raw_data/manifest.json` and independent numbered compressed blocks. The
default buffer holds 500,000 rows. Coefficient streams are also bounded; their
Krylov-block identity is separate from storage-block numbering.

Use `Solvers.open_diagnostic_data(path; require_finalized=true)` for a lazy,
one-block-cached reader of completed results. It accepts legacy v1 monolithic
files and v2 manifests. `Solvers.read_npy_zstd(path)` materializes the full matrix
and is unsuitable for enormous histories. An unfinalized manifest exposes only
a durable prefix, not a completed solve. Follow manifest references rather than
globbing for leftover files from earlier attempts.

New terminal metadata includes `raw_data_storage.attempt`. Writers publish an
active-attempt marker before replacing a stream, and runners require its identity
to match the finalized stream and terminal metadata before skipping a solve or
reusing its Control cost. Interrupted retries therefore cannot inherit an old
completion. Legacy results without attempt markers remain readable.

Here v1 means the monolithic NPY storage schema, not package version 0.1.0.
The older public package's compressed CSV output requires its own reader; this
release does not include an automatic conversion of that format.

Flexible-PCG output includes actual recurrence information and a formal PCG-like
tridiagonal reconstruction. The latter is not asserted to be a Ritz projection
of a single fixed operator. Inspect the coefficient schema before interpreting
spectral quantities. Restart boundaries are recorded explicitly.

## Convergence CSVs

- `*_relative_residual.csv` and `*_relative_normwise_2.csv` report diagnostic
  threshold crossings. They are oracle observations, not necessarily policy stops.
- `*_certification.csv` separates first true hits, simultaneous native/true hits,
  actual certification, and per-tolerance policy simulations.
- Actual certification refers to the tolerance of the run. Simulation at another
  tolerance cannot fabricate iterates after a policy would have changed the
  trajectory; divergence/censoring statuses must be respected.

Direct solver calls return `SolverResult` and write the requested artifacts.
`run_matrix` additionally writes configuration, solver metadata, and resource
records. A terminal result need not be converged. Schema and policy metadata,
not directory existence alone, determine how a result should be interpreted.
