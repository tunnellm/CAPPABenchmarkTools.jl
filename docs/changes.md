# Changes from public 0.1.0

Release 0.2.0 replaces the old package layout and
benchmark interface; old invocation examples should be updated rather than
assumed compatible.

## Numerical interface

- Adds preconditioned conjugate residual (PCR), using the weighted denominator
  `Ap' M^-1 Ap` and recording it in the `ApMinvAp` diagnostic column.
- Six solver entry points accept the prepared matrix, RHS, known-solution
  diagnostic seed, operator norm estimate, and work-counted preconditioner.
- Solvers return structured terminal statuses, work counters, and certification
  and correction-restart provenance.
- Certification and bounded continuation are explicit policies. Historical
  true-residual stopping results are not automatically policy-equivalent.
- Preparation estimates the upper eigenvalue with bounded work and records its
  quality/status. Missing lower estimates must not be interpreted as accurate
  condition-number information.

## Output and performance

- Versioned numbered diagnostic and coefficient blocks bound buffer growth.
  Legacy monolithic NPY diagnostic output remains readable, without rewriting it.
  Public 0.1.0's compressed CSV histories are not accepted by the NPY reader and
  are not automatically converted. Use separate cache and result directories
  when upgrading from that release; preparation-cache compatibility is not a
  blanket guarantee across all historical layouts.
- Threaded/fused kernels and native thread-pool handoffs reduce overhead.
  Floating-point reduction order can change; bitwise equivalence across versions
  or thread counts is not promised.
- Work counters, physical diagnostics, resource samples, and phase events serve
  different purposes and must not be conflated.
- Instrumented HYPRE Chebyshev supports higher host degrees and forwards explicit
  interval ratios. See [the backend note](hypre_chebyshev_extension.md).

## Packaging

- Pinned public source dependencies, explicit Laplacians source selection,
  bounded HYPRE build parallelism, and a checked Python setup path.
- A standard synthetic test entry point and examples without production data.
- Matrix-only release scope. No series orchestration, cluster scheduler,
  historical migration tools, or unpublished study results are included.

These changes do not retroactively change the contract of published datasets.
Use the producing version and recorded schema/policy when interpreting results.
