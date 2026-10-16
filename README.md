# CAPPABenchmarkTools.jl

A Julia framework for benchmarking preconditioned Krylov solvers on sparse
symmetric systems. It records numerical diagnostics and modeled computational
work, separately from measured wall time and resource usage.

This is a matrix-focused infrastructure release, not a production
experiment grid, dataset, scheduler, or paper analysis. See
[release scope](docs/release_scope.md), [changes from 0.1.0](docs/changes.md), and the
[numerical/output contract](docs/numerical_contract.md).

## Capabilities

- PCG, Lanczos-PCG, MINRES, PCR, and two flexible PCG variants.
- Work-counted incomplete factorizations, stationary and polynomial methods,
  sparse approximate inverses, PyAMG, HYPRE, and Laplacians.jl wrappers.
- Component decomposition, symmetric scaling, cached orderings, and bounded
  preparation diagnostics. Preparation estimates the upper eigenvalue; it does
  not promise an accurate smallest eigenvalue or condition number.
- Charged residual certification, bounded continuation without restarts, and
  recorded correction restarts in the default certified mode.
- Bounded diagnostic/coefficient buffers, versioned compressed output, and
  readers for legacy monolithic NPY output (not the old public CSV format).
- Generic phase, resource, and numerical-progress callbacks.

Most methods require an SPD operator and suitable positive-definite
preconditioning. MINRES also supports symmetric indefinite operators with
appropriate SPD preconditioning. This is not a general nonsymmetric solver.

## Installation

The release target is Linux x86-64 with Julia 1.12 (validation uses 1.12.5).
Install Git, CMake, Make, a C/C++ toolchain with OpenMP, and gfortran first.
Allow network access and writable storage during setup. Importing the package
builds missing native libraries; this is not a compilation-free import.

```bash
git clone --recurse-submodules https://github.com/tunnellm/CAPPABenchmarkTools.jl.git
cd CAPPABenchmarkTools.jl
CAPPA_BUILD_JOBS=4 JULIA_NUM_PRECOMPILE_TASKS=2 \
  julia --startup-file=no --project=. scripts/setup.jl
```

`CAPPA_BUILD_JOBS` bounds HYPRE build parallelism (default: at most four
processes); it is not a runtime thread setting.

`Project.toml` selects the pinned Laplacians submodule when this checkout is the
active project. CondaPkg installs NumPy, SciPy, and the pinned editable PyAMG
fork; setup verifies the imported PyAMG path. The forked dependencies carry
work instrumentation and correctness fixes. Stock packages are not equivalent.

For a separate Julia application environment, explicitly develop both the
Laplacians submodule and this package into that environment. Dependency-local
`[sources]` settings must not be assumed to configure a consuming application.
Ordering subprocesses use that application's resolved environment as well.

## Runnable Example

No dataset download is needed:

```bash
julia --startup-file=no --threads=2 --project=. examples/synthetic.jl
```

This solves a small synthetic SPD problem using SGS-preconditioned PCG, checks
the returned solution's residual, and writes diagnostics under `examples/output`.
The known solution is supplied for error diagnostics, not as an initial guess
or input to the stopping policy.

For a complete synthetic preparation -> Control -> preconditioned-run example:

```bash
julia --startup-file=no --threads=2 --project=. examples/matrix_pipeline.jl
```

For file-based benchmarks, place `<name>.mtx` under `matrices/<name>/` and use
`run_matrix(name, preconditioner; ...)`. This is a preparation pipeline, not a
read-only loader: it writes caches and may delete the original MatrixMarket
file after successful preparation. Keep a separate input copy. Non-Control
runs use Control baselines for their work budgets when available. Bounded
continuation requires existing converged baselines; ordinary certified runs can
run without a Control-derived work limit. Components of dimension 50 or less
are skipped by `run_matrix`; use the direct solver API for smaller problems.

## Tests

```bash
julia --startup-file=no --threads=2 --project=. -e 'using Pkg; Pkg.test()'
```

Tests use synthetic fixtures and pinned native/Python backends. They do not
require a cluster or production results. Timing experiments
and machine-dependent allocation thresholds are not presented as correctness
tests in this release.

## Research and License

This software accompanies [An Empirical Study of Conjugate Gradient Preconditioners
for Solving Symmetric Positive Definite Systems of Linear Equations](https://doi.org/10.48550/arXiv.2505.20696).
Results generated with earlier software versions may use different stopping,
accounting, or storage contracts than this release.

To cite the paper:

```bibtex
@misc{tunnell_arxiv_2025,
  author = {Tunnell, Marc A. and Gleich, David F.},
  title = {An Empirical Study of Conjugate Gradient Preconditioners for Solving Symmetric Positive Definite Systems of Linear Equations},
  year = {2025},
  doi = {10.48550/arXiv.2505.20696},
  url = {https://arxiv.org/abs/2505.20696}
}
```

The Julia project is MIT-licensed; see [LICENSE.md](LICENSE.md). External
components retain their own licenses and notices, including the vendored
[LAPACK notice](External/lapack/LICENSE).
