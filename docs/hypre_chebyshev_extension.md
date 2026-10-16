# Instrumented HYPRE Chebyshev

The pinned HYPRE fork extends the host implementation beyond the upstream
degree-four clamp. Degrees one through four use the existing Horner path;
higher degrees use Chebyshev semi-iteration with O(degree) scalar coefficients
and O(n) work vectors. The GPU path rejects unsupported high degrees.

The wrapper forwards `eig_ratio`, including zero, even with explicit spectral
bounds. Ratios must be finite and in [0, 1]. `Chebyshev` and
`ChebyshevAdaptive` include their default `eig_ratio=0.0` in result identities.
BoomerAMG's separate Chebyshev fraction parameter is not this wrapper setting.

For degree m and alternative-variant indicator v, high-degree application work
is `m*nnz + (4m-1-v)*n` unscaled, or `m*nnz + (5m-1-v)*n` scaled. Additional
scalar setup costs `4m-3v`. Lower-degree accounting follows its original path.

`test/test_hypre_chebyshev.jl` compares independent high-precision residual
polynomials with both variants, scaling choices, zero/nonzero initial guesses,
degrees 1-5/8/16/32, and collapsed intervals. It also checks work accounting,
benchmark identities, and AMG integration.

Do not reuse a native library built from another HYPRE revision. Historical
outputs without explicit effective degree and interval provenance cannot be
assumed equivalent to this implementation.
