using CAPPABenchmarkTools
using SparseArrays, LinearAlgebra
import CAPPABenchmarkTools.Solvers as S

function synthetic_example(output::String)
    mkpath(output)
    n = 64
    A = spdiagm(-1 => fill(-0.25, n-1), 0 => ones(n), 1 => fill(-0.25, n-1))
    seed = sin.(collect(1.0:n))
    b = A * seed
    # The direct solver API needs the prepared operator's norm estimate, not
    # the preconditioner's norm. This tiny example uses a dense norm evaluation.
    normA = Float64(opnorm(Matrix(A), 2))
    prec, _ = SGS(A; num_iters=1)
    solution_path = joinpath(output, "solution.npy.zstd")
    result = pcg(A, b, seed, normA, prec, Int64(200),
        joinpath(output, "raw_data.npy.zstd"), joinpath(output, "convergence_data");
        convergence_tolerance=1e-10, write_solution=solution_path)
    result.status == S.CONVERGED || error("Example did not converge: $(result.status)")
    x = read_npy_zstd_1d_f64(solution_path)
    norm(b - A*x) / norm(b) <= 1e-10 || error("Returned solution failed residual check")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = isempty(ARGS) ? joinpath(@__DIR__, "output") : abspath(only(ARGS))
    result = synthetic_example(output)
    println("Status: ", result.status, "; iterations: ", result.iterations,
            "; modeled FMA: ", result.total_fma, "; output: ", output)
end
