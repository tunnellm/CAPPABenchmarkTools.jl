using Test

# Separate processes keep fixture names and native runtime state isolated.
files = [
    "test_solver_boundaries.jl",
    "test_spectral_cache_concurrency.jl",
    "test_release_boundaries.jl",
    "test_streaming_npy.jl",
    "test_thread_handoffs.jl",
    "test_upper_eigenvalue.jl",
    "test_rhs_quadrature.jl",
    "test_decompose_pipeline.jl",
    "test_diagnostic_storage.jl",
    "test_recurrence_storage.jl",
    "test_pcr.jl",
    "test_separate_correction.jl",
    "test_bounded_continuation.jl",
    "test_hypre_chebyshev.jl",
    "test_pyamg_configuration_costs.jl",
    "test_preconditioner_smoke.jl",
    "test_release_examples.jl",
]
project = dirname(Base.active_project())
@testset "Public infrastructure" begin
    for file in files
        @testset "$file" begin
            println("Running ", file)
            flush(stdout)
            command = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$project $(joinpath(@__DIR__, file))`
            @test success(run(ignorestatus(addenv(command, "OPENBLAS_NUM_THREADS" => "1"))))
        end
    end
end
