using Test
include(joinpath(@__DIR__, "..", "examples", "synthetic.jl"))
include(joinpath(@__DIR__, "..", "examples", "matrix_pipeline.jl"))

@testset "Documented synthetic example" begin
    mktempdir() do output
        result = synthetic_example(output)
        @test result.status == S.CONVERGED
        @test result.certification_true_residual_evals > 0
        data = S.open_diagnostic_data(joinpath(output, "raw_data", "manifest.json"); require_finalized=true)
        @test size(data, 1) == result.iterations + 1
        @test isfile(joinpath(output, "convergence_data_certification.csv"))
    end
end

@testset "Matrix pipeline and cached result reuse" begin
    mktempdir() do output
        records = matrix_pipeline_example(output)
        before = Dict(path => read(path) for path in records)
        @test length(records) == 4
        for path in records
            meta = JSON3.read(read(path, String))
            @test meta.raw_data_storage.version == 2
            @test meta.certification_true_residual_evals > 0
        end
        @test sort(matrix_pipeline_example(output)) == sort(records)
        @test all(read(path) == bytes for (path, bytes) in before)
    end
end
