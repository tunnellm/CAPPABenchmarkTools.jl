using Test, SparseArrays, CAPPABenchmarkTools
const U = CAPPABenchmarkTools.Utilities
checkout, output = ARGS

@testset "Consumer ordering subprocesses" begin
    @test realpath(dirname(dirname(pathof(CAPPABenchmarkTools)))) == realpath(checkout)
    @test !isfile(joinpath(checkout,"Manifest.toml"))
    A = spdiagm(-1=>fill(-1.0,3),0=>fill(3.0,4),1=>fill(-1.0,3))
    U.write_npy_zstd_1d_i64(joinpath(output,"colptr.npy.zstd"),A.colptr)
    U.write_npy_zstd_1d_i64(joinpath(output,"rowval.npy.zstd"),A.rowval)
    files = Dict("colptr"=>"colptr.npy.zstd","rowval"=>"rowval.npy.zstd")
    for ordering in ("rcm","amd","metis")
        result = U._bounded_ordering(output,files,4,ordering,output,1;timeout_seconds=120.0)
        @test !result.skipped
        @test result.permutation == U._ordering_permutation(A,ordering)
    end
    timeout = U._bounded_ordering(output,files,4,"amd",output,1;timeout_seconds=0.001)
    @test timeout.skipped
    @test timeout.permutation === nothing
end
