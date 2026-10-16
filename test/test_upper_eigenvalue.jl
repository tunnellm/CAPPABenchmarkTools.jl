using Test, SparseArrays, LinearAlgebra, JSON3
using CAPPABenchmarkTools
const U = CAPPABenchmarkTools.Utilities

@testset "Upper-only preparation estimates" begin
    # A difficult low end should not hold up the well-separated upper end.
    A = spdiagm(0 => vcat(collect(range(1e-10, 1.0; length=399)), 10.0))
    upper = U.estimate_eigbounds(A; target=:upper, m=30, max_restarts=10)
    both = U.estimate_eigbounds(A; target=:both, m=30, max_restarts=10)
    @test upper[1] === upper[5] === upper[7] === nothing
    @test upper[2] ≈ 10.0 rtol=1e-8
    @test upper[6] <= 1e-3 * upper[2]
    @test upper[4] < both[4]
    @test upper[3] < both[3]
    @test_throws ArgumentError U.estimate_eigbounds(A; target=:invalid)
    @test_throws ArgumentError U.estimate_eigbounds(A; m=0)

    fields = U._upper_eigenvalue_metadata(spdiagm(0 => [3.0]))
    @test fields["lambda_min"] === nothing
    @test fields["lambda_max"] == 3.0
    @test fields["lambda_max_status"] == "ritz_converged"
    @test JSON3.read(JSON3.write(fields))["lambda_min"] === nothing

    # An exhausted budget must remain visibly unconverged.
    tight = U.estimate_eigbounds(A; target=:upper, m=2, max_restarts=1, upper_tol=1e-30)
    @test tight[4] == 2
    @test tight[6] > 1e-30 * tight[2]
end

@testset "Preparation checkpoint spectral-policy migration" begin
    mktempdir() do dir
        A = spdiagm(0 => [1.0, 2.0])
        fingerprint = U._component_prepare_fingerprint(A)
        checkpoint = U._new_prepare_checkpoint("test", 1, A, fingerprint, 100)
        checkpoint["eigenbounds_policy"] = Dict("lower_tolerance" => 1e-2)
        for phase in ("ordering_amd", "unscaled_eigenbounds", "scaled_eigenbounds",
                      "quadrature", "component_metadata")
            checkpoint["phases"][phase] = Dict("payload" => "old")
        end
        path = joinpath(dir, "checkpoint.json")
        U._atomic_write_json(path, checkpoint)
        loaded = U._load_prepare_checkpoint(path, "test", 1, A, fingerprint, 100)
        @test collect(keys(loaded["phases"])) == ["ordering_amd"]
        @test loaded["eigenbounds_policy"]["target"] == "upper"
        @test JSON3.read(read(path, String))["eigenbounds_policy"]["target"] == "upper"
    end
end
