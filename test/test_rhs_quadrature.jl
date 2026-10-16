using Test, SparseArrays, LinearAlgebra, JSON3
using CAPPABenchmarkTools
const U = CAPPABenchmarkTools.Utilities

@testset "Fixed RHS quadrature budgets and checkpoint labels" begin
    A = spdiagm(0 => collect(range(1.0, 10.0; length=120)))
    b = ones(120)
    for (scaled, budget, prefix) in ((false, 60, "quadrature"), (true, 30, "scaled_quadrature"))
        fields = U._rhs_quadrature_metadata(A, b, 10.0; scaled)
        @test fields[prefix * "_k_final"] == budget
        @test fields[prefix * "_max_iters"] == budget
        @test fields[prefix * "_early_stop"] == false
        @test fields[prefix * "_k100"] === nothing
        @test fields[prefix] ≈ 10.0 * sum(1.0 ./ diag(A)) / length(b) rtol=1e-8
        @test fields[prefix * (scaled ? "_k30" : "_k60")] == fields[prefix]
        scaled && @test fields[prefix * "_k60"] === nothing
        @test JSON3.read(JSON3.write(fields))[prefix * "_k100"] === nothing
    end

    # Exact one-dimensional Krylov space: stop without padding later checkpoints.
    fields = U._rhs_quadrature_metadata(spdiagm(0 => ones(8)), ones(8), 1.0)
    @test fields["quadrature"] ≈ 1.0
    @test fields["quadrature_early_stop"]
    @test fields["quadrature_k_final"] == 1
    @test fields["quadrature_k10"] === nothing
    @test fields["quadrature_k30"] === nothing
    @test fields["quadrature_k60"] === nothing
end

@testset "Quadrature-only policy change preserves spectral checkpoints" begin
    mktempdir() do dir
        A = spdiagm(0 => [1.0, 2.0])
        fingerprint = U._component_prepare_fingerprint(A)
        state = U._new_prepare_checkpoint("test", 1, A, fingerprint, 100)
        delete!(state, "quadrature_policy")
        for phase in ("ordering_amd", "unscaled_eigenbounds", "scaled_eigenbounds",
                      "quadrature", "component_metadata")
            state["phases"][phase] = Dict("payload" => "old")
        end
        path = joinpath(dir, "state.json")
        U._atomic_write_json(path, state)
        loaded = U._load_prepare_checkpoint(path, "test", 1, A, fingerprint, 100)
        @test Set(keys(loaded["phases"])) == Set(("ordering_amd", "unscaled_eigenbounds", "scaled_eigenbounds"))
        @test loaded["quadrature_policy"]["scaled_max_iters"] == 30
        @test loaded["quadrature_policy"]["unscaled_max_iters"] == 60
    end
end
