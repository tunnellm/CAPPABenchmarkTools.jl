using Test, SparseArrays, LinearAlgebra, PythonCall, CAPPABenchmarkTools
using CAPPABenchmarkTools.Preconditioners: PyAMGRuStu, PyAMGSmAgg, PyAMGRootnode, PyAMGPairwise
using CAPPABenchmarkTools.PreconditionerTypes: OptionedPreconditioner

@testset "PyAMG single-level optioned applications" begin
    n = 8
    A = spdiagm(-1=>fill(-1.0,n-1),0=>fill(3.0,n),1=>fill(-1.0,n-1))
    rhs = sin.(collect(1.0:n))
    for builder in (PyAMGRuStu, PyAMGSmAgg, PyAMGRootnode, PyAMGPairwise)
        hierarchy = builder(A, OptionedPreconditioner; max_coarse=16)
        @test_throws ErrorException hierarchy(cycle=:K, k_cg_iters=0)
        for coarse in (:pinv, :lu, :cholesky), cycle in (:V, :W, :F, :K, :AMLI)
            prec, meta = hierarchy(; cycle, coarse_solver=coarse, num_cycles=2)
            @test meta["num_levels"] == 1
            z = zeros(n)
            work = prec.LinearOperator(z, rhs)
            @test z ≈ A \ rhs
            @test work == meta["apply_fma"] == 2*n^2
            prec.LinearOperator(z, zeros(n))
            @test all(iszero, z)
        end
    end
end

@testset "PyAMG configuration-local setup costs" begin
    n = 128
    A = spdiagm(-1=>fill(-1.0,n-1),0=>fill(3.0,n),1=>fill(-1.0,n-1))
    rhs = sin.(collect(1.0:n))
    hierarchy = PyAMGRuStu(A, OptionedPreconditioner)
    function sample(;kwargs...)
        prec, meta = hierarchy(;kwargs...)
        z = zeros(n)
        work = prec.LinearOperator(z,rhs)
        @test all(isfinite,z)
        @test work > 0
        parts = meta["generation_cost_breakdown"]
        @test prec.generation_fma == meta["generation_fma"] == sum(p["fma"] for p in values(parts))
        @test prec.generation_graph == sum(p["graph"] for p in values(parts))
        return z, meta
    end
    baseline, base_meta = sample()
    jac = (:jacobi,(omega=2/3,iterations=2))
    cheb = (:chebyshev,(degree=3,iterations=1))
    jz, jm = sample(presmoother=jac,postsmoother=jac)
    cz, cm = sample(presmoother=cheb,postsmoother=cheb)
    for meta in (jm,cm)
        @test meta["generation_cost_breakdown"]["smoother"]["fma"] > 0
        @test meta["generation_cost_breakdown"]["hierarchy"] == base_meta["generation_cost_breakdown"]["hierarchy"]
    end
    z, m = sample()
    @test z == baseline
    @test m["generation_cost_breakdown"] == base_meta["generation_cost_breakdown"]
    for (spec, expected_z, expected_meta) in ((cheb,cz,cm),(jac,jz,jm))
        z, m = sample(presmoother=spec,postsmoother=spec)
        @test z == expected_z
        @test m["generation_cost_breakdown"] == expected_meta["generation_cost_breakdown"]
    end
    for coarse in (:lu,:cholesky)
        z, m = sample(coarse_solver=coarse)
        @test z ≈ baseline
        nc = m["coarse_n"]
        @test m["generation_cost_breakdown"]["coarse"]["fma"] == nc^3 ÷ (coarse == :lu ? 3 : 6)
    end
    z, m = sample()
    @test z == baseline
    @test m["generation_cost_breakdown"] == base_meta["generation_cost_breakdown"]
    @test_throws ErrorException hierarchy(coarse_solver=:invalid)
end

@testset "Optioned smoother and coarse alternatives solve" begin
    n = 128
    A = spdiagm(-1=>fill(-1.0,n-1),0=>fill(3.0,n),1=>fill(-1.0,n-1))
    seed = sin.(collect(1.0:n))
    b = A * seed
    specs = [(presmoother=s,postsmoother=s) for s in (
        (:jacobi,(omega=2/3,withrho=true,iterations=1)),
        (:jacobi,(omega=1.0,withrho=true,iterations=1)),
        (:chebyshev,(degree=2,iterations=1)),
        (:chebyshev,(degree=4,iterations=1)))]
    opts = Any[specs...,(coarse_solver=:lu,),(coarse_solver=:cholesky,)]
    mktempdir() do dir
        for builder in (PyAMGRuStu,PyAMGSmAgg,PyAMGRootnode,PyAMGPairwise)
            hierarchy = builder(A,OptionedPreconditioner)
            for (i,kwargs) in enumerate(opts)
                prec, meta = hierarchy(;kwargs...)
                for solver in (pcg,minres,lanczos_pcg)
                    name = "$(nameof(builder))_$(i)_$(nameof(solver))"
                    result = solver(A,b,seed,5.0,prec,Int64(500),
                        joinpath(dir,name*".npy.zstd"),joinpath(dir,name);
                        solver_mode=:bounded_continuation)
                    @test string(result.status) == "CONVERGED"
                    @test result.final_relative_residual <= 1e-10
                end
            end
        end
    end
end
