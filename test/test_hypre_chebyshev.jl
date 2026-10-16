using Test, SparseArrays, LinearAlgebra, Serialization
using CAPPABenchmarkTools
import CAPPABenchmarkTools.Hypre as H

# Independent high-precision residual-polynomial evaluation, not semi-iteration.
function cheby_reference(A, b, x0, degree, variant, scale, lo, hi, ratio)
    D = Diagonal(scale ? inv.(sqrt.(abs.(BigFloat.(diag(A))))) : ones(BigFloat, size(A, 1)))
    B = D * BigFloat.(Matrix(A)) * D
    upper = BigFloat(hi * 1.1)
    lower = BigFloat((hi * 1.1 - lo) * ratio + lo)
    theta, delta = (upper + lower) / 2, (upper - lower) / 2
    r = D * (BigFloat.(b) - BigFloat.(Matrix(A)) * BigFloat.(x0))
    if degree == 1
        return Float64.(BigFloat.(x0) + D * (r / theta))
    end
    m = variant == 1 ? degree - 1 : degree
    if iszero(delta)
        residual = (I - B / theta)^m * r
    else
        t0, t1 = copy(r), (theta * r - B * r) / delta
        c0, c1 = big"1", theta / delta
        for _ in 2:m
            t0, t1 = t1, 2 * (theta * t1 - B * t1) / delta - t0
            c0, c1 = c1, 2 * theta / delta * c1 - c0
        end
        residual = t1 / c1
    end
    variant == 1 && (residual -= B * residual / upper)
    Float64.(BigFloat.(x0) + D * (B \ (r - residual)))
end

function native_cheby(A, b, x0, degree, variant, scale, lo, hi, ratio)
    H.ensure_initialized()
    ijA, parA = H.julia_to_hypre_matrix(A)
    ijb, parb, ijx, parx = H.julia_to_hypre_vectors(length(b))
    solver = Ref{H.HYPRESolver}(C_NULL)
    H.HYPRE_ParChebyCreate(solver)
    s = solver[]
    try
        H.HYPRE_ParChebySetOrder(s, degree)
        H.HYPRE_ParChebySetVariant(s, variant)
        H.HYPRE_ParChebySetScale(s, Int(scale))
        H.HYPRE_ParChebySetMinMaxEigEst(s, lo, hi)
        H.HYPRE_ParChebySetEigRatio(s, ratio)
        H.HYPRE_ParChebySetMaxIterations(s, 1)
        H.HYPRE_ParChebySetTolerance(s, 0.0)
        H.HYPRE_ParChebySetLogging(s, 0)
        H.HYPRE_ParChebySetZeroGuess(s, 0)
        H.HYPRE_ParChebySetup(s, parA, parb, parx)
        indices = Cint.(0:length(b)-1)
        H.HYPRE_IJVectorSetValues(ijb, Cint(length(b)), indices, b)
        H.HYPRE_IJVectorAssemble(ijb)
        H.HYPRE_IJVectorSetValues(ijx, Cint(length(b)), indices, x0)
        H.HYPRE_IJVectorAssemble(ijx)
        H.HYPRE_ParChebySolve(s, parA, parb, parx)
        x = similar(b)
        H.HYPRE_IJVectorGetValues(ijx, Cint(length(b)), indices, x)
        work, setup = Ref{H.HYPREReal}(0), Ref{H.HYPREReal}(0)
        H.HYPRE_ParChebyGetApplyFlops(s, work)
        H.HYPRE_ParChebyGetSetupFlops(s, setup)
        return (; x, work=work[], setup=setup[])
    finally
        H.HYPRE_ParChebyDestroy(s)
        H.HYPRE_IJMatrixDestroy(ijA)
        H.HYPRE_IJVectorDestroy(ijb)
        H.HYPRE_IJVectorDestroy(ijx)
    end
end

const fixtures = [spdiagm(0 => collect(range(0.02, 2.0; length=12))),
                  spdiagm(-1 => fill(-1.0, 11), 0 => fill(2.1, 12), 1 => fill(-1.0, 11)),
                  spdiagm(0 => 10.0 .^ range(-12.0, 0.0; length=12))]

function low_degree_snapshot()
    results = Any[]
    for A in fixtures[1:2], scale in (false, true), variant in (0, 1), degree in 1:4
        D = Diagonal(scale ? inv.(sqrt.(diag(A))) : ones(size(A, 1)))
        lo, hi = extrema(eigvals(Symmetric(Matrix(D * A * D))))
        push!(results, native_cheby(A, sin.(collect(1.0:12)), zeros(12), degree,
                                   variant, scale, lo, hi, 0.3))
    end
    results
end

if !isempty(get(ENV, "CAPPA_CHEBY_WRITE_BASELINE", ""))
    serialize(ENV["CAPPA_CHEBY_WRITE_BASELINE"], low_degree_snapshot())
else
    @testset "Chebyshev requested degree and work" begin
        for A in fixtures, scale in (false, true), variant in (0, 1),
            ratio in (0.0, 0.01, 0.3, 1.0), degree in (1, 2, 3, 4, 5, 8, 16, 32), nonzero in (false, true)
            D = Diagonal(scale ? inv.(sqrt.(diag(A))) : ones(size(A, 1)))
            lo, hi = extrema(eigvals(Symmetric(Matrix(D * A * D))))
            b = sin.(collect(1.0:12))
            x0 = nonzero ? cos.(collect(1.0:12)) : zeros(12)
            result = native_cheby(A, b, x0, degree, variant, scale, lo, hi, ratio)
            expected = cheby_reference(A, b, x0, degree, variant, scale, lo, hi, ratio)
            @test result.x ≈ expected rtol=2e-12 atol=2e-12
            expected_work = degree <= 4 ? degree * (nnz(A) + (scale ? 3 : 1) * 12) :
                degree * nnz(A) + ((scale ? 5 : 4) * degree - 1 - variant) * 12
            @test result.work == expected_work
            @test result.setup == (scale ? 24 : 0) + (degree <= 4 ? 0 : 4degree - 3variant)
        end
    end
    if !isempty(get(ENV, "CAPPA_CHEBY_BASELINE", ""))
        @testset "Degrees 1-4 bitwise unchanged" begin
            old = deserialize(ENV["CAPPA_CHEBY_BASELINE"])
            new = low_degree_snapshot()
            @test length(old) == length(new)
            for (a, b) in zip(old, new)
                @test reinterpret(UInt64, a.x) == reinterpret(UInt64, b.x)
                @test a.work == b.work
                @test a.setup == b.setup
            end
        end
    end
    @testset "Benchmark wrappers and Krylov integration" begin
        A = fixtures[2]
        seed = sin.(collect(1.0:12))
        b = A * seed
        previous = nothing
        for degree in (4, 8, 16)
            prec, meta = Chebyshev(A; order=degree, lanczos_k=10,
                                   method=:conservative, variant=:standard_residual)
            z = zeros(12)
            @test prec.LinearOperator(z, b) == meta["apply_fma"]
            @test meta["effective_order"] == degree
            @test meta["chebyshev_implementation"] ==
                (degree > 4 ? "hypre_host_semi_iteration_v1" : "hypre_legacy_horner")
            ref = cheby_reference(A, b, zeros(12), degree, 0, false,
                                  meta["lambda_min"], meta["lambda_max"], 0.0)
            @test z ≈ ref rtol=2e-12 atol=2e-12
            previous !== nothing && @test norm(z - previous) > 1e-8
            previous = copy(z)
            for solver in (pcg, minres, lanczos_pcg)
                mktempdir() do dir
                    result = solver(A, b, seed, opnorm(Matrix(A)), prec, Int64(100),
                                    joinpath(dir, "raw.npy.zstd"), joinpath(dir, "conv"))
                    @test result.status == CONVERGED
                end
            end
        end
        prec, meta = ChebyshevAdaptive(A; order=8, lanczos_k=:log2_scaled,
                                       method=:conservative, variant=:alternative)
        @test meta["effective_order"] == 8
        z = zeros(12)
        @test prec.LinearOperator(z, b) == meta["apply_fma"]
        @test all(isfinite, z)
        @test_throws ArgumentError H.HypreCheby(A; order=0)
    end

    @testset "Ratio forwarding and cache identity" begin
        A, b = fixtures[2], sin.(collect(1.0:12))
        lo, hi = extrema(eigvals(Symmetric(Matrix(A))))
        for ratio in (0.0, 0.3, 1.0), degree in (1, 4, 8, 16)
            prec, work = H.HypreCheby(A; order=degree, scale=false,
                eig_ratio=ratio, min_eig_est=lo, max_eig_est=hi)
            z = zeros(12)
            @test prec.LinearOperator(z, b) == work
            @test z ≈ cheby_reference(A, b, zeros(12), degree, 0, false, lo, hi, ratio) rtol=2e-12 atol=2e-12
        end
        for ratio in (-0.1, 1.1, NaN, Inf)
            @test_throws ArgumentError H.HypreCheby(A; eig_ratio=ratio)
            @test_throws ArgumentError Chebyshev(A; order=2, lanczos_k=10,
                method=:conservative, variant=:standard_residual, eig_ratio=ratio)
        end
        normalize = CAPPABenchmarkTools.Preconditioners._preconditioner_identity_kwargs
        hash = CAPPABenchmarkTools.TestWrapper._config_hash
        kwargs = Dict{String,Any}("order"=>2, "lanczos_k"=>10,
            "method"=>:conservative, "variant"=>:standard_residual)
        @test normalize(Control, kwargs) === kwargs
        for constructor in (Chebyshev, ChebyshevAdaptive)
            fixed = normalize(constructor, kwargs)
            @test fixed["eig_ratio"] == 0.0
            @test !haskey(kwargs, "eig_ratio")
            @test length(unique(hash.([kwargs, fixed, merge(kwargs, Dict("eig_ratio"=>0.3))]))) == 3
        end
        prec, meta = Chebyshev(A; order=2, lanczos_k=10, method=:conservative,
                              variant=:standard_residual, eig_ratio=0.3)
        z = zeros(12)
        prec.LinearOperator(z, b)
        @test meta["config"]["eig_ratio"] == 0.3
        @test z ≈ cheby_reference(A, b, zeros(12), 2, 0, false,
                                  meta["lambda_min"], meta["lambda_max"], 0.3) rtol=2e-12
    end

    @testset "Shared AMG Chebyshev kernel" begin
        n = 80
        A = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.1, n), 1 => fill(-1.0, n-1))
        b = sin.(collect(1.0:n))
        for scale in (false, true), variant in (0, 1)
            last_work = 0.0
            for degree in (4, 8, 16)
                op = H._build_boomeramg_hierarchy(A; relax_down=:chebyshev,
                    relax_up=:chebyshev, cheby_order=degree, cheby_scale=scale,
                    cheby_variant=variant, cheby_eig_est=10)
                prec, meta = op(; cycle_type=:V)
                x, repeat = zeros(n), zeros(n)
                @test prec.LinearOperator(x, b) == meta["apply_fma"]
                @test prec.LinearOperator(repeat, b) == meta["apply_fma"]
                @test x == repeat
                @test norm(b - A*x) < norm(b)
                @test meta["apply_fma"] > last_work
                last_work = meta["apply_fma"]
            end
        end
    end
end
