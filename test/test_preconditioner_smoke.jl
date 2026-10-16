using Test, SparseArrays, LinearAlgebra, CAPPABenchmarkTools
import CAPPABenchmarkTools.Solvers as S

@testset "Pinned algebraic backends" begin
    n = 32
    A = spdiagm(-1 => fill(-0.25,n-1), 0 => ones(n), 1 => fill(-0.25,n-1))
    seed = sin.(collect(1.0:n))
    b = A*seed
    normA = Float64(opnorm(Matrix(A),2))
    builders = [
        "SSOR" => A -> SSOR(A; num_iters=1, ω=1.0),
        "SGS" => A -> SGS(A; num_iters=1),
        "TNS" => A -> TruncatedNeumann(A; num_iters=2, lanczos_k=0, method=:norm_safe),
        "LLDL-shift" => A -> LimitedLDLShiftPD(A; memory_factor=2, droptol=0.0),
        "LLDL-abs" => A -> LimitedLDLAbsPD(A; memory_factor=2, droptol=0.0),
        "ILUK-shift" => A -> ILUKShiftPD(A; ilu_level=0),
        "ILUK-abs" => A -> ILUKAbsPD(A; ilu_level=0),
        "SSAI" => A -> SSAI(A; fill_factor=2),
        "Laplacians" => A -> LaplaciansDeg(A),
    ]
    for (name, build) in builders
        @testset "$name" begin
            prec, meta = build(copy(A))
            z = zeros(n)
            work = prec.LinearOperator(z,b)
            @test all(isfinite,z)
            @test isfinite(work) && work >= 0
            @test dot(b,z) > 0
            for solver in (S.pcg, S.pcr)
                mktempdir() do dir
                    x = zeros(n)
                    result = solver(A,b,seed,normA,prec,Int64(200),
                        joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv");
                        _experimental_final_iterate=x)
                    @test result.status == S.CONVERGED
                    @test result.final_relative_residual <= 1e-10
                    @test norm(b-A*x)/norm(b) <= 1e-10
                end
            end
        end
    end
end
