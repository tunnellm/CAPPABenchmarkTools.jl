using Test, SparseArrays, LinearAlgebra, JSON3
using CAPPABenchmarkTools
using CAPPABenchmarkTools.PreconditionerTypes: Preconditioner
const S = CAPPABenchmarkTools.Solvers
const U = CAPPABenchmarkTools.Utilities
const T = CAPPABenchmarkTools.TestWrapper
const TU = CAPPABenchmarkTools.ThreadedUtilities
identity_prec(n) = Preconditioner((z,r)->(copyto!(z,r);UInt64(n)), UInt64(0), UInt64(n))

@testset "exact solutions and exhausted recurrences" begin
    A = spdiagm(0=>ones(64)); b = ones(64)
    for solver in (:pcg,:pcr,:minres,:lanczos_pcg,:fpcg1,:fpcg2), mode in (:certified,:bounded_continuation)
        mktempdir() do dir
            raw = joinpath(dir,"raw_data.npy.zstd")
            x = fill(NaN,64)
            result = getfield(S,solver)(A,b,b,1.0,identity_prec(64),Int64(8),raw,joinpath(dir,"conv");
                solver_mode=mode,_experimental_final_iterate=x)
            @test result.status == S.CONVERGED
            @test result.iterations == 1
            @test x == b
            @test result.certification_true_residual_evals == 1
            @test size(S.open_diagnostic_data(raw;require_finalized=true),1) == 2
            zero_result = getfield(S,solver)(A,zeros(64),zeros(64),1.0,identity_prec(64),Int64(8),raw,joinpath(dir,"conv");solver_mode=mode)
            @test zero_result.status == S.CONVERGED
            @test zero_result.iterations == 0
            @test zero_result.total_fma == 0
            @test zero_result.certification_true_residual_evals == 0
        end
    end
    mktempdir() do dir
        result = S.minres(spzeros(64,64),b,b,0.0,identity_prec(64),Int64(4),joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv"))
        @test result.status != S.CONVERGED
    end
    for solver in (:minres,:lanczos_pcg,:fpcg1,:fpcg2)
        mktempdir() do dir
            result=getfield(S,solver)(A,b,b,1.0,identity_prec(64),Int64(4),
                joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv");work_limit=Int64(1))
            @test result.status == S.WORK_LIMIT
        end
    end
    for solver in (:minres,:lanczos_pcg)
        mktempdir() do dir
            calls=Ref(0)
            bad=Preconditioner((z,r)->(calls[]+=1; calls[]==1 ? copyto!(z,r) : fill!(z,0.0);UInt64(64)),UInt64(0),UInt64(64))
            result=getfield(S,solver)(spdiagm(0=>Float64.(1:64)),b,b,64.0,bad,Int64(4),
                joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv"))
            @test result.status == S.BREAKDOWN
            @test result.certification_true_residual_evals == 0
        end
    end
end

@testset "exceptional norm range cannot manufacture certification" begin
    for x in ([1e200,-1e200], [1e-200,-1e-200], [0.0,0.0], [1.0,2.0,3.0])
        @test TU.threaded_norm(x) ≈ norm(x)
        @test TU._norm_from_squared(sum(abs2,x),x) ≈ norm(x)
    end
    @test isinf(TU.threaded_norm([Inf,1.0]))
    @test isnan(TU.threaded_norm([NaN,1.0]))
    for mode in (:certified,:bounded_continuation)
        mktempdir() do dir
            A = spdiagm(0=>[1.0,1.0+1e-10]); b = fill(1e-155,2); x=zeros(2)
            result = S.pcg(A,b,Vector(A\b),1.0+1e-10,identity_prec(2),Int64(1),
                joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv"); convergence_tolerance=1e-12,
                solver_mode=mode,_experimental_final_iterate=x)
            @test result.status != S.CONVERGED
            @test result.final_relative_residual ≈ norm(b-A*x)/norm(b)
            @test result.final_relative_residual > 1e-12
        end
    end
end

@testset "initial breakdown is a finalized attempt" begin
    A=spdiagm(0=>ones(64)); b=ones(64)
    bad=Preconditioner((z,r)->(z .= -r;UInt64(64)),UInt64(0),UInt64(64))
    for solver in (:minres,:lanczos_pcg), limit in (Int64(-1),Int64(1))
        mktempdir() do dir
            raw=joinpath(dir,"raw_data.npy.zstd")
            result=getfield(S,solver)(A,b,b,1.0,bad,Int64(8),raw,joinpath(dir,"conv");work_limit=limit)
            @test result.status == (limit < 0 ? S.BREAKDOWN : S.WORK_LIMIT)
            @test result.iterations == 0
            data=S.open_diagnostic_data(raw;require_finalized=true)
            @test size(data,1)==1
            @test sum(data[1,j] for j in 5:7)==result.total_fma
            @test result.total_fma >= 64
            @test all(isfile(joinpath(dir,"conv_"*suffix*".csv")) for suffix in ("relative_residual","relative_normwise_2","certification"))
            T._write_metadata(dir,result,64,Int64(nnz(A)),Dict{String,Any}())
            @test T._should_skip(dir)
        end
    end
end

@testset "completion belongs to its diagnostic attempt" begin
    mktempdir() do dir
        meta=Dict{String,Any}("status"=>"CONVERGED","total_fma"=>1.0)
        @test S._diagnostic_attempt_complete(meta,dir) # Legacy metadata-only.
        raw=joinpath(dir,"raw_data.npy.zstd")
        writer=S.DiagnosticBlockWriter(raw,1,S.PCG_COLUMN_NAMES;block_rows=2)
        @test !S._diagnostic_attempt_complete(meta,dir)
        writer[1,:]=zeros(length(S.PCG_COLUMN_NAMES))
        S._finish_diagnostic_data!(writer,1)
        @test !S._diagnostic_attempt_complete(meta,dir) # Old metadata/new raw.
        S._diagnostic_storage_metadata!(meta,dir)
        @test S._diagnostic_attempt_complete(meta,dir)
        write(joinpath(dir,"metadata.json"),JSON3.write(meta))
        @test T._should_skip(dir)
        S.DiagnosticBlockWriter(raw,1,S.PCG_COLUMN_NAMES;block_rows=2)
        @test !T._should_skip(dir)
        @test !S._diagnostic_attempt_complete(meta,dir)
    end
    mktempdir() do dir
        path=T._control_metadata_path(dir,"fixture",:pcg,1)
        mkpath(dirname(path))
        write(path,JSON3.write(Dict("status"=>"CONVERGED","total_fma"=>100.0)))
        @test T._load_control_cost(dir,"fixture",:pcg,1)==100.0
        S.DiagnosticBlockWriter(joinpath(dirname(path),"raw_data.npy.zstd"),1,S.PCG_COLUMN_NAMES)
        @test T._load_control_cost(dir,"fixture",:pcg,1)===nothing
    end
end

@testset "largest magnitude estimate and shared cache" begin
    mktempdir() do dir
        A=sparse([1.0 -3.0 -3.0; -3.0 1.0 -3.0; -3.0 -3.0 1.0])
        source=Dict("matrix"=>"fixture", "scaled"=>true)
        calls=Ref(0)
        loader=()->(calls[]+=1;A)
        value=U._cached_spectral_norm(loader,dir,source)
        @test value["estimate"] ≈ 5.0
        @test value["canonical_ordering"]=="natural"
        @test value["iterations"] <= 600
        @test U._cached_spectral_norm(()->error("must reuse"),dir,source)==value
        @test calls[]==1
        U._cached_spectral_norm(loader,dir,Dict("matrix"=>"fixture","scaled"=>false))
        @test calls[]==2
        write(joinpath(dir,value["cache_key"]*".json"),"{}")
        @test U._cached_spectral_norm(loader,dir,source)["estimate"] ≈ 5.0
        @test calls[]==3
        for M in (spdiagm(0=>[-7.0,-2.0,-1.0]), spdiagm(0=>[1e-12,1.0,9.0]), sparse(reshape([-2.0],1,1)))
            lo,hi=U.estimate_eigbounds(M;target=:radius,m=3,max_restarts=2)
            @test max(abs(lo),abs(hi)) ≈ opnorm(Matrix(M))
        end
    end
end
