using Test, SparseArrays, LinearAlgebra, Random, JSON3
using CAPPABenchmarkTools
import CAPPABenchmarkTools.Solvers as S
import CAPPABenchmarkTools.ThreadedUtilities as T
using CAPPABenchmarkTools.PreconditionerTypes: Preconditioner

function recurrence_problem()
    n=30
    A=spdiagm(-1=>fill(-1.,n-1),0=>collect(range(2.2,4.0,length=n)),1=>fill(-1.,n-1))
    seed=randn(MersenneTwister(73),n)
    return A,A*seed,seed
end

@testset "Julia dot reduction, no BLAS dependency" begin
    ws = T.FusedReductionWorkspace()
    for n in (0,1,7,128,65539)
        rng = MersenneTwister(13)
        x = randn(rng,n); y = randn(rng,n)
        ref = sum(BigFloat.(x).*BigFloat.(y); init=BigFloat(0))
        v = T.threaded_dot(x,y,ws)
        @test abs(BigFloat(v)-ref) <= 32eps(Float64)*max(1,sum(abs.(BigFloat.(x).*BigFloat.(y))))
        previous = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            a = T.threaded_dot(x,y,ws)
            BLAS.set_num_threads(3)
            @test isequal(a,T.threaded_dot(x,y,ws))
        finally
            BLAS.set_num_threads(previous)
        end
    end
    @test_throws DimensionMismatch T.threaded_dot(ones(2),ones(3),ws)
    @test isnan(T.threaded_dot([NaN],[1.0],ws))
    @test T.threaded_dot([Inf],[1.0],ws) == Inf
    x=ones(1000); T.threaded_dot(x,x,ws)
    @test (@allocated T.threaded_dot(x,x,ws)) < 4096
end

@testset "coefficient appends do not allocate between flushes" begin
    mktempdir() do dir
        stream=S.CoefficientStream(joinpath(dir,"raw.npy.zstd"),100,100)
        S._append_coefficients!(stream,1.,2.)
        S._append_coefficients!(stream,1.,2.)
        @test (@allocated S._append_coefficients!(stream,1.,2.)) == 0
        flexible=S.FlexibleCoefficientStreams(joinpath(dir,"flex.npy.zstd"),100,100)
        for k in 1:3
            S._record_flexible_coefficients!(flexible,k,1.,0.1,0.2,1.,1.,1.,1.,1.)
        end
        @test (@allocated S._record_flexible_coefficients!(flexible,4,1.,0.1,0.2,1.,1.,1.,1.,1.)) == 0
    end
end

@testset "bounded coefficient storage, delayed beta, and restarts" begin
    mktempdir() do dir
        large=S.CoefficientStream(joinpath(dir,"large.npy.zstd"),10^9,S.DIAGNOSTIC_BLOCK_ROWS)
        S._append_coefficients!(large,1.,2.)
        @test size(large.writer.buffer) == (500_000,2)
        @test sizeof(large.writer.buffer) == 8_000_000
        S._close_coefficients!(large)
        raw=joinpath(dir,"raw.npy.zstd")
        s=S.CoefficientStream(raw,10^9,3)
        for k in 1:8
            S._append_delayed_tridiagonal!(s,Float64(k),Float64(-k))
            s.writer === nothing || @test size(s.writer.buffer) == (3,2)
        end
        snapshot=S.open_diagnostic_data(S._tridiag_block_path(raw,0))
        @test size(snapshot) == (6,2)
        @test snapshot[6,2] == -7.0
        p=S._close_coefficients!(s; next_iteration=9)
        @test isfile(p)
        data=S.read_npy_zstd(p)
        @test data[:,1] == collect(1.0:8.0)
        @test data[:,2] == [-2.,-3.,-4.,-5.,-6.,-7.,-8.,0.]
        manifest=JSON3.read(read(p,String))
        @test manifest.version == 2
        @test manifest.schema.first_iteration == 1
        @test length(manifest.blocks) == 3
        S._append_delayed_tridiagonal!(s,10.,0.)
        p2=S._close_coefficients!(s)
        @test p2 != p
        @test S.read_npy_zstd(p2) == [10. 0.]
        @test JSON3.read(read(p2,String)).schema.first_iteration == 9
        @test S._close_coefficients!(s) === nothing
        # An artifact from an older attempt must not make an empty new block
        # look as though it was written by the current solve.
        records=[Dict{String,Any}("next_tridiag_path"=>p2)]
        S._clear_unwritten_next_tridiag!(records,nothing)
        @test records[1]["next_tridiag_path"] === nothing
        legacy=joinpath(dir,"legacy_tridiag.npy.zstd")
        S.write_npy_zstd(legacy,data)
        @test S.read_npy_zstd(legacy) == data
        @test S.read_npy_zstd(S._coefficient_path(raw,"_tridiag",0)) == data
    end
end

@testset "solver outputs independent of the BLAS pool size" begin
    A,b,seed=recurrence_problem()
    previous=BLAS.get_num_threads()
    try
        mktempdir() do dir
            for name in (:pcg,:pcr,:minres,:lanczos_pcg,:fpcg1,:fpcg2)
                outputs=[]
                for nt in (1,3)
                    BLAS.set_num_threads(nt)
                    raw=joinpath(dir,"$(name)_$(nt).npy.zstd")
                    x=zeros(length(b))
                    prec=Preconditioner((z,r)->(copyto!(z,r);UInt64(0)),UInt64(0),UInt64(0))
                    result=getfield(S,name)(A,b,seed,6.,prec,Int64(8),raw,joinpath(dir,"$(name)_$(nt)");
                        convergence_tolerance=0.,_experimental_final_iterate=x,_diagnostic_block_rows=3)
                    push!(outputs,(result,x,S.read_npy_zstd(raw),S.read_npy_zstd(S._tridiag_block_path(raw,0))))
                end
                a,c=outputs
                @test a[1].total_fma == c[1].total_fma
                @test a[1].status == c[1].status
                for j in 2:4
                    @test reinterpret(UInt64,vec(a[j])) == reinterpret(UInt64,vec(c[j]))
                end
            end
        end
    finally
        BLAS.set_num_threads(previous)
    end
end

@testset "metadata identifies coefficient attempts and signed schema" begin
    A,b,seed=recurrence_problem()
    mktempdir() do dir
        raw=joinpath(dir,"raw_data.npy.zstd")
        prec=Preconditioner((z,r)->(copyto!(z,r);UInt64(0)),UInt64(0),UInt64(0))
        S.fpcg2(A,b,seed,6.,prec,Int64(4),raw,joinpath(dir,"conv"); convergence_tolerance=0.)
        meta=Dict{String,Any}("correction_restarts"=>0)
        S._diagnostic_storage_metadata!(meta,dir)
        @test meta["raw_data_schema"]["version"] == 2
        @test meta["raw_data_schema"]["alpha_semantics"] == "signed"
        @test length(meta["coefficient_storage"]) == 2
        @test all(a->isfile(joinpath(dir,a["manifest"])),meta["coefficient_storage"])
        # Starting a new raw attempt does not adopt prior coefficient files.
        S.DiagnosticBlockWriter(raw,5,S.FPCG_COLUMN_NAMES)
        S._diagnostic_storage_metadata!(meta,dir)
        @test isempty(meta["coefficient_storage"])
    end
end

function variable_prec()
    calls=Ref(0)
    Preconditioner((z,r)->begin
        calls[]+=1
        for j in eachindex(r)
            z[j]=r[j]/(1.0+0.2*sin(j+calls[]))
        end
        UInt64(0)
    end,UInt64(0),UInt64(0))
end

# Independent vector recurrence supplies both the scalar reference and vectors
# needed to check Az_k = sum_j h_jk r_j, including FPCG2's extra band.
function reference_recurrence(A,b,steps,two_direction)
    prec=variable_prec(); r=copy(b); z=similar(r); prec.LinearOperator(z,r)
    p=copy(z); Ap=A*p
    directions=Vector{Float64}[]; images=Vector{Float64}[]
    residuals=[copy(r)]; zs=Vector{Float64}[]
    scalars=[]; c1=0.; c2=0.
    for k in 1:steps
        dad=dot(p,Ap); pr=dot(p,r); alpha=pr/dad
        push!(scalars,(alpha,c1,c2,dot(r,z),pr,dad,norm(r),norm(z)))
        push!(zs,copy(z)); push!(directions,copy(p)); push!(images,copy(Ap))
        r=r-alpha*Ap; push!(residuals,copy(r))
        prec.LinearOperator(z,r)
        c1=-dot(z,images[end])/dot(directions[end],images[end])
        c2=two_direction && k>=2 && mod(k,3)!=0 ?
            -dot(z,images[end-1])/dot(directions[end-1],images[end-1]) : 0.
        p=z+c1*directions[end]
        k>=2 && (p+=c2*directions[end-1])
        Ap=A*p
    end
    return scalars,residuals,zs
end

@testset "flexible actual recurrence and formal reconstruction" begin
    A,b,seed=recurrence_problem()
    mktempdir() do root
        for (name,two) in ((:fpcg1,false),(:fpcg2,true))
            dir=joinpath(root,string(name)); mkpath(dir)
            raw=joinpath(dir,"raw.npy.zstd")
            result=getfield(S,name)(A,b,seed,6.,variable_prec(),Int64(8),raw,joinpath(dir,"conv");
                convergence_tolerance=0.,solver_mode=:bounded_continuation,_diagnostic_block_rows=3)
            @test result.iterations == 8
            actual=S.read_npy_zstd(S._coefficient_path(raw,"_recurrence",0))
            formal=S.read_npy_zstd(S._coefficient_path(raw,"_tridiag",0))
            scalars,residuals,zs=reference_recurrence(A,b,8,two)
            for k in 1:8
                @test actual[k,1:2] == [k,k]
                @test actual[k,3:10] ≈ collect(scalars[k]) rtol=2e-10 atol=1e-13
                represented=actual[k,13]*residuals[k]+actual[k,14]*residuals[k+1]
                k>=2 && (represented+=actual[k,12]*residuals[k-1])
                k>=3 && (represented+=actual[k,11]*residuals[k-2])
                @test A*zs[k] ≈ represented rtol=2e-10 atol=1e-12
                @test formal[k,1] == actual[k,13]
                if k>=2
                    @test formal[k-1,2] ≈ sqrt(abs(actual[k,4]))/actual[k-1,3]
                end
            end
            @test formal[end,2] == 0.0
            two && @test any(!iszero,actual[:,5])
            !two && @test all(iszero,actual[:,5])
            m=JSON3.read(read(S._tridiag_block_path(raw,0),String))
            @test !m.schema.ritz_guarantee
            @test !m.schema.fixed_operator_assumed
            @test length(m.blocks) == 3
            rawmeta=JSON3.read(read(S._diagnostic_manifest(raw),String))
            @test rawmeta.schema.version == 2
            @test rawmeta.schema.alpha_semantics == "signed"
        end
    end
end

@testset "fixed preconditioning recovers PCG coefficients" begin
    A,b,seed=recurrence_problem()
    mktempdir() do dir
        matrices=[]
        for name in (:pcg,:fpcg1,:fpcg2)
            raw=joinpath(dir,string(name)*".npy.zstd")
            prec=Preconditioner((z,r)->(copyto!(z,r);UInt64(0)),UInt64(0),UInt64(0))
            getfield(S,name)(A,b,seed,6.,prec,Int64(8),raw,joinpath(dir,string(name));
                convergence_tolerance=0.,_diagnostic_block_rows=3)
            push!(matrices,S.read_npy_zstd(S._tridiag_block_path(raw,0)))
        end
        @test matrices[1] ≈ matrices[2] rtol=1e-10 atol=1e-12
        @test matrices[1] ≈ matrices[3] rtol=1e-10 atol=1e-12
    end
end

@testset "signed step lengths and flagged formal absolute value" begin
    A,b,seed=recurrence_problem()
    mktempdir() do dir
        for name in (:pcg,:pcr,:fpcg1,:fpcg2)
            raw=joinpath(dir,string(name)*".npy.zstd")
            prec=Preconditioner((z,r)->(copyto!(z,r);UInt64(0)),UInt64(0),UInt64(0))
            getfield(S,name)(-A,-b,seed,6.,prec,Int64(2),raw,joinpath(dir,string(name));
                convergence_tolerance=0.)
            data = S.read_npy_zstd(raw)
            if name == :pcr
                @test isnan(data[1,8]) # weighted denominator is first computed in iteration 1
                @test data[2,8] < 0.0
            else
                @test data[1,8] < 0.0
            end
        end
        raw=joinpath(dir,"formal.npy.zstd")
        s=S.FlexibleCoefficientStreams(raw,3,1)
        S._record_flexible_coefficients!(s,1,2.,0.,0.,1.,1.,1.,1.,1.)
        S._record_flexible_coefficients!(s,2,3.,-4.,0.,1.,1.,1.,1.,1.)
        p,_=S._close_flexible_coefficients!(s;next_iteration=3)
        @test S.read_npy_zstd(p)[1,2] == 1.0
        @test S.read_npy_zstd(S._coefficient_path(raw,"_recurrence",0))[2,end] == 1.0
    end
end
