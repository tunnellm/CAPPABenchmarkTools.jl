using Test, JSON3, SparseArrays, LinearAlgebra, Random
using CAPPABenchmarkTools
using CAPPABenchmarkTools.PreconditionerTypes: Preconditioner
import CAPPABenchmarkTools.Solvers as S

include("pcr_test_helpers.jl")

samebits(a,b) = size(a) == size(b) && reinterpret(UInt64,vec(a)) == reinterpret(UInt64,vec(b))

function store_rows(path, expected; block_rows=3)
    writer = S.DiagnosticBlockWriter(path,size(expected,1),string.(1:size(expected,2)); block_rows)
    buffer = writer.buffer
    for row in axes(expected,1)
        S._prepare_diagnostic_row!(writer,row,size(expected,1))
        writer[row,:] = expected[row,:]
        @test writer.buffer === buffer
    end
    return S._finish_diagnostic_data!(writer,size(expected,1))
end

@testset "v1/v2 diagnostic dispatch and exact block boundaries" begin
    mktempdir() do dir
        for rows in (1,3,4,6,7)
            expected = reshape(collect(1.0:rows*3),rows,3)
            expected[1,1] = -0.0
            expected[end,end] = reinterpret(Float64,0x7ff8000000000042)
            path = joinpath(dir,"data_$rows.npy.zstd")
            S.write_npy_zstd(path,expected)
            @test samebits(S.read_npy_zstd(path),expected)
            lazy = store_rows(path,expected)
            manifest = S._diagnostic_manifest(path)
            meta = JSON3.read(read(manifest,String))
            @test meta.version == 2
            @test meta.finalized && meta.rows == rows
            @test length(meta.blocks) == cld(rows,3)
            @test lazy isa S.DiagnosticBlockReader
            @test samebits(S.read_npy_zstd(path),expected)
            @test samebits(S.read_npy_zstd(dirname(manifest)),expected)
            @test samebits(S.read_npy_zstd(manifest),expected)
            @test isequal(lazy[rows,2],expected[rows,2])
            @test isequal(lazy[1,1],expected[1,1])
            @test size(lazy.cached,1) <= 3
            # A reader snapshot keeps its immutable files across a retry.
            original_paths = copy(lazy.paths)
            newer = ones(rows,3)
            store_rows(path,newer)
            @test all(isfile,original_paths)
            @test samebits(S.read_npy_zstd(path),newer)
            @test isequal(lazy[rows,3],expected[rows,3])
            @test isequal(lazy[1,1],expected[1,1])
        end
        legacy = joinpath(dir,"legacy"); mkpath(legacy)
        expected = reshape(collect(1.0:12.0),4,3)
        S.write_npy_zstd(joinpath(legacy,"single.npy.zstd"),expected)
        write(joinpath(legacy,"manifest.json"),JSON3.write(Dict("version"=>1,"file"=>"single.npy.zstd")))
        @test samebits(S.read_npy_zstd(legacy),expected)
    end
end

@testset "interrupted stream exposes only committed prefix" begin
    mktempdir() do dir
        path = joinpath(dir,"raw.npy.zstd")
        writer = S.DiagnosticBlockWriter(path,10,["iteration","value"]; block_rows=3)
        @test size(S.open_diagnostic_data(path)) == (0,2)
        @test_throws ErrorException S.open_diagnostic_data(path; require_finalized=true)
        for row in 1:4
            S._prepare_diagnostic_row!(writer,row,10)
            writer[row,:] = [row-1.0, row*2.0]
        end
        @test writer.committed == 3
        @test writer[3,2] == 6.0 # prior row remains available to progress reporting
        S.write_npy_zstd(joinpath(writer.attempt,"orphan.npy.zstd"),zeros(10,2))
        @test S.read_npy_zstd(path) == [0.0 2.0; 1.0 4.0; 2.0 6.0]
        # Early breakdown may leave the newly reserved row uncommitted.
        S._finish_diagnostic_data!(writer,3)
        @test size(S.open_diagnostic_data(path; require_finalized=true)) == (3,2)
    end
end

@testset "production-sized buffer is fixed and reused" begin
    mktempdir() do dir
        path=joinpath(dir,"raw_data.npy.zstd")
        writer=S.DiagnosticBlockWriter(path,10^10,string.(1:18))
        buffer=writer.buffer
        @test size(buffer) == (500_000,18)
        @test sizeof(buffer) == 72_000_000
        writer[500_000,18] = -0.0
        S._prepare_diagnostic_row!(writer,500_001,10^10)
        @test writer.buffer === buffer
        @test writer.committed == 500_000
        @test signbit(writer[500_000,18])
        writer[500_001,1] = 500_000.0
        result=S._finish_diagnostic_data!(writer,500_001)
        @test size(result) == (500_001,18)
        @test result[500_001,1] == 500_000.0
        @test signbit(result[500_000,18])
        meta=Dict{String,Any}()
        S._diagnostic_storage_metadata!(meta,dir)
        @test meta["raw_data_storage"]["version"] == 2
        @test meta["raw_data_storage"]["manifest"] == "raw_data/manifest.json"
        @test meta["raw_data_storage"]["attempt"] == JSON3.read(read(S._diagnostic_manifest(path),String))["attempt"]
    end
end

@testset "corrupt diagnostic manifests and blocks fail explicitly" begin
    mktempdir() do dir
        path = joinpath(dir,"raw.npy.zstd")
        store_rows(path,ones(7,2))
        manifest = S._diagnostic_manifest(path)
        text = read(manifest,String)
        for mutate in (
            m->(m["version"]=99),
            m->(m["rows"]=8),
            m->(m["ncols"]=3),
            m->(m["blocks"][2]["first_row"]=3),
            m->(m["blocks"][1]["file"]="../escape.npy.zstd"),
            m->(m["blocks"][1]["file"]="missing.npy.zstd"),
            m->(m["blocks"][2]["file"]=m["blocks"][1]["file"]),
        )
            meta = JSON3.read(text,Dict{String,Any}); mutate(meta)
            write(manifest,JSON3.write(meta))
            @test_throws ErrorException S.open_diagnostic_data(path)
        end
        write(manifest,text)
        reader = S.open_diagnostic_data(path)
        S.write_npy_zstd(reader.paths[1],ones(2,2))
        @test_throws ErrorException reader[1,1]
        bytes = read(reader.paths[2])
        write(reader.paths[2],bytes[1:div(length(bytes),2)])
        @test_throws Exception reader[4,1]
    end
end

@testset "six solvers: block boundaries preserve all numerical outputs" begin
    mktempdir() do dir
        n=40
        A=spdiagm(-1=>fill(-1.0,n-1),0=>fill(2.01,n),1=>fill(-1.0,n-1))
        seed=randn(MersenneTwister(13),n); b=A*seed
        for solver in (S.pcg,S.pcr,S.fpcg1,S.fpcg2,S.minres,S.lanczos_pcg)
            outputs=[]
            for rows in (500_000,3)
                out=joinpath(dir,string(nameof(solver)),string(rows)); mkpath(out)
                prec=Preconditioner((z,r)->(copyto!(z,r); UInt64(0)),UInt64(0),UInt64(0))
                result=solver(A,b,seed,4.01,prec,Int64(20),joinpath(out,"raw.npy.zstd"),joinpath(out,"conv");
                    convergence_tolerance=0.0,correction_restart_normwise_tolerance=0.0,
                    _diagnostic_block_rows=rows)
                push!(outputs,(out,result))
            end
            (a,ra),(bpath,rb)=outputs
            @test ra.status == rb.status == S.MAX_ITERS
            @test ra.iterations == rb.iterations == 20
            @test ra.total_fma == rb.total_fma
            @test samebits(S.read_npy_zstd(joinpath(a,"raw.npy.zstd")),S.read_npy_zstd(joinpath(bpath,"raw.npy.zstd")))
            for file in readdir(a)
                isfile(joinpath(a,file)) || continue
                @test read(joinpath(a,file)) == read(joinpath(bpath,file))
            end
            manifest=JSON3.read(read(joinpath(bpath,"raw","manifest.json"),String))
            @test length(manifest.blocks) == 7
        end
    end
end

@testset "flushes preserve correction setup costs and continuation limits" begin
    mktempdir() do dir
        n=12
        A=spdiagm(-1=>fill(-1.0,n-1),0=>fill(3.0,n),1=>fill(-1.0,n-1))
        seed=sin.(collect(1.0:n)); b=A*seed; normA=Float64(eigmax(Matrix(A)))
        for solver in (S.pcg,S.pcr,S.fpcg1,S.fpcg2,S.minres,S.lanczos_pcg), survey in (false,true)
            outputs=[]
            for rows in (500_000,1)
                out=joinpath(dir,string(nameof(solver)),string(survey),string(rows)); mkpath(out)
                calls=Ref(0); scalar=solver in (S.minres,S.lanczos_pcg)
                prec=Preconditioner((z,r)->begin
                    calls[]+=1
                    calls[] == 2 && survey && !scalar && (r .*= 1e-20)
                    copyto!(z,r)
                    if calls[] == 2
                        scalar && (z .*= 1e-32)
                        !scalar && !survey && fill!(r,0.0) # explicit recurrence fault injection
                    end
                    UInt64(0)
                end,UInt64(0),UInt64(0))
                solver === S.pcr && (prec = pcr_false_crossing_once())
                result=solver(A,b,seed,normA,prec,Int64(200),joinpath(out,"raw.npy.zstd"),joinpath(out,"conv");
                    convergence_tolerance=survey ? 1e-10 : 1e-1,
                    correction_restart_normwise_tolerance=1e-1,
                    solver_mode=survey ? :bounded_continuation : :certified,
                    _diagnostic_block_rows=rows)
                push!(outputs,(out,result))
            end
            (a,ra),(bpath,rb)=outputs
            @test ra.status == rb.status == (survey ? S.CONTINUATION_LIMIT : S.CONVERGED)
            @test ra.correction_restarts == rb.correction_restarts == (survey ? 0 : 1)
            @test ra.total_fma == rb.total_fma
            @test ra.certification_check_iters == rb.certification_check_iters
            @test samebits(S.read_npy_zstd(joinpath(a,"raw.npy.zstd")),S.read_npy_zstd(joinpath(bpath,"raw.npy.zstd")))
            for file in readdir(a)
                isfile(joinpath(a,file)) || continue
                @test read(joinpath(a,file)) == read(joinpath(bpath,file))
            end
        end
    end
end

# Optional comparison against artifacts produced before this storage change.
if length(ARGS) == 2
    @testset "v1 baseline versus v2 solver artifacts" begin
        before,after=ARGS
        @test JSON3.read(read(joinpath(before,"summary.json"),String)) == JSON3.read(read(joinpath(after,"summary.json"),String))
        for (dir,_,files) in walkdir(before), file in files
            file == "summary.json" && continue
            old=joinpath(dir,file); new=joinpath(after,relpath(old,before))
            if file == "raw.npy.zstd"
                @test samebits(S.read_npy_zstd(old),S.read_npy_zstd(new))
            else
                @test read(old) == read(new)
            end
        end
    end
end
