using Test, CAPPABenchmarkTools, LinearAlgebra, SparseArrays, Random
const TU = CAPPABenchmarkTools.ThreadedUtilities
const U = CAPPABenchmarkTools.Utilities

@testset "Solver reductions stay in the Julia pool" begin
    source = read(joinpath(@__DIR__,"..","src","Solvers.jl"),String)
    @test !occursin(r"\b(?<!_)(?:dot|norm|mul!|axpy!|axpby!)\(",source)
end

# Same arithmetic without the scheduling calls, to isolate handoff effects.
let source = read(joinpath(@__DIR__, "..", "src", "Utilities.jl"), String)
    @test count("_sleep_julia_workers_for_blas!()", source) == 4
    for name in ("estimate_eigrange", "_lanczos_quadrature", "_estimate_eigbounds")
        start = first(findfirst("function $name(", source))
        _, stop = Meta.parse(source, start)
        body = source[start:prevind(source, stop)]
        body = replace(body, "function $name(" => "function _handoff_reference$(name)(";
            count=1)
        body = replace(body, "_sleep_julia_workers_for_blas!()" => "nothing")
        Core.eval(U, Meta.parse(body))
    end
end

@testset "BLAS handoff preserves preparation arithmetic and wakes Julia workers" begin
    original = BLAS.get_num_threads()
    n = 192
    A = spdiagm(-1=>fill(-1.0,n-1),0=>collect(range(2.01,4.0;length=n)),1=>fill(-1.0,n-1))
    x = randn(MersenneTwister(37),n)
    b = A*x
    y = zeros(n)
    ws = TU.FusedReductionWorkspace()
    try
        for blas_threads in unique((1,Threads.nthreads()))
            BLAS.set_num_threads(blas_threads)
            for _ in 1:10
                TU.threaded_spmv!(y,A,x)
                reference = copy(y)
                norm_reference = TU.threaded_norm(y)
                @test TU._sleep_julia_workers_for_blas!() === nothing
                @test BLAS.get_num_threads() == blas_threads
                TU.threaded_spmv!(y,A,x)
                @test reinterpret(UInt64,y) == reinterpret(UInt64,reference)
                @test TU.threaded_norm(y) === norm_reference
                @test isfinite(TU.threaded_dot(x,y,ws))
            end
            TU._sleep_julia_workers!()
            @test (@allocated TU._sleep_julia_workers!()) == 0
            @test isequal(U.estimate_eigrange(A,20), U._handoff_referenceestimate_eigrange(A,20))
            @test isequal(U._lanczos_quadrature(A,b;max_iters=20),
                U._handoff_reference_lanczos_quadrature(A,b;max_iters=20))
            for target in (:upper,:both)
                kwargs = (;target,m=12,max_restarts=2,upper_tol=1e-30,lower_tol=1e-30)
                actual = U._estimate_eigbounds(A;kwargs...)
                @test actual[4] == 24
                @test isequal(actual,U._handoff_reference_estimate_eigbounds(A;kwargs...))
                @test BLAS.get_num_threads() == blas_threads
            end
        end
    finally
        BLAS.set_num_threads(original)
    end
end

@testset "Idle policies are observable, not silently changed at runtime" begin
    withenv("OPENBLAS_THREAD_TIMEOUT"=>"7", "OMP_WAIT_POLICY"=>"PASSIVE",
            "GOMP_SPINCOUNT"=>"123", "KMP_BLOCKTIME"=>"0") do
        snapshot = U.execution_provenance()
        for key in ("OPENBLAS_THREAD_TIMEOUT","OMP_WAIT_POLICY","GOMP_SPINCOUNT","KMP_BLOCKTIME")
            @test snapshot["thread_environment"][key] == ENV[key]
        end
        @test snapshot["blas_config"] == string(BLAS.get_config())
    end
end
