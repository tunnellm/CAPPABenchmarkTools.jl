using Test, CAPPABenchmarkTools, SparseArrays, JSON3
const U = CAPPABenchmarkTools.Utilities

@testset "concurrent spectral-cache misses estimate once" begin
    mktempdir() do dir
        script=joinpath(dir,"estimate.jl")
        write(script, """
            using CAPPABenchmarkTools, SparseArrays
            U=CAPPABenchmarkTools.Utilities
            U._cached_spectral_norm(ARGS[1], Dict("input"=>"shared")) do
                open(joinpath(ARGS[1],"calls"),"a") do io
                    println(io,"estimate")
                end
                sleep(1.0)
                spdiagm(0=>[-5.0,2.0,1.0])
            end
            """)
        cmd=`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(dirname(Base.active_project())) $script $dir`
        first=run(cmd;wait=false)
        second=run(cmd;wait=false)
        wait(first); wait(second)
        @test success(first) && success(second)
        @test readlines(joinpath(dir,"calls"))==["estimate"]
        @test U._cached_spectral_norm(()->error("cache miss"),dir,Dict("input"=>"shared"))["estimate"] ≈ 5.0
    end
end
