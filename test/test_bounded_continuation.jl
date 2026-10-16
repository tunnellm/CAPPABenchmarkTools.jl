using Test, SparseArrays, LinearAlgebra, JSON3
using CAPPABenchmarkTools
using CAPPABenchmarkTools.PreconditionerTypes: Preconditioner
import CAPPABenchmarkTools.Solvers as S

include("pcr_test_helpers.jl")

function survey_problem()
    n = 24
    A = spdiagm(-1=>fill(-1.0,n-1), 0=>fill(3.0,n), 1=>fill(-1.0,n-1))
    seed = sin.(collect(1.0:n))
    A, A*seed, seed, Float64(eigmax(Matrix(A)))
end

identity_prec() = Preconditioner((z,r)->(copyto!(z,r); UInt64(0)), UInt64(0), UInt64(0))

function survey_fault(scalar)
    calls = Ref(0)
    Preconditioner((z,r)->begin
        calls[] += 1
        # Explicit-vector fault injection changes r; the scalar fixture changes
        # only the preconditioned norm. Both create a false native crossing.
        if calls[] == 2 && !scalar
            r .*= 1e-20
        end
        z .= r
        calls[] == 2 && scalar && (z .*= 1e-32)
        UInt64(0)
    end, UInt64(0), UInt64(0))
end

@testset "bounded continuation in all six kernels" begin
    A, b, seed, normA = survey_problem()
    for solver in (S.pcg, S.minres, S.lanczos_pcg, S.pcr, S.fpcg1, S.fpcg2)
        @testset "$(nameof(solver))" begin
            mktempdir() do dir
                events = Dict{String,Any}[]
                scalar = solver in (S.minres, S.lanczos_pcg)
                prec = solver === S.pcr ? pcr_false_crossing_once() : survey_fault(scalar)
                result = solver(A,b,seed,normA,prec,Int64(100),
                    joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv");
                    solver_mode=:bounded_continuation,
                    progress_callback=e->push!(events,copy(e)))
                @test result.status == S.CONTINUATION_LIMIT
                @test result.correction_restarts == result.residual_replacements == 0
                @test isempty(result.correction_restart_records)
                @test result.native_candidate_iter !== nothing
                data = S.read_npy_zstd(joinpath(dir,"raw.npy.zstd"))
                crossing = result.native_candidate_iter + 1
                crossing_work = sum(data[crossing,5:7])
                cap = 2 * crossing_work
                @test result.solver_policy["native_candidate_total_fma"] == crossing_work
                @test result.solver_policy["continuation_work_limit"] == cap
                @test result.total_fma >= cap
                @test sum(data[end-1,5:7]) < cap
                @test result.certification_check_iters == collect(result.native_candidate_iter:result.iterations)
                @test result.physical_true_residual_evals == result.iterations
                @test isfile(joinpath(dir,"conv_relative_residual.csv"))
                @test isfile(joinpath(dir,"conv_relative_normwise_2.csv"))
                @test isfile(joinpath(dir,"conv_certification.csv"))
                numerical = filter(e->get(e,"progress_kind",nothing)=="numerical",events)
                @test any(e->get(e,"policy_event",nothing)=="native_candidate",numerical)
                @test last(numerical)["status"] == "CONTINUATION_LIMIT"
                @test last(numerical)["continuation_work_limit"] == cap

                pass = solver(A,b,seed,normA,identity_prec(),Int64(100),
                    joinpath(dir,"pass.npy.zstd"),joinpath(dir,"pass");solver_mode=:bounded_continuation)
                @test pass.status == S.CONVERGED
                @test pass.correction_restarts == 0

                no_crossing = solver(A,b,seed,normA,identity_prec(),Int64(2),
                    joinpath(dir,"no_cross.npy.zstd"),joinpath(dir,"no_cross");
                    solver_mode=:bounded_continuation,convergence_tolerance=1e-30)
                @test no_crossing.status == S.MAX_ITERS
                @test no_crossing.solver_policy["continuation_work_limit"] === nothing
                limited = solver(A,b,seed,normA,identity_prec(),Int64(100),
                    joinpath(dir,"limit.npy.zstd"),joinpath(dir,"limit");
                    solver_mode=:bounded_continuation,work_limit=Int64(1))
                @test limited.status == S.WORK_LIMIT

                default = solver(A,b,seed,normA,identity_prec(),Int64(100),
                    joinpath(dir,"default.npy.zstd"),joinpath(dir,"default"))
                explicit_default = solver(A,b,seed,normA,identity_prec(),Int64(100),
                    joinpath(dir,"explicit.npy.zstd"),joinpath(dir,"explicit");solver_mode=:certified)
                @test isequal(S.read_npy_zstd(joinpath(dir,"default.npy.zstd")),
                    S.read_npy_zstd(joinpath(dir,"explicit.npy.zstd")))
                @test default.total_fma == explicit_default.total_fma
            end
        end
    end
end

@testset "survey budget precedence and configuration identity" begin
    policy = S.SolverPolicyState(:bounded_continuation,2.0,1000)
    S._observe_native_candidate!(policy,true,4,100.0)
    S._observe_native_candidate!(policy,true,5,150.0)
    @test policy.continuation_work_limit == 200
    @test S._policy_stop_status(policy,200.0,true) == S.CONVERGED
    @test S._policy_stop_status(policy,200.0,false) == S.CONTINUATION_LIMIT
    @test S._policy_stop_status(policy,201.0,true) == S.CONTINUATION_LIMIT
    @test S._policy_stop_status(policy,1001.0,true) == S.CONTINUATION_LIMIT
    policy.work_limit = 150.0
    @test S._policy_stop_status(policy,201.0,true) == S.WORK_LIMIT
    @test_throws ArgumentError S.SolverPolicyState(:typo,2.0,-1)
    @test_throws ArgumentError S.SolverPolicyState(:bounded_continuation,1.0,-1)
    @test_throws ArgumentError S.SolverPolicyState(:bounded_continuation,Inf,-1)
    cfg = Dict{String,Any}("alpha"=>1)
    hash = CAPPABenchmarkTools.TestWrapper._config_hash
    @test hash(cfg) == hash(S._solver_policy_config(cfg,:certified,2.0))
    @test hash(cfg) != hash(S._solver_policy_config(cfg,:bounded_continuation,2.0))
    @test hash(S._solver_policy_config(cfg,:bounded_continuation,2.0)) !=
          hash(S._solver_policy_config(cfg,:bounded_continuation,3.0))
end

@testset "survey tolerance simulation respects continuation budget" begin
    data = zeros(3,S.PCG_NUM_COLS)
    data[:,1] .= [1,2,3]
    data[:,2] .= [1e-9,1e-9,1e-11]
    data[:,3] .= 1e-12
    data[:,5] .= [100,200,300]
    data[:,9] .= 1e-12
    sim = S._simulate_certification_policy(data,9,1.0,1e-10,:scalar,1e-10;
        solver_mode=:bounded_continuation,actual_check_iters=[1,2,3])
    @test sim.status == "continuation_work_limit"
    @test sim.divergence_iter == 2
    @test sim.row === nothing
    sim = S._simulate_certification_policy(data,9,1.0,1e-10,:scalar,1e-10;
        solver_mode=:bounded_continuation,continuation_work_factor=4.0,actual_check_iters=[1,2,3])
    @test sim.status == "certified_from_raw"
    @test sim.iter == 3
end
