using Test, SparseArrays, LinearAlgebra, JSON3
using CAPPABenchmarkTools
using CAPPABenchmarkTools.PreconditionerTypes: Preconditioner
import CAPPABenchmarkTools.Solvers as S

# Independent ordinary CR on L^-1 A L^-T, in higher precision, rather than a
# second copy of the production preconditioned recurrence.
function transformed_cr(A, b, M, steps)
    L = cholesky(Symmetric(BigFloat.(M))).L
    C = L \ BigFloat.(A) / L'
    r = L \ BigFloat.(b)
    y = zeros(BigFloat, length(b))
    p = copy(r)
    q = C * p
    rho = dot(r, q)
    for k in 1:steps
        alpha = rho / dot(q, q)
        y += alpha * p
        r -= alpha * q
        Ar = C * r
        rho_next = dot(r, Ar)
        beta = rho_next / rho
        p = r + beta * p
        q = Ar + beta * q
        rho = rho_next
    end
    return Float64.(L' \ y)
end

function run_pcr_case(A, b, M, steps; kwargs...)
    calls = Ref(0)
    factor = cholesky(Symmetric(M))
    prec = Preconditioner((z, r) -> begin
        calls[] += 1
        copyto!(z, r)
        ldiv!(factor, z)
        UInt64(17)
    end, UInt64(0), UInt64(17))
    mktempdir() do dir
        x = zeros(length(b))
        path = joinpath(dir, "raw.npy.zstd")
        result = S.pcr(sparse(A), b, A \ b, opnorm(A), prec, Int64(steps),
            path, joinpath(dir, "conv"); _experimental_final_iterate=x, kwargs...)
        data = S.read_npy_zstd(path)
        columns = JSON3.read(read(replace(path, r"\.[^/]*$" => "_columns.json"), String))
        return (; result, x, data, columns, calls=calls[])
    end
end

@testset "PCR scalar preconditioner scaling regression" begin
    for m in (0.25, 1.0, 4.0, 16.0)
        run = run_pcr_case(ones(1,1), [1.0], fill(m,1,1), 1)
        @test run.result.status == S.CONVERGED
        @test run.x == [1.0]
        @test run.result.final_relative_residual == 0.0
        @test run.calls == 2
        @test run.data[end, S.PCR_COL_ALPHA] == m
        @test run.data[end, S.PCR_COL_AP_MINV_AP] == inv(m)^3
        @test run.data[end, S.PCR_COL_APAP] == inv(m)^2
        @test run.data[end, 5:7] == [3.0, 34.0, 8.0]
        @test run.result.total_fma == sum(run.data[end, 5:7])
        @test run.columns[S.PCR_COL_AP_MINV_AP] == "ApMinvAp"
        @test !haskey(run.result.solver_policy, "pcr_recurrence_version")
    end
    for (limit, expected_status) in ((44,S.WORK_LIMIT),(45,S.CONVERGED))
        run = run_pcr_case(ones(1,1),[1.0],fill(4.0,1,1),2; work_limit=Int64(limit))
        @test run.result.status == expected_status
        @test run.result.total_fma == 45
        @test run.x == [1.0]
    end
end

@testset "PCR agrees with transformed CR" begin
    n = 9
    A = Matrix(SymTridiagonal(collect(3.0:11.0), fill(-0.7,n-1)))
    b = A * sin.(collect(1.0:n))
    matrices = (Matrix{Float64}(I,n,n), Matrix(Diagonal(collect(1.0:n))),
        Matrix(SymTridiagonal(collect(2.0:10.0), fill(0.3,n-1))))
    for M in matrices, steps in (1,2,4,6), rows in (1,500_000)
        run = run_pcr_case(A,b,M,steps; convergence_tolerance=1e-30,
            correction_restart_normwise_tolerance=0.0, _diagnostic_block_rows=rows)
        @test run.result.status == S.MAX_ITERS
        @test run.x ≈ transformed_cr(A,b,M,steps) rtol=2e-12 atol=2e-13
        @test run.data[end,2] ≈ norm(b-A*run.x)/norm(b) rtol=1e-8 atol=1e-15
        @test run.calls == steps + 1
        @test run.data[end,5] == nnz(sparse(A))*(steps+1)
        @test run.data[end,6] == 17*(steps+1)
        @test run.data[end,7] == n*(7steps-2)
    end
    exact = run_pcr_case(A,b,A,5)
    @test exact.result.status == S.CONVERGED
    @test exact.result.iterations == 1
    @test exact.x ≈ A \ b
    for M in matrices
        complete = run_pcr_case(A,b,M,30)
        @test complete.result.status == S.CONVERGED
        @test norm(b-A*complete.x)/norm(b) <= 1e-10
    end
end

@testset "PCR invalid denominator preserves the candidate" begin
    for bad in (0.0, -1.0, NaN)
        calls = Ref(0)
        prec = Preconditioner((z,r) -> begin
            calls[] += 1
            z .= (calls[] == 1 ? 1.0 : bad) .* r
            UInt64(17)
        end, UInt64(0), UInt64(17))
        mktempdir() do dir
            x = zeros(1)
            path = joinpath(dir,"raw.npy.zstd")
            result = S.pcr(spdiagm(0=>[1.0]),[1.0],[1.0],1.0,prec,Int64(2),
                path,joinpath(dir,"conv"); _experimental_final_iterate=x)
            @test result.status == S.BREAKDOWN
            @test x == [0.0]
            @test result.final_relative_residual == 1.0
            @test result.correction_restarts == 0
            @test S.read_npy_zstd(path)[end,5:7] == [1.0,34.0,2.0]
        end
    end
end

@testset "PCR preconditioned correction matches a fresh correction solve" begin
    n = 12
    A = Matrix(SymTridiagonal(fill(3.0,n),fill(-1.0,n-1)))
    seed = sin.(collect(1.0:n))
    b = A*seed
    for M in (4.0*Matrix{Float64}(I,n,n),
              Matrix(SymTridiagonal(fill(4.0,n),fill(0.1,n-1))))
        factor = cholesky(Symmetric(M))
        calls = Ref(0)
        residual = Ref(Float64[])
        initial_z = Ref(Float64[])
        correction_rhs = Ref(Float64[])
        prec = Preconditioner((out,input) -> begin
            calls[] += 1
            copyto!(out,input)
            ldiv!(factor,out)
            if calls[] == 1
                residual[] = input
                initial_z[] = copy(out)
            elseif calls[] == 2
                # Deliberate native-residual corruption, without altering z or Ap.
                alpha = dot(initial_z[],input)/dot(input,out)
                residual[] .= alpha .* input
            elseif calls[] == 3
                correction_rhs[] = copy(input)
            end
            UInt64(17)
        end,UInt64(0),UInt64(17))
        mktempdir() do dir
            x = zeros(n)
            raw = joinpath(dir,"raw.npy.zstd")
            result = S.pcr(sparse(A),b,seed,opnorm(A),prec,Int64(100),raw,
                joinpath(dir,"conv"); convergence_tolerance=0.1,
                correction_restart_normwise_tolerance=0.1,_experimental_final_iterate=x)
            @test result.status == S.CONVERGED
            @test result.correction_restarts == 1
            record = only(result.correction_restart_records)
            base = CAPPABenchmarkTools.Utilities.read_npy_zstd_1d_f64(record["base_solution_path"])
            steps = result.iterations-record["after_iteration"]
            @test steps > 0
            @test correction_rhs[] ≈ b-A*base rtol=1e-14 atol=1e-14
            @test x ≈ base+transformed_cr(A,correction_rhs[],M,steps) rtol=2e-12 atol=2e-13
            @test calls[] == 2+result.iterations
            @test norm(b-A*x)/norm(b) <= 0.1
            data = S.read_npy_zstd(raw)
            @test data[end,5] == nnz(sparse(A))*(2+result.iterations+result.certification_true_residual_evals)
            @test data[end,6] == 17*calls[]
            @test data[end,7] == n*(7result.iterations-3+steps+3result.certification_true_residual_evals)
        end
    end
end
