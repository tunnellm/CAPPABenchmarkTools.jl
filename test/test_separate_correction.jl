# Includes policy fixtures and existing artifact/accounting regressions.
include("test_residual_replacement.jl")

function capture_restart_rhs(make_prec)
    prec = make_prec()
    calls = Ref(0)
    rhs = Ref(Float64[])
    apply! = function(z, r)
        calls[] += 1
        calls[] == 3 && (rhs[] = copy(r))
        return prec.LinearOperator(z, r)
    end
    return Preconditioner(apply!, UInt64(0), UInt64(0)), rhs
end

@testset "correction exits retain the global candidate and charged work" begin
    A, b, seed, bound = replacement_test_problem()
    for name in (:pcg, :pcr, :fpcg1, :fpcg2, :lanczos_pcg, :minres)
        solver = getproperty(Solvers, name)
        make_prec = name in (:minres, :lanczos_pcg) ?
            tiny_preconditioned_residual_once :
            name == :pcr ? pcr_false_crossing_once : zero_recursive_residual_once
        mktempdir() do dir
            function run_case(label, iters; work_limit=0)
                x = zeros(length(b))
                raw = joinpath(dir, label * ".npy.zstd")
                result = solver(A, b, seed, bound, make_prec(), Int64(iters), raw,
                    joinpath(dir, label); convergence_tolerance=0.1,
                    correction_restart_normwise_tolerance=0.1, work_limit=work_limit,
                    _experimental_final_iterate=x)
                return result, x, read_npy_zstd(raw)
            end
            anchor, x0, d0 = run_case("anchor", 1)
            @test anchor.correction_restarts == 1
            @test anchor.status == Solvers.MAX_ITERS
            base = CAPPABenchmarkTools.Utilities.read_npy_zstd_1d_f64(
                only(anchor.correction_restart_records)["base_solution_path"])
            @test x0 == base
            trial, x1, d1 = run_case("one_correction_step", 2)
            @test trial.correction_restarts == 1
            @test d1[end, 2] ≈ norm(b-A*x1)/norm(b) rtol=1e-12
            limit = Int(sum(d1[end, 5:7]) - length(b)/2)
            @test limit > sum(d0[end, 5:7])
            limited, xl, dl = run_case("work_limit", 200; work_limit=limit)
            @test limited.status == WORK_LIMIT
            @test limited.iterations == 2
            @test limited.correction_restarts == 1
            @test xl == x1
            @test dl[end, 5:7] == d1[end, 5:7]
        end
    end
end

@testset "MINRES post-restart breakdown preserves prior costs" begin
    A, b, seed, bound = replacement_test_problem()
    calls = Ref(0)
    apply! = function(z, r)
        calls[] += 1
        z .= (calls[] == 2 ? 1e-32 : calls[] == 4 ? -1.0 : 1.0) .* r
        return UInt64(0)
    end
    mktempdir() do dir
        raw = joinpath(dir, "breakdown.npy.zstd")
        result = minres(A, b, seed, bound, Preconditioner(apply!, UInt64(0), UInt64(0)),
            Int64(20), raw, joinpath(dir, "conv"); convergence_tolerance=0.1,
            correction_restart_normwise_tolerance=0.1)
        @test result.status == BREAKDOWN
        @test result.correction_restarts == 1
        data = read_npy_zstd(raw)
        @test all(diff(data[:, 5]) .>= 0)
        @test all(diff(data[:, 7]) .>= 0)
        @test data[end, 5] == nnz(A)*(result.iterations+result.certification_true_residual_evals)
    end
end

@testset "separate correction equals a cold correction solve" begin
    A, b, seed, spectral_norm = replacement_test_problem()
    identity_prec() = Preconditioner((z, r) -> (copyto!(z, r); UInt64(0)), UInt64(0), UInt64(0))
    for name in (:pcg, :pcr, :fpcg1, :fpcg2, :lanczos_pcg, :minres), block_rows in (1, 500_000)
        solver = getproperty(Solvers, name)
        make_prec = name in (:minres, :lanczos_pcg) ?
            tiny_preconditioned_residual_once :
            name == :pcr ? pcr_false_crossing_once : zero_recursive_residual_once
        flags = (_diagnostic_block_rows=block_rows,)
        mktempdir() do dir
            prec, rhs = capture_restart_rhs(make_prec)
            x = zeros(length(b))
            raw = joinpath(dir, "global.npy.zstd")
            result = solver(A, b, seed, spectral_norm, prec, Int64(200), raw,
                joinpath(dir, "global"); convergence_tolerance=0.1,
                correction_restart_normwise_tolerance=0.1,
                _experimental_final_iterate=x, flags...)
            @test result.status == CONVERGED
            record = only(result.correction_restart_records)
            @test record["solution_accumulation"] == "separate_correction"
            @test record["candidate_assembly_fma_per_iteration"] == length(b)
            base = CAPPABenchmarkTools.Utilities.read_npy_zstd_1d_f64(record["base_solution_path"])
            steps = result.iterations - record["after_iteration"]
            @test steps > 0
            delta = zeros(length(b))
            # Disable stopping/restarting in the independent correction solve.
            # Its RHS is exactly the vector supplied to restart initialization.
            inner = solver(A, rhs[], A \ rhs[], spectral_norm, identity_prec(),
                Int64(steps), joinpath(dir, "inner.npy.zstd"), joinpath(dir, "inner");
                convergence_tolerance=0.0, correction_restart_normwise_tolerance=0.0,
                _experimental_final_iterate=delta, flags...)
            @test inner.correction_restarts == 0
            # Compare separately accumulated corrections at rounding-error scale.
            @test x ≈ base + delta rtol=32eps(Float64) atol=32eps(Float64)
            data = read_npy_zstd(raw)
            @test data[end, 2] ≈ norm(b - A*x)/norm(b) rtol=1e-12
            @test data[end, 4] ≈ norm(seed-x)/norm(seed) rtol=1e-12
            @test data[end, 2] <= 0.1
            @test all(isfinite, x)
        end
    end
end
