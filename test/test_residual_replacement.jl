using Test
using SparseArrays
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using CAPPABenchmarkTools
using CAPPABenchmarkTools.PreconditionerTypes: Preconditioner
import CAPPABenchmarkTools.Solvers
using CAPPABenchmarkTools.Solvers:
    pcg, lanczos_pcg, minres, pcr, fpcg1, fpcg2, read_npy_zstd,
    CONVERGED, WORK_LIMIT, BREAKDOWN, STAGNATED

include("pcr_test_helpers.jl")

function replacement_test_problem(n::Int = 12)
    A = spdiagm(
        -1 => fill(-1.0, n - 1),
         0 => fill( 3.0, n),
         1 => fill(-1.0, n - 1),
    )
    seed = sin.(collect(1.0:n))
    b = A * seed
    spectral_norm = Float64(eigmax(Matrix(A)))
    return A, b, seed, spectral_norm
end

function zero_recursive_residual_once()
    calls = Ref(0)
    function apply!(z::Vector{Float64}, r::Vector{Float64})::UInt64
        calls[] += 1
        z .= r
        # Fault injection: mutate the input residual once so the method-native
        # recursive residual reports convergence before the true residual does.
        if calls[] == 2
            r .= 0.0
        end
        return UInt64(0)
    end
    return Preconditioner(apply!, UInt64(0), UInt64(0))
end

function tiny_preconditioned_residual_once()
    calls = Ref(0)
    function apply!(z::Vector{Float64}, r::Vector{Float64})::UInt64
        calls[] += 1
        if calls[] == 2
            z .= 1.0e-32 .* r
        else
            z .= r
        end
        return UInt64(0)
    end
    return Preconditioner(apply!, UInt64(0), UInt64(0))
end

function tiny_then_indefinite_preconditioner()
    calls = Ref(0)
    function apply!(z::Vector{Float64}, r::Vector{Float64})::UInt64
        calls[] += 1
        if calls[] == 2
            z .= 1.0e-32 .* r
        elseif calls[] == 3
            z .= -r
        else
            z .= r
        end
        return UInt64(0)
    end
    return Preconditioner(apply!, UInt64(0), UInt64(0))
end

function certification_sim_data(rows)
    data = fill(NaN, length(rows), Solvers.PCG_COL_NORM_R_TRUE)
    for (idx, row) in enumerate(rows)
        data[idx, Solvers.PCG_COL_ITER] = Float64(row.iter)
        data[idx, Solvers.PCG_COL_RELATIVE_RESIDUAL] = row.true_relres
        data[idx, Solvers.PCG_COL_RELATIVE_NORMWISE_2] = row.normwise
        data[idx, Solvers.PCG_COL_CUM_BASE_FMA] = row.base
        data[idx, Solvers.PCG_COL_CUM_PREC_FMA] = row.prec
        data[idx, Solvers.PCG_COL_CUM_VECTOR_FMA] = row.vector
        data[idx, Solvers.PCG_COL_NORM_R] = row.native_norm
        data[idx, Solvers.PCG_COL_NORM_R_TRUE] = 1.0
    end
    return data
end

function package_simulation(data; policy::Symbol, tol::Float64 = 1.0e-10,
                            restart_tol::Float64 =
                                Solvers.correction_restart_normwise_tolerance_default)
    return Solvers._simulate_certification_policy(
        data,
        Solvers.PCG_COL_NORM_R,
        1.0,
        tol,
        policy,
        restart_tol,
    )
end

@testset "certification policy simulation branches" begin
    @test Solvers.correction_restart_normwise_tolerance_default == sqrt(eps(Float64))

    cases = [
        (
            name = "direct certification",
            policy = :scalar,
            rows = [(iter = 1, true_relres = 5.0e-11, normwise = 5.0e-11,
                     native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "certified_from_raw",
            iter = 1,
            checks = 1,
            false_native = 0,
            divergence_iter = nothing,
        ),
        (
            name = "explicit failed certification without restart trigger remains pending",
            policy = :explicit,
            rows = [(iter = 1, true_relres = 1.0e-9, normwise = 1.0e-9,
                     native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "not_certified_in_raw_after_native_candidate",
            iter = nothing,
            checks = 1,
            false_native = 1,
            divergence_iter = nothing,
        ),
        (
            name = "explicit failed certification diverges to correction restart",
            policy = :explicit,
            rows = [(iter = 1, true_relres = 1.0e-9, normwise = 1.0e-14,
                     native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "trajectory_diverged_correction_restart",
            iter = nothing,
            checks = 1,
            false_native = 1,
            divergence_iter = 1,
        ),
        (
            name = "explicit restart uses convergence tolerance by default",
            policy = :explicit,
            rows = [(iter = 1, true_relres = 1.0e-9, normwise = 5.0e-11,
                     native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "trajectory_diverged_correction_restart",
            iter = nothing,
            checks = 1,
            false_native = 1,
            divergence_iter = 1,
        ),
        (
            name = "scalar failed certification diverges to correction restart",
            policy = :scalar,
            rows = [(iter = 1, true_relres = 1.0e-9, normwise = 1.0e-14,
                     native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "trajectory_diverged_correction_restart",
            iter = nothing,
            checks = 1,
            false_native = 1,
            divergence_iter = 1,
        ),
        (
            name = "scalar pending certification later succeeds",
            policy = :scalar,
            rows = [
                (iter = 1, true_relres = 1.0e-9, normwise = 1.0e-9,
                 native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0),
                (iter = 2, true_relres = 5.0e-11, normwise = 5.0e-11,
                 native_norm = 1.0e-11, base = 200.0, prec = 20.0, vector = 40.0),
            ],
            status = "certified_from_raw",
            iter = 2,
            checks = 2,
            false_native = 1,
            divergence_iter = nothing,
        ),
        (
            name = "scalar pending audit later triggers correction restart",
            policy = :scalar,
            rows = [
                (iter = 1, true_relres = 1.0e-9, normwise = 1.0e-9,
                 native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0),
                (iter = 2, true_relres = 1.0e-9, normwise = 1.0e-14,
                 native_norm = 2.0e-11, base = 200.0, prec = 20.0, vector = 40.0),
            ],
            status = "trajectory_diverged_correction_restart",
            iter = nothing,
            checks = 2,
            false_native = 1,
            divergence_iter = 2,
        ),
        (
            name = "scalar native mismatch continues after one decade",
            policy = :scalar,
            rows = [
                (iter = 1, true_relres = 1.0e-9, normwise = 1.0e-9,
                 native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0),
                (iter = 2, true_relres = 1.0e-9, normwise = 1.0e-9,
                 native_norm = 9.0e-12, base = 200.0, prec = 20.0, vector = 40.0),
            ],
            status = "not_certified_in_raw_after_native_candidate",
            iter = nothing,
            checks = 2,
            false_native = 1,
            divergence_iter = nothing,
        ),
        (
            name = "native candidate never certifies",
            policy = :scalar,
            rows = [(iter = 1, true_relres = 1.0e-9, normwise = 1.0e-9,
                     native_norm = 5.0e-11, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "not_certified_in_raw_after_native_candidate",
            iter = nothing,
            checks = 1,
            false_native = 1,
            divergence_iter = nothing,
        ),
        (
            name = "native threshold is never reached",
            policy = :scalar,
            rows = [(iter = 1, true_relres = 1.0e-9, normwise = 1.0e-9,
                     native_norm = 1.0e-9, base = 100.0, prec = 10.0, vector = 20.0)],
            status = "native_not_reached_in_raw",
            iter = nothing,
            checks = 0,
            false_native = 0,
            divergence_iter = nothing,
        ),
    ]

    for case in cases
        data = certification_sim_data(case.rows)
        package_result = package_simulation(data; policy = case.policy)
        @test package_result.status == case.status
        @test package_result.iter == case.iter
        @test package_result.certification_checks == case.checks
        @test package_result.false_native_crossings == case.false_native
        @test package_result.divergence_iter == case.divergence_iter
    end
end

@testset "simulated certification cost rewrites recorded charge" begin
    data = certification_sim_data([
        (iter = 1, true_relres = 5.0e-11, normwise = 5.0e-11,
         native_norm = 5.0e-11, base = 300.0, prec = 10.0, vector = 90.0),
    ])
    package_vals = Solvers._row_values_with_simulated_certification_cost(
        data,
        1,
        Solvers.PCG_COL_NORM_R,
        1.0,
        10,
        100,
        2,
        [1],
    )
    @test package_vals[5] == 400.0
    @test package_vals[7] == 120.0
    @test package_vals[8] == 530.0
end

@testset "explicit false certification starts correction restart" begin
    A, b, seed, spectral_norm = replacement_test_problem()

    cases = [
        (:pcg, pcg, zero_recursive_residual_once),
        (:pcr, pcr, pcr_false_crossing_once),
        (:fpcg1, fpcg1, zero_recursive_residual_once),
        (:fpcg2, fpcg2, zero_recursive_residual_once),
    ]

    mktempdir() do dir
        for (name, solver, make_prec) in cases
            raw = joinpath(dir, string(name, "_raw.npy.zstd"))
            conv = joinpath(dir, string(name, "_conv"))
            result = solver(
                A, b, seed, spectral_norm, make_prec(), Int64(200), raw, conv;
                convergence_tolerance = 1.0e-1,
                correction_restart_normwise_tolerance = 1.0e-1,
            )

            @test result.status == CONVERGED
            @test result.residual_replacements == 0
            @test isempty(result.residual_replacement_records)
            @test result.correction_restarts == 1
            @test length(result.correction_restart_records) == 1
            @test result.certification_true_residual_evals >= 2
            @test length(result.certification_records) >= 2
            @test first(result.certification_records)["action"] == "correction_restart"
            @test last(result.certification_records)["action"] == "accept_convergence"
            @test result.final_relative_residual <= 1.0e-1
            @test isfile(conv * "_certification.csv")

            data = read_npy_zstd(raw)
            expected_base_units = result.iterations + result.certification_true_residual_evals
            if name in (:pcr, :fpcg1, :fpcg2)
                expected_base_units += 1 + result.correction_restarts
                @test data[1, 5] == nnz(A)
            else
                @test data[1, 5] == 0.0
            end
            @test data[end, 5] == expected_base_units * nnz(A)

            record = only(result.correction_restart_records)
            @test record["restart_index"] == 1
            @test record["kind"] == "global_correction_restart"
            @test record["after_iteration"] >= 1
            @test isfile(record["base_solution_path"])
            @test isfile(record["closed_tridiag_path"])
            @test isfile(record["next_tridiag_path"])
            @test record["closed_tridiag_path"] != record["next_tridiag_path"]
        end
    end
end

@testset "scalar native mismatch does not imply stagnation" begin
    A, b, seed, spectral_norm = replacement_test_problem()

    cases = [
        (:lanczos_pcg, lanczos_pcg),
        (:minres, minres),
    ]

    mktempdir() do dir
        for (name, solver) in cases
            raw = joinpath(dir, string(name, "_raw.npy.zstd"))
            conv = joinpath(dir, string(name, "_conv"))
            result = solver(
                A, b, seed, spectral_norm,
                tiny_preconditioned_residual_once(),
                Int64(200), raw, conv;
                convergence_tolerance = 1.0e-10,
            )

            @test result.residual_replacements == 0
            @test isempty(result.residual_replacement_records)
            @test result.status != STAGNATED
            @test result.correction_restarts == 0
            @test isempty(result.correction_restart_records)
            @test isfile(conv * "_certification.csv")
            @test result.certification_true_residual_evals >= 1
            @test !isempty(result.certification_check_iters)
            @test result.certification_check_iters ==
                  result.true_residual_used_for_decision_iters
            @test result.false_native_crossings >= 1
            @test result.native_candidate_iter == first(result.certification_check_iters)
            @test result.native_candidate_true_relres !== nothing
            @test result.native_candidate_normwise !== nothing
            @test !isempty(result.certification_records)

            first_record = first(result.certification_records)
            @test first_record["event"] == "native_candidate"
            @test first_record["true_residual_role"] == "certification"
            @test first_record["certification_charged"] == true
            @test first_record["action"] == "continue_recurrence"
            @test result.iterations > first_record["iter"]
        end
    end
end

@testset "scalar correction restart solves global residual" begin
    A, b, seed, spectral_norm = replacement_test_problem()

    cases = [
        (:lanczos_pcg, lanczos_pcg),
        (:minres, minres),
    ]

    mktempdir() do dir
        for (name, solver) in cases
            raw = joinpath(dir, string(name, "_raw.npy.zstd"))
            conv = joinpath(dir, string(name, "_conv"))
            result = solver(
                A, b, seed, spectral_norm,
                tiny_preconditioned_residual_once(),
                Int64(200), raw, conv;
                convergence_tolerance = 1.0e-1,
                correction_restart_normwise_tolerance = 1.0e-1,
            )

            @test result.status == CONVERGED
            @test result.residual_replacements == 0
            @test isempty(result.residual_replacement_records)
            @test result.correction_restarts == 1
            @test length(result.correction_restart_records) == 1
            @test result.false_native_crossings == 1
            @test result.certification_true_residual_evals >= 2
            @test result.final_relative_residual <= 1.0e-1

            first_record = first(result.certification_records)
            @test first_record["event"] == "native_candidate"
            @test first_record["action"] == "correction_restart"
            @test first_record["certified"] == false
            @test last(result.certification_records)["action"] == "accept_convergence"

            restart_record = only(result.correction_restart_records)
            @test restart_record["kind"] == "global_correction_restart"
            @test restart_record["restart_index"] == 1
            @test restart_record["after_iteration"] == first(result.certification_check_iters)
            @test isfile(restart_record["base_solution_path"])
            @test isfile(restart_record["closed_tridiag_path"])
            @test isfile(restart_record["next_tridiag_path"])
            @test restart_record["closed_tridiag_path"] != restart_record["next_tridiag_path"]

            data = read_npy_zstd(raw)
            @test data[1, end] == 0.0
            @test data[first(result.certification_check_iters) + 1, end] == 0.0
            @test data[end, end] == 1.0
        end
    end
end

@testset "scalar correction restart breakdown is not recorded as successful restart" begin
    A, b, seed, spectral_norm = replacement_test_problem()

    cases = [
        (:lanczos_pcg, lanczos_pcg),
        (:minres, minres),
    ]

    mktempdir() do dir
        for (name, solver) in cases
            raw = joinpath(dir, string(name, "_raw.npy.zstd"))
            conv = joinpath(dir, string(name, "_conv"))
            result = solver(
                A, b, seed, spectral_norm,
                tiny_then_indefinite_preconditioner(),
                Int64(200), raw, conv;
                convergence_tolerance = 1.0e-1,
                correction_restart_normwise_tolerance = 1.0e-1,
            )

            @test result.status == BREAKDOWN
            @test result.correction_restarts == 0
            @test isempty(result.correction_restart_records)
            @test result.false_native_crossings == 1
            @test first(result.certification_records)["action"] == "correction_restart_breakdown"
            @test !isfile(Solvers._tridiag_block_path(raw, 1))

            data = read_npy_zstd(raw)
            @test all(data[:, end] .== 0.0)
        end
    end
end

@testset "explicit correction restart clears unwritten next tridiag path" begin
    A, b, seed, spectral_norm = replacement_test_problem()

    mktempdir() do dir
        raw = joinpath(dir, "pcg_raw.npy.zstd")
        conv = joinpath(dir, "pcg_conv")
        result = pcg(
            A, b, seed, spectral_norm,
            zero_recursive_residual_once(),
            Int64(1), raw, conv;
            convergence_tolerance = 1.0e-1,
            correction_restart_normwise_tolerance = 1.0e-1,
        )

        @test result.status != CONVERGED
        @test result.residual_replacements == 0
        @test isempty(result.residual_replacement_records)
        @test result.correction_restarts == 1

        record = only(result.correction_restart_records)
        @test isfile(record["closed_tridiag_path"])
        @test isfile(record["base_solution_path"])
        @test record["next_tridiag_path"] === nothing
        @test !isfile(Solvers._tridiag_block_path(raw, 1))
    end
end
