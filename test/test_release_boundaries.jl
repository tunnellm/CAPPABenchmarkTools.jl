using Test, SparseArrays, JSON3, CSV, TOML, PythonCall
using CAPPABenchmarkTools
const S = CAPPABenchmarkTools.Solvers
const U = CAPPABenchmarkTools.Utilities

@testset "Exact zero RHS in all six solvers" begin
    A = spdiagm(0 => [1.0, 2.0, 3.0])
    prec = Preconditioner((z,r) -> error("zero RHS must not apply the preconditioner"), UInt64(0), UInt64(0))
    for name in (:pcg, :lanczos_pcg, :minres, :pcr, :fpcg1, :fpcg2),
        mode in (:certified, :bounded_continuation), iterations in (0, 3)
        mktempdir() do dir
            raw = joinpath(dir, "raw_data.npy.zstd")
            conv = joinpath(dir, "convergence_data")
            solution = joinpath(dir, "solution.npy.zstd")
            events = Dict{String,Any}[]
            x = fill(NaN, 3)
            result = getfield(S, name)(A, [0.0,-0.0,0.0], zeros(3), 3.0,
                prec, Int64(iterations), raw, conv; solver_mode=mode, work_limit=Int64(1),
                _experimental_final_iterate=x, write_solution=solution,
                _diagnostic_block_rows=1, progress_callback=e -> push!(events, copy(e)))
            @test result.status == CONVERGED
            @test result.iterations == 0
            @test result.total_fma == result.final_relative_residual == 0.0
            @test result.correction_restarts == result.residual_replacements == 0
            @test result.physical_true_residual_evals == result.certification_true_residual_evals == 0
            @test isempty(result.certification_check_iters)
            @test isempty(result.true_residual_used_for_decision_iters)
            @test result.native_candidate_iter === nothing
            @test result.solver_policy["continuation_work_limit"] === nothing
            @test result.solver_policy["initial_solution"] == "exact_zero_rhs"
            record = only(result.certification_records)
            @test record["certified"] && !record["certification_charged"]
            @test record["certification_check_source"] == "zero_rhs_identity"
            @test x == zeros(3)
            @test U.read_npy_zstd_1d_f64(solution) == x
            data = S.open_diagnostic_data(raw; require_finalized=true)
            @test size(data,1) == 1
            @test all(iszero, data[1,1:7])
            @test isnan(data[1,8]) # no recurrence scalar was computed
            @test data[1,9] == data[1,10] == 0.0
            for suffix in ("relative_residual", "relative_normwise_2")
                rows = collect(CSV.File(conv * "_" * suffix * ".csv"))
                @test length(rows) == length(S.tolerances)
                @test all(r -> r.iteration == r.actual == 0, rows)
            end
            rows = collect(CSV.File(conv * "_certification.csv"))
            @test length(rows) == length(S.tolerances)
            @test all(r -> r.certified_iter == r.sim_certified_iter == r.true_hit_iter == 0, rows)
            @test all(r -> r.certified_total_fma == r.sim_certified_total_fma == 0, rows)
            @test all(r -> r.certification_checks_to_certified == r.sim_certification_checks_to_certified == 0, rows)
            @test all(r -> r.certified_native_relres == r.sim_certified_native_relres == 0.0, rows)
            @test all(r -> r.sim_status == "certified_from_raw", rows)
            meta = Dict{String,Any}("correction_restarts"=>0)
            S._diagnostic_storage_metadata!(meta, dir)
            @test meta["raw_data_storage"]["version"] == 2
            @test isempty(meta["coefficient_storage"])
            terminal = only(filter(e -> get(e,"progress_kind",nothing) == "numerical" && get(e,"final",false), events))
            @test terminal["iteration"] == 0 && terminal["status"] == "CONVERGED"
            @test isempty(terminal["invalid_fields"])
            @test !terminal["certification_charged"]
            @test count(e -> get(e,"event",nothing)=="phase_start", events) ==
                  count(e -> get(e,"event",nothing)=="phase_done", events)
        end
    end

    @testset "Tiny nonzero RHS is not an exact zero" begin
        mktempdir() do dir
            @test_throws ErrorException S.pcr(A, fill(1e-200,3), zeros(3), 3.0, prec,
                Int64(1),joinpath(dir,"raw.npy.zstd"),joinpath(dir,"conv"))
        end
    end
    @testset "Retry does not expose stale coefficients" begin
        mktempdir() do dir
            identity, _ = Control(A)
            raw = joinpath(dir,"raw_data.npy.zstd")
            conv = joinpath(dir,"conv")
            S.pcr(A, ones(3), [1.0,0.5,1/3], 3.0, identity, Int64(1),raw,conv)
            @test isfile(S._tridiag_block_path(raw,0))
            S.pcr(A, zeros(3), zeros(3), 3.0, prec, Int64(0),raw,conv)
            meta = Dict{String,Any}("correction_restarts"=>0)
            S._diagnostic_storage_metadata!(meta,dir)
            @test isempty(meta["coefficient_storage"])
        end
    end
end

@testset "Legacy-cache errors give usable public guidance" begin
    mktempdir() do dir
        comp_dir = joinpath(dir,"old","components")
        mkpath(comp_dir)
        metadata = Dict("components" => [Dict("index"=>1,"dimension"=>2,
            "reorderings"=>Dict("amd"=>"amd_c1.npy"))])
        open(joinpath(comp_dir,"metadata.json"),"w") do io
            JSON3.write(io,metadata)
        end
        desc = U.ComponentDescriptor(name="old",component_index=1,base_dir=dir,
            dimension=2,nnz=2,classification=:sddm,lambda_max=1.0,lambda_min=nothing,
            ritz_min_residual=nothing,ritz_max_residual=0.0,ritz_min_half_gap=nothing,
            ritz_max_half_gap=0.0,scaled_lambda_max=1.0,scaled_lambda_min=nothing,
            scaled_ritz_min_residual=nothing,scaled_ritz_max_residual=0.0,
            scaled_ritz_min_half_gap=nothing,scaled_ritz_max_half_gap=0.0,
            unscaled_is_sdd=true,scaled_is_sdd=true)
        for f in (() -> U.load_components("old",dir),
                  () -> U.load_reordering("old",dir,1,"amd"),
                  () -> U.load_component_matrix(desc))
            err = try f(); nothing catch e; e end
            @test err isa ErrorException
            message = sprint(showerror,err)
            @test occursin("prepare a fresh cache",message)
            @test occursin("Preserve the existing data",message)
            @test !occursin("scripts/",message)
        end
    end
end

@testset "Ordering workers use the consumer environment" begin
    root = dirname(@__DIR__)
    original_env = dirname(Base.active_project())
    python = pyconvert(String,pyimport("sys").executable)
    mktempdir() do dir
        checkout = joinpath(dir,"checkout")
        app = joinpath(dir,"application")
        mkpath(checkout)
        mkpath(app)
        cp(joinpath(root,"src"),joinpath(checkout,"src"))
        cp(joinpath(root,"Project.toml"),joinpath(checkout,"Project.toml"))
        cp(joinpath(root,"CondaPkg.toml"),joinpath(checkout,"CondaPkg.toml"))
        symlink(joinpath(root,"External"),joinpath(checkout,"External"))
        project = TOML.parsefile(joinpath(root,"Project.toml"))
        # Reuse the test environment's resolved dependencies without contacting
        # registries or creating a manifest in the package checkout.
        manifest = TOML.parsefile(joinpath(original_env,"Manifest.toml"))
        for entries in values(manifest["deps"]), entry in entries
            haskey(entry,"path") || continue
            entry["path"] = abspath(joinpath(original_env,entry["path"]))
        end
        manifest["deps"]["CAPPABenchmarkTools"] = [Dict(
            "uuid"=>project["uuid"],"version"=>project["version"],"path"=>checkout,
            "deps"=>collect(keys(project["deps"])))]
        open(joinpath(app,"Manifest.toml"),"w") do io
            TOML.print(io,manifest)
        end
        open(joinpath(app,"Project.toml"),"w") do io
            TOML.print(io,Dict("deps"=>Dict("CAPPABenchmarkTools"=>project["uuid"])))
        end
        command = `$(Base.julia_cmd()) --startup-file=no --threads=2 --project=$app $(joinpath(@__DIR__,"ordering_consumer.jl")) $checkout $dir`
        @test success(run(ignorestatus(addenv(command,
            "JULIA_LOAD_PATH"=>"@:@stdlib", "JULIA_CONDAPKG_BACKEND"=>"Null",
            "JULIA_PYTHONCALL_EXE"=>python))))
        @test !isfile(joinpath(checkout,"Manifest.toml"))
    end
end
