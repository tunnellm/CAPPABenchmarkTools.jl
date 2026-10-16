using CAPPABenchmarkTools
using SparseArrays, MatrixMarket, JSON3

function matrix_pipeline_example(output::String)
    name = "synthetic_spd"
    inputs = joinpath(output, "inputs")
    results = joinpath(output, "results")
    source = joinpath(inputs, name, name * ".mtx")
    if !isdir(joinpath(inputs, name, "components"))
        mkpath(dirname(source))
        n = 64
        A = spdiagm(-1 => fill(-0.25, n-1), 0 => ones(n), 1 => fill(-0.25, n-1))
        MatrixMarket.mmwrite(source, A)
    end
    # Only generated input is used here: preparation may delete its .mtx file.
    for builder in (Control, SGS)
        kwargs = builder === Control ? Dict{String,Any}() : Dict{String,Any}("num_iters" => 1)
        run_matrix(name, builder; preconditioner_kwargs=kwargs,
            solvers=[:pcg, :minres], base_dir=inputs, results_location=results)
    end
    records = String[]
    for (root, _, files) in walkdir(results)
        if "metadata.json" in files
            path = joinpath(root, "metadata.json")
            meta = JSON3.read(read(path, String))
            meta.status == "CONVERGED" || error("Example failed: $path: $(meta.status)")
            push!(records, path)
        end
    end
    length(records) == 4 || error("Expected four solver results; found $(length(records))")
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = isempty(ARGS) ? joinpath(@__DIR__, "output", "pipeline") : abspath(only(ARGS))
    println("Verified ", length(matrix_pipeline_example(output)), " results under ", output)
end
