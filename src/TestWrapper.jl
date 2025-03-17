module TestWrapper

using SparseArrays
using LinearAlgebra

using ..Preconditioners
using ..Utilities
using ..Solvers

export run_matrix


"""
    iter_print_statement(matrix, preconditioner, solver, max_iters, iters) -> String

Generates a standardized message describing the outcome of an iterative solver run.

# Arguments
- `matrix`: The name or identifier of the matrix being solved.
- `preconditioner`: The preconditioner applied during the solver execution.
- `solver`: The name of the iterative solver used (e.g., `"Conjugate Gradient"`, `"MINRES"`).
- `max_iters::Int`: The maximum number of iterations allowed for the solver.
- `iters::Int`: The actual number of iterations performed.

# Returns
- `String`: A formatted message summarizing the solver's termination condition.

# Notes
- If `iters == -2`, the solver stopped due to exceeding a predefined work limit.
- If `iters == -1`, the solver halted due to convergence issues.
- If `iters == 0`, the solver failed before performing any iterations.
- If `iters == max_iters`, the solver stopped due to reaching the iteration limit.
- Otherwise, the solver met its convergence criteria successfully.
"""
function iter_print_statement(matrix, preconditioner, solver, max_iters, iters)
    iters == -2 && return "$matrix with $(preconditioner) halted in $(solver) due to work limit"
    iters == -1 && return "$matrix with $(preconditioner) halted due to convergence issues in $(solver)"
    iters == 0 && return "$matrix with $(preconditioner) failed prior to starting iterations."
    iters == max_iters && return "$matrix with $(preconditioner) halted in $(solver) due to iteration limit."
    return "$matrix with $(preconditioner) met $(solver) convergence criteria."
end


"""
    run_matrix(
        matrix_name::String,
        preconditioners::Vector{Function},
        args,
        reordering::String;
        run_conjugate_gradient::Bool=true,
        run_minres::Bool=true,
        loader::Symbol=:SPD_matrix,
        results_location::String="./results/"
    )

Runs iterative solvers on a given matrix with multiple preconditioners and stores the results.

# Arguments
- `matrix_name::String`: The name of the matrix to be loaded.
- `preconditioners::Vector{Function}`: A list of preconditioner functions to apply.
- `args`: Arguments for each preconditioner, either as a tuple or `nothing` if no arguments are needed.
- `reordering::String`: The reordering strategy used when loading the matrix.
- `run_conjugate_gradient::Bool` (optional, default `true`): Whether to run the Conjugate Gradient (CG) solver.
- `run_minres::Bool` (optional, default `true`): Whether to run the MINRES solver.
- `loader::Symbol` (optional, default `:SPD_matrix`): Specifies how to load the matrix, either as an SPD matrix or as a graph.
- `results_location::String` (optional, default `"./results/"`): Directory where results are stored.

# Notes
- The matrix is loaded using either `load_matrix` or `load_graph`, depending on `loader`.
- Iterative solvers are executed for each preconditioner unless results already exist.
- The function first checks previous convergence data using `check_previous` to avoid redundant computation.
- Preconditioners that cause errors are skipped with a printed warning.
- Results are stored in structured subdirectories under `results_location/matrix_name/reordering/`.
- Each solver is run under a computational work limit determined by prior convergence data.
- Solver outcomes are printed using `iter_print_statement`.

# Behavior
1. **Prepare Paths:** Creates directories for storing results.
2. **Check Previous Runs:** Skips solvers if results already exist.
3. **Apply Preconditioners:** If an error occurs during setup, the preconditioner is skipped.
4. **Run Iterative Solvers:** Runs CG and/or MINRES and logs results.
5. **Store Results:** Saves raw iteration data and convergence metrics.

# Error Handling
- If a preconditioner fails, the function prints an error and continues to the next.
- If an iterative solver encounters an error, it is logged, and execution continues.
"""
function run_matrix(matrix_name::String, preconditioners::Vector{Function}, args, reordering::String; run_conjugate_gradient::Bool=true, run_minres::Bool=true, loader::Symbol=:SPD_matrix, results_location::String="./results/")
    
    @assert run_conjugate_gradient == true || run_minres == true

    mat::problem_interface = loader == :SPD_matrix ? load_matrix(matrix_name, reordering) : load_graph(matrix_name, reordering)

    run_matrix(mat, preconditioners, args, reordering; run_conjugate_gradient=run_conjugate_gradient, run_minres=run_minres, results_location=results_location)

    return
end

"""
    run_matrix(
        matrix::problem_interface,
        preconditioners::Vector{Function},
        args,
        reordering::String;
        run_conjugate_gradient::Bool=true,
        run_minres::Bool=true,
        loader::Symbol=:SPD_matrix,
        results_location::String="./results/"
    )

Runs iterative solvers on a given matrix with multiple preconditioners and stores the results.

# Arguments
- `matrix::problem_interface`: The matrix or graph to be solved.
- `preconditioners::Vector{Function}`: A list of preconditioner functions to apply.
- `args`: Arguments for each preconditioner, either as a tuple or `nothing` if no arguments are needed.
- `reordering::String`: The reordering strategy used when loading the matrix.
- `run_conjugate_gradient::Bool` (optional, default `true`): Whether to run the Conjugate Gradient (CG) solver.
- `run_minres::Bool` (optional, default `true`): Whether to run the MINRES solver.
- `loader::Symbol` (optional, default `:SPD_matrix`): Specifies how to load the matrix, either as an SPD matrix or as a graph.
- `results_location::String` (optional, default `"./results/"`): Directory where results are stored.

# Notes
- Iterative solvers are executed for each preconditioner unless results already exist.
- The function first checks previous convergence data using `check_previous` to avoid redundant computation.
- Preconditioners that cause errors are skipped with a printed warning.
- Results are stored in structured subdirectories under `results_location/matrix.matrix_names/reordering/`.
- Each solver is run under a computational work limit determined by prior convergence data.
- Solver outcomes are printed using `iter_print_statement`.

# Behavior
1. **Prepare Paths:** Creates directories for storing results.
2. **Check Previous Runs:** Skips solvers if results already exist.
3. **Apply Preconditioners:** If an error occurs during setup, the preconditioner is skipped.
4. **Run Iterative Solvers:** Runs CG and/or MINRES and logs results.
5. **Store Results:** Saves raw iteration data and convergence metrics.

# Error Handling
- If a preconditioner fails, the function prints an error and continues to the next.
- If an iterative solver encounters an error, it is logged, and execution continues.
"""
function run_matrix(matrix::problem_interface, preconditioners::Vector{Function}, args, reordering::String; run_conjugate_gradient::Bool=true, run_minres::Bool=true, results_location::String="./results/")


    matrix_size = size(matrix.Scaled.A, 1)

    base_location = results_location * matrix.name * "/" * reordering

    cg_location = base_location * "/cg/"
    mr_location = base_location * "/minres/"

    if run_conjugate_gradient == true
        mkpath(cg_location * "Control/")
    end
    if run_minres == true
        mkpath(mr_location * "Control/")
    end

    control_iters_cg = 10 * matrix_size
    control_iters_mr = 10 * matrix_size

    cg_control_work = check_previous(cg_location, "Control/", true) * nnz(matrix.Scaled.A)
    mr_control_work = check_previous(mr_location, "Control/", true) * nnz(matrix.Scaled.A)

    for (i, preconditioner) in enumerate(preconditioners)
        
        output_filename = string(preconditioner) * "/"
        
        if typeof(args[i]) <: Tuple
            for arg in args[i]
                output_filename *= string(arg) * "/"
            end
        elseif args[i] !== nothing
            output_filename *= string(args[i]) * "/"
        end

        local p
        if run_conjugate_gradient == true
            cg_iters = check_previous(cg_location, output_filename, false; control_work=4 * cg_control_work)
        else
            cg_iters = 1
        end
        if run_minres == true
            mr_iters = check_previous(mr_location, output_filename, false; control_work=4 * mr_control_work)
        else
            mr_iters = 1
        end
        if cg_iters > 0 && mr_iters > 0
            println("Skipping $(string(preconditioner)) for $(matrix.name), $output_filename")
            continue
        end
        try
            if args[i] === nothing
                p = preconditioner(matrix)
            else
                p = preconditioner(matrix, args[i]...)
            end
        catch e
            println(e)
            println("Error in preconditioner $(string(preconditioner)) for $(matrix.name), $(matrix.name)")
            continue
        end

        if cg_iters <= 0 && run_conjugate_gradient == true
            output_location = cg_location * output_filename
            mkpath(output_location)
            try
                iters = preconditioned_conjugate_gradient(p, control_iters_cg, output_location * "raw_data.csv.zstd", output_location * "convergence_data"; work_limit=4*cg_control_work)
                println(iter_print_statement(matrix.name, string(preconditioner), "Conjugate Gradient", control_iters_cg, iters))
            catch e
                println("Error returned by Conjugate Gradient for preconditioner $(string(preconditioner)) for matrix $(matrix.name)")
                continue
            end
        end
        if mr_iters <= 0 && run_minres == true
            output_location = mr_location * output_filename
            mkpath(output_location)
            try
                iters = preconditioned_minres(p, control_iters_mr, output_location * "raw_data.csv.zstd", output_location * "convergence_data"; work_limit=4*mr_control_work)
                println(iter_print_statement(matrix.name, string(preconditioner), "MINRES", control_iters_mr, iters))
            catch e
                println("Error returned by MINRES for preconditioner $(string(preconditioner)) for matrix $(matrix.name)")
                continue
            end
        end
    end
end

end # module