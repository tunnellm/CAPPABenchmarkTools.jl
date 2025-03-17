module CAPPABenchmarkTools

using Reexport

include("Algorithms.jl")

@reexport using .Algorithms

# export sparse_triangular_solve, sparse_triangular_solve!

include("Utilities.jl")

@reexport using .Utilities

# export check_previous, load_matrix_names 
# export load_matrix, load_graph, package
# export norm_information
# export compute_tolerances, compute_operator_norms
# export problem_interface
# export is_laplacian, make_augmented, extract_negative, laplace_to_adj

include("Preconditioners.jl")

@reexport using .Preconditioners

# export Preconditioner
# export Control
# export TruncatedNeumann  # Truncated Neumann
# export SOR, SSOR, GaussSeidel, SymmetricGaussSeidel  # Iterative methods
# export IncompleteCholesky, ModifiedIncompleteCholesky  # Incomplete Cholesky (MATLAB)
# export SuperILU, SuperLLT  # SuperLU 
# export SymmetricSPAI  # Saunders (MATLAB)
# export AMG_rube_stuben, AMG_smoothed_aggregation  # PyAmg
# export LaplaceStripped # Laplacians.jl
# export CombinatorialMG  # CombinatorialMultigrid.jl
# export SteepestDescent  # Basic iterative method

include("Solvers.jl")

@reexport using .Solvers

# export preconditioned_conjugate_gradient, preconditioned_minres

include("TestWrapper.jl")

@reexport using .TestWrapper

# export run_problem

export Algorithms, Utilities, Solvers, Preconditioners, TestWrapper


end # module CAPPABenchmarkTools
