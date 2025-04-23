# CAPPABenchmarkTools.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

A lightweight Julia toolbox for benchmarking sparse linear solvers and preconditioners.

## Features

- **Sparse triangular solves** via built-in and custom kernels
- **CG** and **MINRES** implementations that emit per-iteration data for detailed profiling
- **Uniform “preconditioner” wrapper**—turn any Julia‐callable into a `Preconditioner` object that tracks operator applications
- Utilities for loading MatrixMarket graphs & matrices, computing norms, reordering, scaling, and more

## Installation

```julia
] add https://github.com/tunnellm/CAPPABenchmarkTools.jl
using CAPPABenchmarkTools
```

## Dependencies

* SparseArrays, LinearAlgebra, Statistics
* Laplacians, CombinatorialMultigrid, LDLFactorizations
* PyCall, MATLAB, CSV, DataFrames, CodecZstd, TranscodingStreams, BufferedStreams, MatrixMarket
* StableRNGs, Random, AMD, SymRCM, GenericArpack

```julia
] add SparseArrays, add LinearAlgebra, add Statistics
] add Laplacians, add CombinatorialMultigrid, add LDLFactorizations
] add PyCall, add MATLAB, add CSV, add DataFrames, add CodecZstd, add TranscodingStreams, add BufferedStreams, add MatrixMarket
] add StableRNGs, add Random, add AMD, add SymRCM, add GenericArpack
```

## Directory layout assumptions

This package assumes the following directory layout for graphs and matrices.

```
.
├─ matrices/
│  └─ <matrix_name>/
│     └─ <matrix_name>.mtx
└─ graphs/
   └─ <graph_name>/
      └─ <graph_name>.mtx
```

Please make sure these files exist before calling `Utilities.load_matrix` or `Utilities.load_graph`.


## Loading the package

You can load everything into the namespace:

```julia
using CAPPABenchmarkTools
```

or load just what you need:

```julia
using CAPPABenchmarkTools.Utilities
using CAPPABenchmarkTools: Preconditioner, preconditioned_conjugate_gradient
```

## Quickstart guide

```julia
using CAPPABenchmarkTools

# load a "problem interface"
prob = load_matrix("1138_bus", "natural");

# Access the scaled system
pkg = prob.Scaled

# 2. Build a preconditioner (wrap a function, lets use identity)
M = Preconditioner(
  LinearOperator = (y, x) -> y .= x, # our linear operator overwrites the y with the result applied to x.
  num_multiplications = 0, # keep track of work applied by the preconditioner, here we do 0 work.
  system = pkg
)

# Alternatively, we could have called
Q = Control(prob); # this uses the Scaled system by default.

# 3. Run CG or MINRES
res = preconditioned_conjugate_gradient(
  preconditioner = M, # the struct already holds the system data.
  maxiters = 100,
  raw_data_out = "temp/raw_data.csv.zstd", # location to output all performance data.
  convergence_data_out = "temp/convergence_data_"; # this will output convergence data with respect to multiple metrics.
  work_limit = 40000 # stops the solver if we exceeed an amount of work. We use this to limit a preconditioned run if it exceeds some multiple of the work to converge without preconditioning.
);

# Alternatively, we could use the wrapper as follows.

run_matrix(
  matrix_name = "1138_bus", # provide just the matrix name
  preconditioners = Function[Control], # provide a vector of preconditioner functions (defined in Preconditioners)
  args = [nothing], # provide a vector of tuples of arguments, or provide "nothing" if the preconditioner takes no arguments.
  reordering = "Natural"; # the name of the ordering. Currently: Natural, AMD, or RCM
  run_conjugate_gradient = true, # run one of or both solvers.
  run_minres = false
)

# Or as follows:

run_matrix(
  matrix = prob, # provide just problem interface
  preconditioners = Function[Control], # provide a vector of preconditioner functions (defined in Preconditioners)
  args = [nothing], # provide a vector of tuples of arguments, or provide "nothing" if the preconditioner takes no arguments.
  reordering = "Natural"; # the name of the ordering. Currently: Natural, AMD, or RCM
  run_conjugate_gradient = true, # run one of or both solvers.
  run_minres = false
)
```

## Core types

```julia
struct norm_information
    spectral_norm::Float64
    infinity_norm::Float64
    frobenius_norm::Float64
    jacobian::Float64
end

struct package
    A::SparseMatrixCSC
    b::Vector
    seed::Vector
    name::String
    norms::norm_information
    diagonally_dominant::Bool
    nullspace::Int64
end

struct Preconditioner
    LinearOperator::Function
    num_multiplications::UInt64
    system::package
end
```

## Utilities

All behind-the-scenes routines live in `Utilities`:

- I/O: `load_matrix`, `load_graph`, `load_matrix_names`, `check_previous`
- Matrix transforms: `make_augmented`, `extract_negative`, `laplace_to_adj`
- Norms & tolerances: `compute_operator_norms`, `compute_tolerances`
- Problem inspection: `is_laplacian`, `is_sdd`



License: This project is MIT-licensed. See [license](LICENSE.md).
