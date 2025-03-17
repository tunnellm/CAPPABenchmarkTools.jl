global const tolerances = [10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10]
global const maximum_tolerance = maximum(tolerances)
global const minimum_tolerance = minimum(tolerances)

global const buffer_size = 128 * 1024 * 1024  # 128 MB

global const raw_header = b"Iteration, Relative Residual, Error, Relative Normwise 2, Relative Normwise ∞, Relative Normwise F, Base Cost, Preconditioner Cost, |μ|, |r1|, r1⋅z1, pA'p, |p|, polynomial\n"
global const convergence_header = b"Iteration, Tolerance Met, Actual, Base Cost, Preconditioner Cost\n"
global const minres_raw_header = b"Iteration, Relative Residual, Error, Relative Normwise 2, Relative Normwise ∞, Relative Normwise F, Base Cost, Preconditioner Cost, |phi|, |r1|, r2y, beta, alfa, anorm, acond, dxnorm, rnorm, epsx, epsr, |w|, polynomial\n"