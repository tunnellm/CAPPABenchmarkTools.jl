using Pkg

root = dirname(@__DIR__)
for (directory, required) in (("Laplacians", "Project.toml"),
                              ("FastSSAI3", "src/FastSSAI3.jl"),
                              ("LimitedLDLFactorizations", "src/LimitedLDLFactorizations.jl"),
                              ("ILUK", "src/ILUK.jl"),
                              ("hypre", "src/CMakeLists.txt"),
                              ("pyamg", "pyamg/__init__.py"))
    isfile(joinpath(root, "External", directory, required)) ||
        error("Missing submodule $directory. Run git submodule update --init --recursive first.")
end

Pkg.activate(root)
Pkg.instantiate()
using CondaPkg
CondaPkg.resolve()
using CAPPABenchmarkTools
using PythonCall

expected = realpath(joinpath(root, "External", "pyamg", "pyamg", "__init__.py"))
actual = realpath(pyconvert(String, pyimport("pyamg").__file__))
actual == expected || error("Python selected a different PyAMG installation: $actual")
println("Setup complete. PyAMG: ", actual)
