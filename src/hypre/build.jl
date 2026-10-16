using Libdl

const HYPRE_SOURCE_DIR = normpath(joinpath(@__DIR__, "..", "..", "External", "hypre"))
const HYPRE_BUILD_DIR = joinpath(HYPRE_SOURCE_DIR, "build-$(String(Sys.ARCH))")

# Determine shared library name and path
const LIBHYPRE_NAME = Sys.isapple() ? "libHYPRE.dylib" : "libHYPRE.so"
const LIBHYPRE = joinpath(HYPRE_BUILD_DIR, "lib", LIBHYPRE_NAME)

function build_hypre()
    NativeLibraries.allow_build()
    @info "Building HYPRE shared library..." source=HYPRE_SOURCE_DIR build=HYPRE_BUILD_DIR

    if !isdir(HYPRE_SOURCE_DIR) || !isfile(joinpath(HYPRE_SOURCE_DIR, "src", "CMakeLists.txt"))
        error("HYPRE source not found at $HYPRE_SOURCE_DIR")
    end

    mkpath(HYPRE_BUILD_DIR)

    cmake_cmd = `cmake $(joinpath(HYPRE_SOURCE_DIR, "src"))
        -DBUILD_SHARED_LIBS=ON
        -DHYPRE_ENABLE_MPI=OFF
        -DHYPRE_ENABLE_OPENMP=ON
        -DCMAKE_BUILD_TYPE=Release
        -DHYPRE_BUILD_TESTS=OFF`

    nprocs = parse(Int, get(ENV, "CAPPA_BUILD_JOBS", string(min(4, Sys.CPU_THREADS))))
    nprocs > 0 || throw(ArgumentError("CAPPA_BUILD_JOBS must be positive"))
    build_cmd = `cmake --build . -j$nprocs`

    cd(HYPRE_BUILD_DIR) do
        run(cmake_cmd)
        run(build_cmd)
    end

    if !isfile(LIBHYPRE)
        error("HYPRE build completed but $LIBHYPRE_NAME not found at expected location: $LIBHYPRE")
    end

    @info "HYPRE built successfully" lib=LIBHYPRE
end

function ensure_built()
    NativeLibraries.ensure(LIBHYPRE, build_hypre)
end
