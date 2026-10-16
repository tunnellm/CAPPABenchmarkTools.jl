"""
Rigorous tests for the matrix loading/decomposition pipeline in Utilities.jl.

Covers:
  1. classify_component — all classification categories
  2. adjacency_to_laplacian — basic, negative weights, self-loops
  3. decompose_and_cache — SPD, Laplacian, adjacency, pinning-disconnects
  4. describe_components — cache hit, cache miss, delete_original
  5. matrix_type override — :auto, :adjacency, :laplacian
  6. prepare_system — load, reorder, scale, RHS generation
  7. Edge cases — 1×1 matrices, 2×2 matrices, fully connected graphs, isolated nodes
"""

using CAPPABenchmarkTools
using CAPPABenchmarkTools.Utilities
using SparseArrays
using LinearAlgebra
using MatrixMarket
using StableRNGs
using JSON3
using Test

# ─── Helper ──────────────────────────────────────────────────────────

function make_path_laplacian(n::Int)
    W = spzeros(n, n)
    for i in 1:n-1
        W[i, i+1] = 1.0
        W[i+1, i] = 1.0
    end
    return adjacency_to_laplacian(W), W
end

function make_star_laplacian(n::Int)
    # Star with hub at node 1, n-1 leaves
    W = spzeros(n, n)
    for i in 2:n
        W[1, i] = 1.0
        W[i, 1] = 1.0
    end
    return adjacency_to_laplacian(W), W
end

function make_cycle_laplacian(n::Int)
    W = spzeros(n, n)
    for i in 1:n
        j = mod1(i + 1, n)
        W[i, j] = 1.0
        W[j, i] = 1.0
    end
    return adjacency_to_laplacian(W), W
end

function make_spd(n::Int)
    # Tridiagonal SPD matrix (not a Laplacian — positive diagonal excess)
    A = spzeros(n, n)
    for i in 1:n
        A[i, i] = 4.0
        if i > 1; A[i, i-1] = -1.0; end
        if i < n; A[i, i+1] = -1.0; end
    end
    return A
end

function write_matrix(base_dir, name, A)
    mkpath(joinpath(base_dir, name))
    MatrixMarket.mmwrite(joinpath(base_dir, name, "$name.mtx"), A)
end

# ─── 0. MatrixMarket reader ─────────────────────────────────────────

@testset "read_matrixmarket_coord fields" begin
    A_real = read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate real general
        3 3 3
        1 2 2.5
        2 3 0.0
        3 1 -1.25
        """))
    @test eltype(A_real.nzval) == Float64
    @test size(A_real) == (3, 3)
    @test A_real[1, 2] == 2.5
    @test A_real[2, 3] == 0.0
    @test A_real[3, 1] == -1.25
    @test nnz(A_real) == 3

    A_integer = read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate integer general
        3 3 2
        1 1 7
        3 2 -4
        """))
    @test eltype(A_integer.nzval) == Float64
    @test A_integer[1, 1] == 7.0
    @test A_integer[3, 2] == -4.0

    A_int = read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate int general
        2 2 1
        2 1 3
        """))
    @test eltype(A_int.nzval) == Float64
    @test A_int[2, 1] == 3.0

    A_pattern = read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate pattern symmetric
        3 3 2
        1 2
        3 3
        """))
    @test eltype(A_pattern.nzval) == Float64
    @test A_pattern[1, 2] == 1.0
    @test A_pattern[2, 1] == 1.0
    @test A_pattern[3, 3] == 1.0
    @test nnz(A_pattern) == 3

    A_binary = read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate binary general
        3 3 3
        1 2
        2 3 0
        3 1 true
        """))
    @test eltype(A_binary.nzval) == Float64
    @test A_binary[1, 2] == 1.0
    @test A_binary[2, 3] == 0.0
    @test A_binary[3, 1] == 1.0
    @test nnz(A_binary) == 3

    @test_throws ErrorException read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate complex general
        1 1 1
        1 1 1.0 0.0
        """))
    @test_throws ErrorException read_matrixmarket_coord(IOBuffer("""%%MatrixMarket matrix coordinate binary general
        1 1 1
        1 1 2
        """))
end

# ─── 1. classify_component ──────────────────────────────────────────

@testset "classify_component" begin
    @testset "graph_laplacian" begin
        L, _ = make_path_laplacian(5)
        @test classify_component(L) == :graph_laplacian

        L2, _ = make_cycle_laplacian(6)
        @test classify_component(L2) == :graph_laplacian

        L3, _ = make_star_laplacian(4)
        @test classify_component(L3) == :graph_laplacian
    end

    @testset "adjacency — non-negative, zero diagonal" begin
        _, W = make_path_laplacian(5)
        @test classify_component(W) == :adjacency

        _, W2 = make_star_laplacian(4)
        @test classify_component(W2) == :adjacency
    end

    @testset "adjacency — negative weights, zero diagonal" begin
        W = sparse([0.0 -2.0 1.0; -2.0 0.0 3.0; 1.0 3.0 0.0])
        @test classify_component(W) == :adjacency
    end

    @testset "NOT adjacency — self-loops (nonzero diagonal)" begin
        W = sparse([2.0 1.0; 1.0 3.0])
        @test classify_component(W) != :adjacency
    end

    @testset "sddm" begin
        A = make_spd(5)
        @test classify_component(A) == :sddm
    end

    @testset "general_sdd — positive off-diagonals, diagonally dominant" begin
        A = sparse([4.0 1.0 -1.0; 1.0 4.0 1.0; -1.0 1.0 4.0])
        @test classify_component(A) == :general_sdd
    end

    @testset "symmetric — positive diagonal, not diagonally dominant" begin
        # SPD but not SDD: row 1 has diag=1.0, off-diag sum=1.8 > 1.0
        A = sparse([1.0 -0.9 -0.9; -0.9 2.0 -0.1; -0.9 -0.1 2.0])
        @test classify_component(A) == :symmetric
    end

    @testset "symmetric_indefinite — non-positive diagonal" begin
        A = sparse([-0.9 -0.9; -0.9 1.0])
        @test classify_component(A) == :symmetric_indefinite
    end

    @testset "1×1 matrix" begin
        A = sparse([5.0;;])
        @test classify_component(A) == :sddm
    end

    @testset "2×2 Laplacian" begin
        L = sparse([1.0 -1.0; -1.0 1.0])
        @test classify_component(L) == :graph_laplacian
    end

    @testset "identity matrix — not adjacency (nonzero diagonal)" begin
        I3 = sparse(1.0I, 3, 3)
        @test classify_component(I3) != :adjacency
        @test classify_component(I3) == :sddm
    end

    @testset "zero matrix — adjacency (zero diagonal)" begin
        Z = spzeros(3, 3)
        @test classify_component(Z) == :adjacency
    end
end

# ─── 2. adjacency_to_laplacian ──────────────────────────────────────

@testset "adjacency_to_laplacian" begin
    @testset "basic conversion" begin
        _, W = make_path_laplacian(5)
        L = adjacency_to_laplacian(W)
        L_expected, _ = make_path_laplacian(5)
        @test L ≈ L_expected
    end

    @testset "zero row sums" begin
        _, W = make_star_laplacian(6)
        L = adjacency_to_laplacian(W)
        n = size(L, 1)
        for i in 1:n
            @test abs(sum(L[i, :])) < 1e-12
        end
    end

    @testset "negative weights — degree uses absolute values" begin
        W = sparse([0.0 -2.0; -2.0 0.0])
        L = adjacency_to_laplacian(W)
        # degree(1) = |-2| = 2, degree(2) = |-2| = 2
        # L = diag([2, 2]) - W = [2 2; 2 2]
        @test L ≈ sparse([2.0 2.0; 2.0 2.0])
        # Diagonal should be non-negative
        @test all(diag(L) .>= 0)
    end

    @testset "self-loops ignored in degree" begin
        W = sparse([3.0 1.0; 1.0 0.0])
        L = adjacency_to_laplacian(W)
        # degree(1) = |1| = 1 (self-loop ignored), degree(2) = |1| = 1
        # L = diag([1, 1]) - W + diag(diag(W)) = [1-3+3  -1; -1  1-0+0] = [1 -1; -1 1]
        @test L ≈ sparse([1.0 -1.0; -1.0 1.0])
    end

    @testset "symmetry preserved" begin
        W = sparse([0.0 1.0 2.0; 1.0 0.0 3.0; 2.0 3.0 0.0])
        L = adjacency_to_laplacian(W)
        @test issymmetric(L)
    end
end

# ─── 3. decompose_and_cache ─────────────────────────────────────────

@testset "decompose_and_cache" begin
    @testset "SPD matrix — stored directly" begin
        tmp = mktempdir()
        A = make_spd(10)
        write_matrix(tmp, "spd10", A)
        A_read = MatrixMarket.mmread(joinpath(tmp, "spd10", "spd10.mtx"))
        meta = decompose_and_cache(A_read, "spd10", tmp)

        @test meta["num_components"] == 1
        comp = meta["components"][1]
        @test comp["dimension"] == 10
        @test Symbol(comp["classification"]) == :sddm
        @test !haskey(comp, "origin_component")

        # Verify files exist (new schema: 3 .npy.zstd files per component)
        comp_dir = joinpath(tmp, "spd10", "components")
        @test isfile(joinpath(comp_dir, "metadata.json"))
        @test haskey(comp, "files")
        @test isfile(joinpath(comp_dir, comp["files"]["colptr"]))
        @test isfile(joinpath(comp_dir, comp["files"]["rowval"]))
        @test isfile(joinpath(comp_dir, comp["files"]["nzval"]))
    end

    @testset "graph Laplacian — pinned and stored" begin
        tmp = mktempdir()
        L, _ = make_path_laplacian(8)
        meta = decompose_and_cache(L, "path8", tmp)

        # Should have been pinned — no graph_laplacian in output
        for comp in meta["components"]
            @test Symbol(comp["classification"]) != :graph_laplacian
            @test haskey(comp, "origin_component")
            @test comp["origin_classification"] == "graph_laplacian"
            @test haskey(comp, "pinned_node")
        end
    end

    @testset "adjacency matrix — converted, pinned, stored" begin
        tmp = mktempdir()
        _, W = make_path_laplacian(6)
        meta = decompose_and_cache(W, "adj6", tmp)

        for comp in meta["components"]
            @test Symbol(comp["classification"]) != :adjacency
            @test Symbol(comp["classification"]) != :graph_laplacian
            @test comp["origin_classification"] == "graph_laplacian"
        end
    end

    @testset "star graph — pinning disconnects into n-1 components" begin
        tmp = mktempdir()
        L, _ = make_star_laplacian(7)
        meta = decompose_and_cache(L, "star7", tmp)

        @test meta["num_components"] == 6  # 7 nodes, pin hub, 6 leaves
        # All 6 leaves are dim=1 trivials, batched into a single "batch" entry.
        for comp in meta["components"]
            @test get(comp, "batch", false) == true
            @test comp["count"] == 6
            @test comp["origin_classification"] == "graph_laplacian"
        end
    end

    @testset "block diagonal — multiple independent components" begin
        tmp = mktempdir()
        # 3×3 SPD block + 2×2 Laplacian block
        A_spd = make_spd(3)
        L_lap = sparse([1.0 -1.0; -1.0 1.0])
        A_block = blockdiag(A_spd, L_lap)
        meta = decompose_and_cache(A_block, "mixed_block", tmp)

        # Trivial-component batches don't carry "classification"; filter those out.
        classifications = [Symbol(c["classification"]) for c in meta["components"]
                           if haskey(c, "classification")]
        # SPD block should be stored directly as sddm
        # Laplacian block should be pinned → batched trivials (no classification field)
        @test :sddm ∈ classifications
        # Check that the Laplacian block was pinned
        laplacian_derived = [c for c in meta["components"] if haskey(c, "origin_classification")]
        @test length(laplacian_derived) >= 1
    end

    @testset "delete_original flag" begin
        tmp = mktempdir()
        A = make_spd(4)
        write_matrix(tmp, "del_test", A)
        mtx_path = joinpath(tmp, "del_test", "del_test.mtx")
        @test isfile(mtx_path)

        A_read = MatrixMarket.mmread(mtx_path)

        # Without flag: original preserved
        decompose_and_cache(A_read, "del_test", tmp; delete_original=false)
        @test isfile(mtx_path)

        # Clean up cache and redo with flag
        rm(joinpath(tmp, "del_test", "components"); recursive=true)
        decompose_and_cache(A_read, "del_test", tmp; delete_original=true)
        @test !isfile(mtx_path)
    end

    @testset "matrix_type metadata recorded" begin
        tmp = mktempdir()
        A = make_spd(3)
        meta = decompose_and_cache(A, "mt_test", tmp; matrix_type=:adjacency)
        @test meta["matrix_type"] == "adjacency"

        rm(joinpath(tmp, "mt_test", "components"); recursive=true)
        meta2 = decompose_and_cache(A, "mt_test", tmp; matrix_type=:auto)
        @test meta2["matrix_type"] == "auto"
    end

    @testset "reorderings are cached" begin
        tmp = mktempdir()
        A = make_spd(10)
        meta = decompose_and_cache(A, "reord_test", tmp)
        comp = meta["components"][1]

        @test haskey(comp, "reorderings")
        reord = comp["reorderings"]
        @test haskey(reord, "rcm")
        @test haskey(reord, "amd")
        @test haskey(reord, "metis")

        reorder_dir = joinpath(tmp, "reord_test", "components", "reorderings")
        for (_, filename) in reord
            @test isfile(joinpath(reorder_dir, filename))
        end
    end

    @testset "AMD is skipped above the configured nnz limit" begin
        tmp = mktempdir()
        A = make_spd(10)
        meta = decompose_and_cache(A, "amd_skip", tmp; amd_max_nnz=0)
        comp = only(meta["components"])

        @test meta["amd_max_nnz"] == 0
        @test !haskey(comp["reorderings"], "amd")
        @test haskey(comp["reorderings"], "rcm")
        @test haskey(comp["reorderings"], "metis")
        @test comp["reordering_skips"]["amd"]["reason"] == "nnz_limit"
        @test comp["reordering_skips"]["amd"]["nnz"] == nnz(A)
        @test comp["reordering_skips"]["amd"]["max_nnz"] == 0
        @test comp["eigenbounds_policy"]["target"] == "upper"
        @test comp["eigenbounds_policy"]["lower_tolerance"] === nothing
        @test comp["lambda_min"] === nothing
        @test comp["lambda_min_status"] == "not_estimated"
        @test comp["lambda_max_status"] == "ritz_converged"
        @test comp["quadrature_policy"]["scaled_max_iters"] == 30
        @test comp["quadrature_policy"]["unscaled_max_iters"] == 60
        for q in comp["quadrature"]
            @test q["quadrature_k_final"] <= 60
            @test q["scaled_quadrature_k_final"] <= 30
            @test q["quadrature_k100"] === nothing
            @test q["scaled_quadrature_k100"] === nothing
        end
        @test comp["eigenbounds_policy"]["upper_tolerance"] == 1e-3
        @test comp["eigenbounds_policy"]["max_restarts"] <= 100

        @test_throws ErrorException load_reordering("amd_skip", tmp, 1, "amd")
        @test isempty(describe_components(
            "amd_skip"; base_dir=tmp, reordering="amd",
        ))
        @test length(describe_components("amd_skip"; base_dir=tmp, reordering="natural")) == 1
    end

    @testset "component preparation resumes from durable phase checkpoints" begin
        tmp = mktempdir()
        A = make_spd(12)
        checkpoint_path = joinpath(tmp, "resume_prepare", "components",
                                   "prepare_state", "c1.json")

        first_progress = Dict{String,Any}[]
        crash_after_amd = event -> begin
            event["phase"] == "ordering_amd" && error("intentional prepare interruption")
        end
        @test_throws ErrorException decompose_and_cache(
            copy(A), "resume_prepare", tmp;
            prepare_checkpoint_callback = crash_after_amd,
            prepare_progress_callback = event -> push!(first_progress, copy(event)),
        )
        @test any(get(event, "subphase", nothing) == "ordering_amd" &&
                  event["subphase_event"] == "start" for event in first_progress)
        @test any(get(event, "subphase", nothing) == "ordering_amd" &&
                  event["subphase_event"] == "complete" for event in first_progress)
        @test isfile(checkpoint_path)
        @test !isfile(joinpath(tmp, "resume_prepare", "components", "metadata.json"))
        checkpoint = JSON3.read(read(checkpoint_path, String))
        @test haskey(checkpoint["phases"], "unscaled_eigenbounds")
        @test haskey(checkpoint["phases"], "ordering_rcm")
        @test haskey(checkpoint["phases"], "ordering_amd")
        @test !haskey(checkpoint["phases"], "ordering_metis")
        @test all(!isabspath(String(path))
                  for phase in values(checkpoint["phases"])
                  for path in phase["artifacts"])

        resumed_phases = String[]
        resume_progress = Dict{String,Any}[]
        meta = decompose_and_cache(
            copy(A), "resume_prepare", tmp;
            prepare_checkpoint_callback = event -> push!(resumed_phases, event["phase"]),
            prepare_progress_callback = event -> push!(resume_progress, copy(event)),
        )
        @test meta["num_components"] == 1
        @test resumed_phases == [
            "ordering_metis", "scaled_analysis", "scaled_eigenbounds",
            "quadrature", "component_metadata",
        ]
        @test any(get(event, "subphase", nothing) == "ordering_amd" &&
                  event["subphase_event"] == "resumed" for event in resume_progress)
        @test any(get(event, "subphase", nothing) == "ordering_metis" &&
                  event["subphase_event"] == "start" for event in resume_progress)

        # A missing phase artifact invalidates and rebuilds only that phase and
        # its final metadata consumer, not unrelated expensive phases.
        rcm_file = meta["components"][1]["reorderings"]["rcm"]
        rm(joinpath(tmp, "resume_prepare", "components", "reorderings", rcm_file))
        repaired_phases = String[]
        repaired = decompose_and_cache(
            copy(A), "resume_prepare", tmp;
            prepare_checkpoint_callback = event -> push!(repaired_phases, event["phase"]),
        )
        @test repaired["num_components"] == 1
        @test repaired_phases == ["ordering_rcm", "component_metadata"]

        # Reusing a checkpoint for changed numeric input is rejected rather
        # than silently attaching stale spectral payloads.
        changed = copy(A)
        changed.nzval[1] = nextfloat(changed.nzval[1])
        @test_throws ErrorException decompose_and_cache(
            changed, "resume_prepare", tmp)
    end
end

# ─── 4. describe_components ─────────────────────────────────────────

@testset "describe_components" begin
    @testset "cache miss — decomposes from .mtx" begin
        tmp = mktempdir()
        A = make_spd(5)
        write_matrix(tmp, "desc_miss", A)

        descs = describe_components("desc_miss"; base_dir=tmp)
        @test length(descs) == 1
        @test descs[1].dimension == 5
        @test descs[1].classification == :sddm
    end

    @testset "cache hit — no re-decomposition" begin
        tmp = mktempdir()
        A = make_spd(5)
        write_matrix(tmp, "desc_hit", A)

        descs1 = describe_components("desc_hit"; base_dir=tmp)
        # Remove the .mtx — should still work from cache
        rm(joinpath(tmp, "desc_hit", "desc_hit.mtx"))
        descs2 = describe_components("desc_hit"; base_dir=tmp)

        @test length(descs1) == length(descs2)
        @test descs1[1].dimension == descs2[1].dimension
        @test descs1[1].lambda_max == descs2[1].lambda_max
    end

    @testset "no .mtx and no cache — error" begin
        tmp = mktempdir()
        mkpath(joinpath(tmp, "nonexistent"))
        @test_throws ErrorException describe_components("nonexistent"; base_dir=tmp)
    end

    @testset "delete_original forwarded" begin
        tmp = mktempdir()
        A = make_spd(4)
        write_matrix(tmp, "desc_del", A)
        mtx_path = joinpath(tmp, "desc_del", "desc_del.mtx")
        @test isfile(mtx_path)

        describe_components("desc_del"; base_dir=tmp, delete_original=true)
        @test !isfile(mtx_path)
    end

    @testset "reordering is stored in descriptor" begin
        tmp = mktempdir()
        A = make_spd(10)
        write_matrix(tmp, "desc_reord", A)

        descs = describe_components("desc_reord"; base_dir=tmp, reordering="rcm")
        @test descs[1].reordering == "rcm"
    end

    @testset "provenance fields populated for Laplacian components" begin
        tmp = mktempdir()
        L, _ = make_path_laplacian(6)
        write_matrix(tmp, "desc_prov", L)

        descs = describe_components("desc_prov"; base_dir=tmp)
        for d in descs
            @test d.origin_classification == :graph_laplacian
            @test d.prolific_node !== nothing
            @test d.prolific_criterion !== nothing
            @test d.origin_component !== nothing
            @test d.sub_component_index !== nothing
            @test d.sub_component_count !== nothing
        end
    end

    @testset "provenance fields are nothing for SPD components" begin
        tmp = mktempdir()
        A = make_spd(5)
        write_matrix(tmp, "desc_noprov", A)

        descs = describe_components("desc_noprov"; base_dir=tmp)
        @test descs[1].origin_classification === nothing
        @test descs[1].prolific_node === nothing
        @test descs[1].origin_component === nothing
    end
end

# ─── 5. matrix_type override ────────────────────────────────────────

@testset "matrix_type override" begin
    @testset ":adjacency forces adjacency treatment (self-loops)" begin
        tmp = mktempdir()
        # Adjacency with self-loops — auto won't detect as adjacency
        W = sparse([2.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 0.5])
        W = (W + W') .* 0.5
        @test classify_component(W) != :adjacency

        write_matrix(tmp, "selfloop_override", W)
        descs = describe_components("selfloop_override"; base_dir=tmp, matrix_type=:adjacency)
        for d in descs
            @test d.classification != :adjacency
            @test d.classification != :graph_laplacian
            @test d.origin_classification == :graph_laplacian
        end
    end

    @testset ":laplacian forces pinning" begin
        tmp = mktempdir()
        L, _ = make_path_laplacian(5)
        write_matrix(tmp, "lap_override", L)

        descs = describe_components("lap_override"; base_dir=tmp, matrix_type=:laplacian)
        for d in descs
            @test d.classification != :graph_laplacian
            @test d.origin_classification == :graph_laplacian
        end
    end

    @testset ":auto does not pin SPD" begin
        tmp = mktempdir()
        A = make_spd(5)
        write_matrix(tmp, "auto_spd", A)

        descs = describe_components("auto_spd"; base_dir=tmp, matrix_type=:auto)
        @test length(descs) == 1
        @test descs[1].classification == :sddm
        @test descs[1].origin_classification === nothing
    end

    @testset ":adjacency on negative-weight graph — pinned submatrix happens to be SDDM" begin
        tmp = mktempdir()
        # W has negative weight on edge (1,2). The Laplacian L has positive
        # off-diag at (1,2)/(2,1) → L is general_sdd, not M-matrix.
        # Pinning node 2 removes the positive off-diag entries, so the
        # remaining 2×2 submatrix [4 -1; -1 3] is coincidentally SDDM.
        W = sparse([0.0 -3.0 1.0; -3.0 0.0 2.0; 1.0 2.0 0.0])
        write_matrix(tmp, "neg_adj", W)

        descs_auto = describe_components("neg_adj"; base_dir=tmp)
        for d in descs_auto
            @test d.origin_classification == :graph_laplacian
        end

        # Force adjacency should produce same result
        rm(joinpath(tmp, "neg_adj", "components"); recursive=true)
        descs_forced = describe_components("neg_adj"; base_dir=tmp, matrix_type=:adjacency)
        for d in descs_forced
            @test d.origin_classification == :graph_laplacian
        end
    end

    @testset ":adjacency on negative-weight graph — pinned submatrix retains positive off-diags" begin
        tmp = mktempdir()
        # 4-node graph where negative edges between nodes 2-3 and 2-4
        # ensure positive off-diagonals survive in the pinned submatrix
        # regardless of which node is pinned.
        W = sparse([
            0.0  1.0   1.0   1.0;
            1.0  0.0  -1.0  -1.0;
            1.0 -1.0   0.0   1.0;
            1.0 -1.0   1.0   0.0
        ])
        L = adjacency_to_laplacian(W)

        # The Laplacian should have positive off-diagonals (from negative weights)
        @test !is_m_matrix(L)

        write_matrix(tmp, "neg_survive", W)
        descs = describe_components("neg_survive"; base_dir=tmp)
        for d in descs
            @test d.origin_classification == :graph_laplacian
            # Pinned submatrix should NOT be SDDM since positive off-diags survive
            @test d.classification != :sddm
            @test d.classification ∈ (:general_sdd, :symmetric, :symmetric_indefinite)
        end
    end

    @testset "invalid matrix_type rejected" begin
        tmp = mktempdir()
        A = make_spd(3)
        @test_throws AssertionError decompose_and_cache(A, "bad_type", tmp; matrix_type=:nonsense)
    end
end

# ─── 6. prepare_system ──────────────────────────────────────────────

@testset "prepare_system" begin
    @testset "basic load and scale" begin
        tmp = mktempdir()
        A = make_spd(10)
        write_matrix(tmp, "prep_basic", A)

        descs = describe_components("prep_basic"; base_dir=tmp)
        A_out, seed, b = prepare_system(descs[1])

        @test size(A_out, 1) == 10
        @test length(seed) == 10
        @test length(b) == 10
    end

    @testset "reordering applied" begin
        tmp = mktempdir()
        A = make_spd(10)
        write_matrix(tmp, "prep_reord", A)

        descs_nat = describe_components("prep_reord"; base_dir=tmp, reordering="natural")
        descs_rcm = describe_components("prep_reord"; base_dir=tmp, reordering="rcm")

        A_nat, _, _ = prepare_system(descs_nat[1])
        A_rcm, _, _ = prepare_system(descs_rcm[1])

        # Same eigenvalues (same matrix, different ordering)
        eigs_nat = sort(eigvals(Matrix(A_nat)))
        eigs_rcm = sort(eigvals(Matrix(A_rcm)))
        @test eigs_nat ≈ eigs_rcm atol=1e-10
    end

    @testset "pinned Laplacian components are SPD" begin
        tmp = mktempdir()
        L, _ = make_path_laplacian(8)
        write_matrix(tmp, "prep_lap", L)

        descs = describe_components("prep_lap"; base_dir=tmp)
        for d in descs
            A_out, _, _ = prepare_system(d)
            # Should be positive definite
            eigs = eigvals(Matrix(A_out))
            @test all(eigs .> -1e-12)
        end
    end

    @testset "scaling decision respects SDD status" begin
        tmp = mktempdir()
        A = make_spd(5)
        write_matrix(tmp, "prep_sdd", A)

        descs = describe_components("prep_sdd"; base_dir=tmp)
        d = descs[1]
        A_out, seed, b = prepare_system(d)

        if d.scaled_is_sdd || !d.unscaled_is_sdd
            # Should have been scaled — diagonal should be all 1.0
            for i in 1:size(A_out, 1)
                @test A_out[i, i] ≈ 1.0
            end
        end

        # wants_sdd path: if unscaled is SDD and scaled is not, should not scale
        A_out_sdd, _, _ = prepare_system(d; wants_sdd=true)
        if d.unscaled_is_sdd && !d.scaled_is_sdd
            A_orig = load_component_matrix(d)
            for i in 1:size(A_out_sdd, 1)
                @test A_out_sdd[i, i] ≈ A_orig[i, i]
            end
        end
    end

    @testset "RHS is consistent with matrix" begin
        tmp = mktempdir()
        A = make_spd(10)
        write_matrix(tmp, "prep_rhs", A)

        descs = describe_components("prep_rhs"; base_dir=tmp)
        A_out, seed, b = prepare_system(descs[1])

        # b should be nonzero
        @test norm(b) > 0

        # seed should be nonzero
        @test norm(seed) > 0

        # dimensions must match
        @test length(b) == size(A_out, 1)
        @test length(seed) == size(A_out, 1)
    end
end

# ─── 7. Edge cases ──────────────────────────────────────────────────

@testset "edge cases" begin
    @testset "2×2 Laplacian" begin
        tmp = mktempdir()
        L = sparse([1.0 -1.0; -1.0 1.0])
        write_matrix(tmp, "tiny_lap", L)

        # 2×2 Laplacian: pin one node → single trivial (dim=1) sub-component →
        # batched into a single metadata entry with no descriptor produced.
        descs = describe_components("tiny_lap"; base_dir=tmp)
        @test length(descs) == 0
        for d in descs
            @test d.classification != :graph_laplacian
        end
    end

    @testset "complete graph K5" begin
        tmp = mktempdir()
        n = 5
        W = ones(n, n) - I(n)
        W = sparse(W)
        L = adjacency_to_laplacian(W)
        write_matrix(tmp, "complete5", L)

        descs = describe_components("complete5"; base_dir=tmp)
        # Pinning any node in K5 leaves K4, which is connected
        @test length(descs) == 1
        @test descs[1].dimension == n - 1
    end

    @testset "barbell graph — pinning bridge disconnects" begin
        tmp = mktempdir()
        # Two triangles connected by a single bridge node (node 4)
        # Triangle 1: nodes 1-2-3, Triangle 2: nodes 5-6-7
        # Bridge: node 4, connected to node 3 and node 5
        n = 7
        W = spzeros(n, n)
        # Triangle 1
        for (i, j) in [(1,2), (2,3), (1,3)]
            W[i, j] = 1.0; W[j, i] = 1.0
        end
        # Bridge
        W[3, 4] = 1.0; W[4, 3] = 1.0
        W[4, 5] = 1.0; W[5, 4] = 1.0
        # Triangle 2
        for (i, j) in [(5,6), (6,7), (5,7)]
            W[i, j] = 1.0; W[j, i] = 1.0
        end
        L = adjacency_to_laplacian(W)
        write_matrix(tmp, "barbell", L)

        descs = describe_components("barbell"; base_dir=tmp)
        # Node 4 has degree 2, so it may or may not be the prolific node.
        # Regardless, pinning the prolific node may produce 1 or more components.
        @test length(descs) >= 1
        total_dim = sum(d.dimension for d in descs)
        @test total_dim == n - 1  # one node pinned

        # All should be SPD
        for d in descs
            @test d.classification != :graph_laplacian
            @test d.classification != :adjacency
        end
    end

    @testset "weighted graph with mixed signs" begin
        tmp = mktempdir()
        W = sparse([0.0 -2.0 3.0; -2.0 0.0 1.0; 3.0 1.0 0.0])
        write_matrix(tmp, "mixed_wt", W)

        descs = describe_components("mixed_wt"; base_dir=tmp)
        for d in descs
            @test d.classification != :adjacency
            @test d.classification != :graph_laplacian
        end
        total_dim = sum(d.dimension for d in descs)
        @test total_dim == 2  # 3 nodes, one pinned
    end

    @testset "load_component_matrix round-trip" begin
        tmp = mktempdir()
        A_orig = make_spd(8)
        A_sym = (A_orig + A_orig') .* 0.5  # decompose_and_cache symmetrizes
        write_matrix(tmp, "roundtrip", A_orig)

        descs = describe_components("roundtrip"; base_dir=tmp)
        A_loaded = load_component_matrix(descs[1])
        @test A_loaded ≈ A_sym
    end

    @testset "multiple reorderings produce valid permutations" begin
        tmp = mktempdir()
        A = make_spd(20)
        write_matrix(tmp, "perm_test", A)

        descs = describe_components("perm_test"; base_dir=tmp)
        d = descs[1]
        n = d.dimension

        for ordering in ["rcm", "amd", "metis"]
            perm = load_reordering(d.name, d.base_dir, d.component_index, ordering)
            @test length(perm) == n
            @test sort(perm) == 1:n  # valid permutation
        end
    end
end

@testset "Utilities.estimate_eigbounds accuracy" begin
    # 1D Laplacian (tridiagonal): known eigenvalues λ_j = 2 - 2cos(jπ/(n+1))
    for n in [50, 100, 500, 1000]
        dv = fill(2.0, n)
        ev = fill(-1.0, n - 1)
        A = spdiagm(0 => dv, 1 => ev, -1 => ev)

        true_eigs = eigvals(Matrix(A))
        true_min = true_eigs[1]
        true_max = true_eigs[end]

        est_min, est_max, fma, iters = Utilities.estimate_eigbounds(A)

        # Lanczos with adaptive convergence at tol=1e-2 default; relative
        # error in extreme Ritz values is bounded by ritz_residual / |θ|.
        # Empirically <1e-7 for max (fast convergence at top of spectrum)
        # and <1e-3 for min on these tridiagonal Laplacians.
        @test isapprox(est_max, true_max; rtol=1e-7)
        @test isapprox(est_min, true_min; rtol=1e-3)
        @test est_min > 0
        @test fma > 0
    end

    # Random SPD matrix
    for n in [30, 100]
        rng = StableRNG(42)
        R = randn(rng, n, n)
        M = R' * R + I

        A = sparse(M)
        true_eigs = eigvals(M)
        true_min = true_eigs[1]
        true_max = true_eigs[end]

        est_min, est_max, _, iters = Utilities.estimate_eigbounds(A)

        @test isapprox(est_max, true_max; rtol=1e-7)
        @test isapprox(est_min, true_min; rtol=1e-3)
    end

    # Ill-conditioned diagonal: κ = 10000
    n = 200
    dv = collect(range(1e-4, 1.0, length=n))
    A = spdiagm(0 => dv)
    est_min, est_max, _, iters = Utilities.estimate_eigbounds(A)
    @test isapprox(est_max, 1.0; rtol=1e-10)
    @test isapprox(est_min, 1e-4; rtol=1e-6)
end

println("\n✅ All tests passed!")
