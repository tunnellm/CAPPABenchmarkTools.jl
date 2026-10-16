using Test, CAPPABenchmarkTools, CodecZstd
const U = CAPPABenchmarkTools.Utilities
@testset "Streaming typed NPY" begin
    mktempdir() do dir
        path = joinpath(dir,"values.npy.zstd")
        for x in (Float64[], [0.0,-0.0,Inf,-Inf,NaN], collect(1.0:300000.0),
                  view(collect(1.0:600000.0),1:2:600000))
            U.write_npy_zstd_1d_f64(path,x)
            @test reinterpret(UInt64,U.read_npy_zstd_1d_f64(path)) == reinterpret(UInt64,collect(x))
        end
        for x in (Int64[], Int64[typemin(Int64),typemax(Int64)], Int32[1,2,3])
            U.write_npy_zstd_1d_i64(path,x)
            @test U.read_npy_zstd_1d_i64(path) == x
        end
        @test_throws Exception U.read_npy_zstd_1d_f64(path)
        U.write_npy_zstd_1d_f64(path, collect(1.0:100.0))
        encoded = read(path)
        decoded = transcode(ZstdDecompressor,encoded)
        for bytes in (encoded[1:end-1], encoded[1:5])
            write(path,bytes)
            @test_throws Exception U.read_npy_zstd_1d_f64(path)
        end
        for bytes in (decoded[1:5],decoded[1:end-8],vcat(decoded,zeros(UInt8,8)))
            write(path,transcode(ZstdCompressor,bytes))
            @test_throws Exception U.read_npy_zstd_1d_f64(path)
        end
        # Same-length header tampering: wrong magic/version.
        for offset in (1,7)
            bytes = copy(decoded); bytes[offset] = 0xff
            write(path,transcode(ZstdCompressor,bytes))
            @test_throws Exception U.read_npy_zstd_1d_f64(path)
        end
        for shape in ("(2, 2)", "($(typemax(Int)),)")
            header = "{'descr': '<f8', 'fortran_order': False, 'shape': $shape, }\n"
            bytes = vcat(UInt8[0x93,0x4e,0x55,0x4d,0x50,0x59,1,0,
                length(header) & 0xff, length(header) >> 8], codeunits(header))
            write(path,transcode(ZstdCompressor,bytes))
            @test_throws Exception U.read_npy_zstd_1d_f64(path)
        end
        x = collect(1.0:1000000.0)
        U.write_npy_zstd_1d_f64(path,x)
        U.read_npy_zstd_1d_f64(path)
        allocated = @allocated U.read_npy_zstd_1d_f64(path)
        @test allocated < sizeof(x) + 2_000_000
        @test (@allocated U.write_npy_zstd_1d_f64(path,x)) < 2_000_000
    end
end
