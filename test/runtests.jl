using CliMADatasets
using Test

@testset "CliMADatasets.jl" begin
    @testset "CelebAHQ" begin
        for split in (:train, :test)
            for gender in (:male, :female)
                for resolution in 2 .^ (5:8)
                    d = CelebAHQ(split, resolution=resolution, gender=gender)
                    @test d.split == split
                    @test d.resolution == resolution
                    @test d.gender == gender
                    @test size(d[:])[1:2] == (resolution, resolution)
                end
            end
        end
    end

    @testset "Turbulence2D" begin
        d = Turbulence2D(:train, resolution=:high)
        @test d.split == :train
        @test d.resolution == :high
        @test size(d[:])[1:2] == (512, 512)

        d = Turbulence2D(Float64, resolution=:low)
        @test d.split == :train
        @test d.resolution == :low
        @test d.features isa Array{Float64}

        d = Turbulence2D(split=:test, resolution=:high)
        @test d.split == :test
        @test d.features isa Array{Float32}
        @test size(d[:])[1:2] == (512, 512)
    end

    @testset "Turbulence2DContext" begin
        d = Turbulence2DContext(:train, resolution=512, wavenumber=2.0)
        @test d.split == :train
        @test d.resolution == 512
        @test size(d.features)[3] == 3
        @test size(d[:])[1:2] == (512, 512)

        d = Turbulence2DContext(Float64, resolution=64, wavenumber=1.0)
        @test d.split == :train
        @test d.resolution == 64
        @test size(d.features)[3] == 3
        @test d.features isa Array{Float64}

        d = Turbulence2DContext(split=:test, resolution=512, wavenumber=2.0)
        @test d.split == :test
        @test size(d.features)[3] == 3
        @test d.features isa Array{Float32}
        @test size(d[:])[1:2] == (512, 512)

        d = Turbulence2DContext(split=:test, resolution=512, wavenumber=:all, fraction=0.5)
        @test d.split == :test
        @test size(d.features)[3] == 3
        @test d.features isa Array{Float32}
        @test size(d[:])[1:2] == (512, 512)
        @test size(d[:])[4] == 500
    end    
end
