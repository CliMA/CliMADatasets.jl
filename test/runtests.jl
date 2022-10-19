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
end
