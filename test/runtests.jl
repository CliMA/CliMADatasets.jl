using CliMADatasets
using Test

@testset "CliMADatasets.jl" begin
    @testset "Turbulence2D" begin
        n_features = (512, 512)

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
