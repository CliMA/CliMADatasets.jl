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
    
    @testset "CorrelatedOU2D" begin
        d = CorrelatedOU2D(:train; f = 0.5)
        @test d.split == :train
        @test d.resolution == 32
        @test size(d[:])[1:2] == (32, 32)
        @test d.features isa Array{Float32}
        @test size(d.features)[end] == Int((1e6-100)/2*0.8)
    end

    @testset "CorrelatedOU1D" begin
        d = CorrelatedOU1D(:test; f = 0.1)
        @test d.split == :test
        @test d.n_pixels == 64
        @test d.n_time == 64
        @test size(d[:])[1:2] == (64, 64)
        @test d.features isa Array{Float32}
        @test size(d.features)[end] == Int(3e4*0.1*0.2)
    end

    @testset "Giorgini2D" begin
        for res in [8, 16, 32]
            for nl in [:strong, :medium, :weak]
                d = Giorgini2D(:train; f = 0.5, resolution = res, nonlinearity = nl)
                @test d.split == :train
                @test d.resolution == res
                @test size(d[:])[1:2] == (res, res)
                @test d.features isa Array{Float32}
                @test size(d.features)[end] == 12000 # 80% training data of 50% of total dataset
            end
        end
    end

    @testset "Turbulence2DLowRes" begin
        for res in [32]
            for nl in [:strong, :medium, :weak]
                d = Turbulence2DLowRes(:train; f = 0.5, resolution = res, nonlinearity = nl)
                @test d.split == :train
                @test d.resolution == res
                @test size(d[:])[1:2] == (res, res)
                @test d.features isa Array{Float32}
                @test size(d.features)[end] == 12061 # 80% training data of 50% of total dataset
            end
        end
    end

    @testset "KuramotoSivashinsky1D" begin
        d = KuramotoSivashinsky1D(:train; f = 0.5)
        @test d.split == :train
        @test d.n_pixels == d.n_time == 128
        @test size(d[:])[1:2] == (res, res)
        @test d.features isa Array{Float32}
        @test size(d.features)[end-1] = 1
        @test size(d.features)[end] == 11997 # 80% training data of 50% of total dataset
    end
    
    @testset "MNIST3D" begin
        res = 64
        d = MNIST3D(:train; f = 0.5)
        @test d.split == :train
        @test size(d[:])[1:2] == (res, res)
        @test d.features isa Array{Float32}
        @test size(d.features)[end] == 4000 # 80% training data of 50% of total dataset
    end
end
