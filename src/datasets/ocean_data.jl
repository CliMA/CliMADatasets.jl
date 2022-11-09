using Dierckx
using ProgressBars

function __init__ocean_data()
    DEPNAME = "OceanData"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Surface prognostic fields from global ocean simulations.
        Authors: Climate Modeling Alliance
        Website: N/A
        ocean dataset.
        """,
        "https://www.dropbox.com/s/85jh27w54gdyjio/near_global_lat_lon_1440_600_48_fine_surface.jld2?dl=1",
        "https://www.dropbox.com/s/m5gc3b7apbmf7vs/near_global_lat_lon_360_150_48_fine_surface.jld2?dl=1",
    ))

    datadep"OceanData"
end

"""
    OceanData(; split=:train, degree=0.25, dir=nothing)
A two-dimensional ocean surface dataset.
Authors: Climate Modeling Alliance
"""
struct OceanData <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    degree::Number
    features::Array{}
end

function OceanData(Tx::Type=Float32; degree = 0.25, split = :train,
                   pixels=64, patch_center=(-100, -0), ref_degree=0.25, 
                   spline_order = (1, 1), spline_smoothing = 0.0)
    
    if degree == 0.25
        features_path = datadep"OceanData/near_global_lat_lon_1440_600_48_fine_surface.jld2"
    elseif degree == 1
        features_path = datadep"OceanData/near_global_lat_lon_360_150_48_fine_surface.jld2"
    end

    λ = (-179.75:ref_degree:180)
    ϕ =  (-74.75:ref_degree:75)

    λ_deg = (-179.75:degree:180)
    ϕ_deg =  (-74.75:degree:75)

    i = argmin(abs.(λ_deg .- patch_center[1]))
    j = argmin(abs.(ϕ_deg .- patch_center[2]))
    
    λ = λ[i-pixels÷2:i+pixels÷2-1]
    ϕ = ϕ[j-pixels÷2:j+pixels÷2-1]

    kx, ky = spline_order
    s      = spline_smoothing

    # checks
    @assert degree ∈ [0.25, 1.0]
    @assert split  ∈ [:train, :test]
    
    # loading
    fid        = jldopen(features_path)
    iterations = keys(fid["timeseries/t"])[2:end]

    # TODO: add λ, ϕ and bathymetry to the channels
    features = zeros(pixels, pixels, 2, length(iterations))
    
    for (n, iter) in ProgressBar(iterations)
        udata = fid["timeseries/u/" * iter][:, :, 1]
        vdata = 0.5 .* (fid["timeseries/v/" * iter][:, 1:end-1, 1] .+ 
                        fid["timeseries/v/" * iter][:, 2:end, 1])

        uvel = Spline2D(λ_deg, ϕ_deg, udata; kx, ky, s)
        vvel = Spline2D(λ_deg, ϕ_deg, vdata; kx, ky, s)

        for (i, lat) in enumerate(λ), (j, lon) in enumerate(ϕ)
            features[i, j, 1, n] = uvel(lat, lon)
            features[i, j, 2, n] = vvel(lat, lon)
        end
    end
    close(fid)
    
    # splitting
    if split == :train
        features, _ = MLUtils.splitobs(features, at=0.8)
    elseif split == :test
        _, features = MLUtils.splitobs(features, at=0.8)
    end
    
    # useful side information
    metadata = Dict{String,Any}()
    metadata["n_observations"] = size(features)[end]

    return OceanData(metadata, split, degree, Tx.(features))
end
