function __init__turbulence_2d_low_res()
    DEPNAME = "Turbulence2DLowRes"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Two-dimensional images generated via a forced-dissipative turbulence simulation with
        passive idealized condensation tracer.
        Authors: Climate Modeling Alliance
        Website: N/A
        """,
        "https://caltech.box.com/shared/static/4kn6b5g68chgti5adho05jyan5184opc.hdf5",
    ))
end

"""
    Turbulence2DLowRes <: MLDatasets.UnsupervisedDataset

Two-dimensional images generated via a forced-dissipative turbulence simulation with
passive tracer.

Due to correlations in time, all images are only approximately independent.

Authors: Climate Modeling Alliance
"""
struct Turbulence2DLowRes <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Int
    features::Array{}
end

"""
Creates the Turbulence2DLowRes dataset given
- split: :train or :test
- f: fraction of data desired
- Tx: the float type. The dataset was created using Float32
- resolution: the resolution of the images in the dataset (only 32 is supported currently.)
"""
function Turbulence2DLowRes(split::Symbol; f = 1.0, Tx = Float32, resolution = 32, nonlinearity = :strong)
    DEPNAME = "Turbulence2DLowRes"
    HDF5FILE = "two_dimensional_turbulence_with_condensation.hdf5"

    # checks
    @assert resolution ∈ [32]
    @assert nonlinearity ∈ [:weak, :medium, :strong]
    @assert split ∈ [:train, :test]

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, nothing)

    # loading
    N = resolution
    fid = h5open(features_path, "r")
    if nonlinearity == :weak
        τc = 1.0
        ν, nν = 1e-6, 4
        μ, nμ = 1e-2, 0
        ε = 0.1
        γ₀ = 1.0
        e = 1.0
        dt = 0.5e-1
        dt_save = 10.0
        τ = 10.0
        features = read(fid, "$(n)_$(ν)_$(nν)_$(μ)_$(nμ)_$(ε)_$(γ₀)_$(e)_$(τc)_$(dt)_$(dt_save)")
    elseif nonlinearity == :medium
        τc = 0.1
        ν, nν = 1e-6, 4
        μ, nμ = 1e-2, 0
        ε = 0.1
        γ₀ = 1.0
        e = 1.0
        dt = 0.5e-1
        dt_save = 10.0
        τ = 10.0
        features = read(fid, "$(n)_$(ν)_$(nν)_$(μ)_$(nμ)_$(ε)_$(γ₀)_$(e)_$(τc)_$(dt)_$(dt_save)")
    elseif nonlinearity == :strong
        τc = 0.01
        ν, nν = 1e-6, 4
        μ, nμ = 1e-2, 0
        ε = 0.1
        γ₀ = 1.0
        e = 1.0
        dt = 0.5e-1
        dt_save = 10.0
        τ = 10.0
        features = read(fid, "$(n)_$(ν)_$(nν)_$(μ)_$(nμ)_$(ε)_$(γ₀)_$(e)_$(τc)_$(dt)_$(dt_save)")
    end
    close(fid)
    
    n_observations = size(features)[end]
    n_data = Int(round(n_observations*f))
    features = features[:,:,1:1:n_data]
    
    # splitting
    if split == :train
        features, _ = MLUtils.splitobs(features, at=0.8)
    elseif split == :test
        _, features = MLUtils.splitobs(features, at=0.8)
    end
    
    # useful side information
    metadata = Dict{String,Any}()
    metadata["n_data"] = n_data
    metadata["τc"] = τc
    metadata["ν"] = ν
    metadata["nν"] = nν
    metadata["μ"] = μ
    metadata["nμ"] = nμ
    metadata["ε"] = ε
    metadata["γ₀"] = γ₀
    metadata["e"] = e
    metadata["dt"] = dt
    metadata["dt_save"] = dt_save
    metadata["τ"] = τ

    return Turbulence2DLowRes(metadata, split, resolution, Tx.(features))
end
