function __init__giorgini_2d()
    DEPNAME = "Giorgini2D"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Two-dimensional images generated via a nonlinear SDE based on Giorgini etal. 2024.
        Authors: Climate Modeling Alliance
        Website: N/A
        """,
        "https://caltech.box.com/shared/static/4kn6b5g68chgti5adho05jyan5184opc.hdf5",
    ))
end

"""
    Giorgini2D <: MLDatasets.UnsupervisedDataset

A dataset consisting of images generated via a nonlinear SDE based on Giorgini etal. 2024.

Due to correlations in time, all images are only approximately independent.

Authors: Climate Modeling Alliance
"""
struct Giorgini2D <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Int
    features::Array{}
end

"""
Creates the Giorgini2D dataset given
- split: :train or :test
- f: fraction of data desired
- Tx: the float type. The dataset was created using Float32
- resolution: the resolution of the images in the dataset (only 32 is supported currently.)
"""
function Giorgini2D(split::Symbol; f = 1.0, Tx = Float32, resolution = 32, nonlinearity = :strong)
    DEPNAME = "Giorgini2D"
    HDF5FILE = "nonlinear_giorgini.hdf5"

    # checks
    @assert resolution ∈ [8, 16, 32]
    @assert nonlinearity ∈ [:weak, :medium, :strong]
    @assert split ∈ [:train, :test]

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, nothing)

    # loading
    N = resolution
    fid = h5open(features_path, "r")
    if nonlinearity == :weak
        σ = 2.0
        α = 0.0
        β = 0.5
        γ = 0.1
        dt = 0.5
        dt_save = 100.0
        τ = 100.0
        features = read(fid, string("ludo_res", "_$N", "x$N", "_$σ","_$α","_$β","_$γ","_$dt","_$dt_save"))
    elseif nonlinearity == :medium
        σ = 2.0
        α = 0.1
        β = 0.5
        γ = 1.0
        dt = 0.5
        dt_save = 100.0
        τ = 100.0
        features = read(fid, string("ludo_res", "_$N", "x$N", "_$σ","_$α","_$β","_$γ","_$dt","_$dt_save"))
    elseif nonlinearity == :strong
        σ = 2.0
        α = 0.3
        β = 0.5
        γ = 10.0
        dt = 0.5
        dt_save = 100.0
        τ = 100.0
        features = read(fid, string("ludo_res", "_$N", "x$N", "_$σ","_$α","_$β","_$γ","_$dt","_$dt_save"))
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
    metadata["σ"] = σ
    metadata["α"] = α
    metadata["β"] = β
    metadata["γ"] = γ
    metadata["dt"] = dt
    metadata["dt_save"] = dt_save
    metadata["τ"] = τ

    return Giorgini2D(metadata, split, resolution, Tx.(features))
end
