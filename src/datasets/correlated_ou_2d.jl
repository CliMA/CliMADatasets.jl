function __init__correlated_ou_2d()
    DEPNAME = "CorrelatedOU2D"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Two-dimensional images generated via an OU process with correlated spatial noise.
        Authors: Climate Modeling Alliance
        Website: N/A
        """,
        "https://caltech.box.com/shared/static/wul8b4ejb07ftj43p1guxmeso2d4mhja.hdf5",
    ))
end

"""
    CorrelatedOU2D <: MLDatasets.UnsupervisedDataset

A dataset consisting of images generated via an OU process with correlated spatial noise.

Due to correlations in time, all images are not independent.

Authors: Climate Modeling Alliance
"""
struct CorrelatedOU2D <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Int
    features::Array{}
end

"""
Creates the CorrelatedOU2D dataset given
- split: :train or :test
- f: fraction of data desired
- Tx: the float type. The dataset was created using Float32
- resolution: the resolution of the images in the dataset (only 32 is supported currently.)
"""
function CorrelatedOU2D(split::Symbol; f = 1.0, Tx = Float32, resolution = 32,)
    DEPNAME = "CorrelatedOU2D"
    HDF5FILE = "grf.hdf5"

    # checks
    @assert resolution ∈ [32,]
    @assert split ∈ [:train, :test]

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, nothing)

    # loading
    fid = h5open(features_path, "r")
    if resolution == 32
        # For now, these are our only options. If we expand, they will become kwargs
        σ = 1.0
        θ = 0.5
        dt = 0.25
        dt_save = 1.0
        # Autocorrelation time: specific to dataset
        τ = 100.0
        features = read(fid, string("res_32x32_$σ","_$θ","_$dt","_$dt_save"))
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
    metadata["θ"] = θ
    metadata["dt"] = dt
    metadata["dt_save"] = dt_save
    metadata["τ"] = τ

    return CorrelatedOU2D(metadata, split, resolution, Tx.(features))
end
