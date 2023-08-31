function __init__correlated_ou_1d()
    DEPNAME = "CorrelatedOU1D"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: One-dimensional output generated via an OU process with correlated spatial noise.
        Authors: Climate Modeling Alliance
        Website: N/A
        """,
        ["https://caltech.box.com/shared/static/5bpioxl7oj4qw5noxhk4rnfjhm2y1lag.hdf5",],
    ))
end

"""
    CorrelatedOU1D <: MLDatasets.UnsupervisedDataset

A dataset consisting of 1d fields generated via an OU process with correlated spatial noise.
The field is saved every dt_save during the numerical integration, and has not been shuffled
in time.

Authors: Climate Modeling Alliance
"""
struct CorrelatedOU1D <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    n_pixels::Int
    n_time::Int
    features::Array{}
end

"""
Creates the CorrelatedOU1D dataset, images of size n_pixels x n_time, given

- split: :train or :test
- f: fraction of data desired
- Tx: the float type. The dataset was created using Float32
- n_pixels: the spatial extent of the box used in the simulation
            Note that: Δx = 1; so n_pixels is the size of the box.
                       Only 64 is supported currently.
- n_time: the number of time samples to use when creating the 2d ``images".
"""
function CorrelatedOU1D(split::Symbol; f = 1.0, Tx = Float32, n_pixels = 64, n_time = 64)
    # checks
    @assert n_pixels ∈ [64,]
    @assert split ∈ [:train, :test]

    DEPNAME = "CorrelatedOU1D"
    HDF5FILE = "grf_1d.hdf5"

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, nothing)

    # loading
    fid = h5open(features_path, "r")
    # For now, these are our only options. If we expand, they will become kwargs
    σ = 1.0
    θ = 30.0
    dt = 0.01
    dt_save = 1.0
    # Autocorrelation time: specific to dataset
    τ = 16.0
    features = read(fid, string("N_64_$σ","_$θ","_$dt","_$dt_save"))
    close(fid)
    
    n_observations = size(features)[end]
    # compute the number we want to keep based on the fraction of the data we want to use
    n_keep = Int(round(n_observations*f))
    # Cut and reshape into n_pixels x n_time images
    n_data = div(n_keep, n_time, RoundDown)
    features = reshape(features[:,1:n_data*n_time], (n_pixels, n_time, n_data))
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

    return CorrelatedOU1D(metadata, split, n_pixels, n_time, Tx.(features))
end
