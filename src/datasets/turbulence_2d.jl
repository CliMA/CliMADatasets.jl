function __init__turbulence_2d()
    DEPNAME = "Turbulence2D"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Two-dimensional turbulence with moisture field.
        Authors: Climate Modeling Alliance
        Website: N/A
        Just a two-dimensional turbulence dataset.
        """,
        "https://caltech.box.com/shared/static/0golc3ynh76v0lnv25xk2dsnxvpgacav.hdf5",
    ))
end

"""
    Turbulence2D(; split=:train, resolution=:high, dir=nothing)
Just a two-dimensional turbulence dataset.
Authors: Climate Modeling Alliance
"""
struct Turbulence2D <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Symbol
    features::Array{}
end

Turbulence2D(; split=:train, resolution=:high, Tx=Float32, dir=nothing) = Turbulence2D(Tx, resolution, split; dir)
Turbulence2D(split::Symbol; kws...) = Turbulence2D(; split, kws...)
Turbulence2D(resolution::Symbol; kws...) = Turbulence2D(; resolution, kws...)
Turbulence2D(Tx::Type; kws...) = Turbulence2D(; Tx, kws...)

function Turbulence2D(Tx::Type, resolution::Symbol, split::Symbol; dir=nothing)
    DEPNAME = "Turbulence2D"
    HDF5FILE = "turbulence_2d_nx512_ny512.hdf5"

    # checks
    @assert resolution ∈ [:high, :low]
    @assert split ∈ [:train, :test]

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, dir)

    # loading
    fid = h5open(features_path, "r")
    if resolution == :low
        features = read(fid, "low_resolution/")
    elseif resolution == :high
        features = read(fid, "high_resolution/")
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

    return Turbulence2D(metadata, split, resolution, Tx.(features))
end
