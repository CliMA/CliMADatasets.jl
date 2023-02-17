function __init__turbulence_2d_context()
    DEPNAME = "Turbulence2DContext"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Two-dimensional turbulence with moisture field and context.
        Authors: Climate Modeling Alliance
        Website: N/A
        Just a two-dimensional turbulence dataset with context.
        """,
        "https://caltech.box.com/shared/static/jocmnexxzhqrjku9rc32is68ehkcudbg.hdf5",
    ))
end

"""
    Turbulence2DContext(; split=:train, resolution=512, wavenumber=1.0 dir=nothing)
Just a two-dimensional turbulence dataset with context. The wavenumber specifies the context.
Authors: Climate Modeling Alliance
"""
struct Turbulence2DContext{L} <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Int
    features::Array{}
    wavenumber::L
end

Turbulence2DContext(; split=:train, resolution=512, wavenumber=1.0, Tx=Float32, dir=nothing) = Turbulence2DContext(Tx, resolution, wavenumber, split; dir)
Turbulence2DContext(split::Symbol; kws...) = Turbulence2DContext(; split, kws...)
Turbulence2DContext(resolution::Symbol; kws...) = Turbulence2DContext(; resolution, kws...)
Turbulence2DContext(Tx::Type; kws...) = Turbulence2DContext(; Tx, kws...)

function Turbulence2DContext(Tx::Type, resolution::Int, wavenumber::Real, split::Symbol; dir=nothing)
    DEPNAME = "Turbulence2DContext"
    HDF5FILE = "turbulence_2d_context.hdf5"

    # checks
    @assert resolution ∈ [512, 64]
    @assert wavenumber ∈ [1.0, 2.0, 4.0, 8.0, 16.0]
    @assert split ∈ [:train, :test]

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, dir)

    # loading
    fid = h5open(features_path, "r")
    features = read(fid, "$(resolution)x$(resolution)x2_wn$(wavenumber)/fields")
    label = read(fid, "$(resolution)x$(resolution)x2_wn$(wavenumber)/label")
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

    return Turbulence2DContext(metadata, split, resolution, Tx.(features), Tx.(label))
end
