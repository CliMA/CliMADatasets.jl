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
struct Turbulence2DContext <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Int
    features::Array{}
end

Turbulence2DContext(; split=:train, resolution=64, wavenumber=1.0, Tx=Float32, fraction=1.0, dir=nothing) = Turbulence2DContext(Tx, resolution, wavenumber, split; fraction, dir)
Turbulence2DContext(split::Symbol; kws...) = Turbulence2DContext(; split, kws...)
Turbulence2DContext(resolution::Symbol; kws...) = Turbulence2DContext(; resolution, kws...)
Turbulence2DContext(Tx::Type; kws...) = Turbulence2DContext(; Tx, kws...)

function Turbulence2DContext(Tx::Type, resolution::Int, wavenumber::Real, split::Symbol; fraction=1.0, dir=nothing)
    DEPNAME = "Turbulence2DContext"
    HDF5FILE = "turbulence_2d_with_context.hdf5"

    # checks
    @assert resolution ∈ [512, 64]
    @assert wavenumber ∈ [1.0, 2.0, 4.0, 8.0, 16.0]
    @assert split ∈ [:train, :test]
    if resolution == 64 && wavenumber != 1.0
        error("resolution $resolution must have wavenumber 1.0 only.")
    end

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, dir)

    # loadingn
    fid = h5open(features_path, "r")
    features = read(fid, "$(resolution)x$(resolution)x2_wn$(wavenumber)/fields")
    label = read(fid, "$(resolution)x$(resolution)x2_wn$(wavenumber)/label")
    close(fid)

    # build and attach context to base version of dataset
    amp = resolution == 512 ? 1.0 : 0.0 # context is zero for low resolution version
    n = resolution
    k = wavenumber
    L = 2π
    xx = ones(n) * LinRange(0,L,n)' 
    yy = LinRange(0,L,n) * ones(n)'
    context = @. amp * sin(2π * k * xx / L) * sin(2π *  k * yy / L)
    context = context[:,:,:,:]
    context = permutedims(context, (1, 2, 4, 3))
    context = repeat(context, inner=(1, 1, 1, size(features)[end]))
    features = cat(features, context, dims=3)

    # take only a fraction of the dataset
    new_nobs = Int(floor(size(features)[end] * fraction))
    features = features[:,:,:,1:new_nobs]

    # splitting
    if split == :train
        features, _ = MLUtils.splitobs(features, at=0.8)
    elseif split == :test
        _, features = MLUtils.splitobs(features, at=0.8)
    end
    # useful side information
    metadata = Dict{String,Any}()
    metadata["n_observations"] = size(features)[end]

    return Turbulence2DContext(metadata, split, resolution, Tx.(features))
end

function Turbulence2DContext(Tx::Type, resolution::Int, wavenumber::Symbol, split::Symbol; fraction=1.0, dir=nothing)
    features = []
    wavenumbers = resolution == 512 ? [1.0, 2.0, 4.0, 8.0, 16.0] : [1.0]
    for k in wavenumbers
        push!(features, Turbulence2DContext(Tx, resolution, k, split, fraction=fraction).features)
    end
    features = cat(features..., dims=4)

    # useful side information
    metadata = Dict{String,Any}()
    metadata["n_observations"] = size(features)[end]

    return Turbulence2DContext(metadata, split, resolution, Tx.(features))
end
