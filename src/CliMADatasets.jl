module CliMADatasets

using DataDeps
using HDF5
using MLDatasets
using MLUtils

include("datasets/celeba_hq.jl")
include("datasets/turbulence_2d.jl")
include("datasets/turbulence_2d_context.jl")
include("datasets/correlated_ou_2d.jl")
include("datasets/correlated_ou_1d.jl")

export CelebAHQ
export Turbulence2D
export Turbulence2DContext
export CorrelatedOU2D
export CorrelatedOU1D

function __init__()
    __init__celeba_hq()
    __init__turbulence_2d()
    __init__turbulence_2d_context()
    __init__correlated_ou_2d()
    __init__correlated_ou_1d()
end

end #module
