module CliMADatasets

using DataDeps
using HDF5
using MLDatasets
using MLUtils

include("datasets/celeba_hq.jl")
include("datasets/turbulence_2d.jl")
include("datasets/turbulence_2d_context.jl")

export CelebAHQ
export Turbulence2D
export Turbulence2DContext

function __init__()
    __init__celeba_hq()
    __init__turbulence_2d()
    __init__turbulence_2d_context()
end

end #module
