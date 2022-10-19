module CliMADatasets

using DataDeps
using HDF5
using MLDatasets
using MLUtils

include("datasets/celeba_hq.jl")
include("datasets/turbulence_2d.jl")

export CelebAHQ
export Turbulence2D

function __init__()
    __init__celeba_hq()
    __init__turbulence_2d()
end

end #module
