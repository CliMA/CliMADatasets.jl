module CliMADatasets

using DataDeps
using HDF5
using MLDatasets
using MLUtils

include("datasets/turbulence_2d.jl")
export Turbulence2D

function __init__()
    __init__turbulence_2d()
end

end #module