module CliMADatasets

using DataDeps
using HDF5
using NPZ
using MLDatasets
using MLUtils

include("datasets/celeba_hq.jl")
include("datasets/turbulence_2d.jl")
include("datasets/turbulence_2d_low_res.jl")
include("datasets/turbulence_2d_context.jl")
include("datasets/correlated_ou_2d.jl")
include("datasets/correlated_ou_1d.jl")
include("datasets/giorgini_2d.jl")
include("datasets/ks_1d.jl")
include("datasets/mnist_3d.jl")

export CelebAHQ
export Turbulence2D
export Turbulence2DLowRes
export Turbulence2DContext
export CorrelatedOU2D
export CorrelatedOU1D
export Giorgini2D
export MNIST3D
export KuramotoSivashinsky1D

function __init__()
    __init__celeba_hq()
    __init__turbulence_2d()
    __init__turbulence_2d_low_res()
    __init__turbulence_2d_context()
    __init__correlated_ou_2d()
    __init__correlated_ou_1d()
    __init__giorgini_2d()
    __init__ks_1d()
    __init__mnist_3d()
end

end #module
