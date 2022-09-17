module CliMADatasets



export getobs, numobs # From MLUtils.jl



include("datasets/turbulence_2d.jl")
export Turbulence2D

function __init__()
    __init__turbulence_2d()
end

end #module