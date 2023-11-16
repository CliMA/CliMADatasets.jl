function __init__mnist_3d()
    DEPNAME = "MNIST3D"
    
    register(DataDep(
        DEPNAME,
        """
        Dataset: Moving MNIST
        Authors: Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov, ICML 2015 
        Website: http://www.cs.toronto.edu/~nitish/unsupervised_video/
        """,
        "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
    ))
end

"""
    MNIST3D <: MLDatasets.UnsupervisedDataset

    Three-dimensional moving MNIST dataset for video problems.

Authors: Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov, ICML 2015
"""
struct MNIST3D <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    features::Array{}
end

"""
Creates the MNIST3D dataset given
- split: :train or :test
- f: fraction of data desired
- Tx: the float type. The dataset was created using Float32
"""
function MNIST3D(split::Symbol; f = 1.0, Tx = Float32)
    DEPNAME = "MNIST3D"
    NPYFILE = "mnist_test_seq.npy"

    # checks
    @assert split ∈ [:train, :test]

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, NPYFILE, nothing)

    # loading
    features = npzread(features_path)
    features = permutedims(features, [3, 4, 1, 2])
    
    n_observations = size(features)[end]
    n_data = Int(round(n_observations*f))
    features = features[:,:,:,1:1:n_data]
    
    # splitting
    if split == :train
        features, _ = MLUtils.splitobs(features, at=0.8)
    elseif split == :test
        _, features = MLUtils.splitobs(features, at=0.8)
    end
    
    # useful side information
    metadata = Dict{String,Any}()
    metadata["n_data"] = n_data

    return MNIST3D(metadata, split, Tx.(features))
end
