function __init__celeba_hq()
    DEPNAME = "CelebAHQ"

    celeba_hq_32x32_train_female = "https://caltech.box.com/shared/static/h94hiq9vm7m1dc1aailaslldjmoitfab.hdf5"
    celeba_hq_32x32_train_male = "https://caltech.box.com/shared/static/gqzz6yh98wsjbdiv4cf5kmpv3hkf9hlq.hdf5"
    celeba_hq_32x32_val_male = "https://caltech.box.com/shared/static/44hf2gg87w0in2jufq55s5mk6iokatdy.hdf5"
    celeba_hq_32x32_val_female = "https://caltech.box.com/shared/static/0azdg1ss4871h8o170h1zx8lnpdtbvua.hdf5"

    celeba_hq_64x64_train_female = "https://caltech.box.com/shared/static/g5yw9byvylb1db8c6bv76fdu9mcnqlcn.hdf5"
    celeba_hq_64x64_train_male = "https://caltech.box.com/shared/static/wilbz7y64inq1v0pfj0okxjvyctcby5q.hdf5"
    celeba_hq_64x64_val_male = "https://caltech.box.com/shared/static/kgf4h5h1npa4a5xpc9uc2isoqi1mszkx.hdf5"
    celeba_hq_64x64_val_female = "https://caltech.box.com/shared/static/6816j46mhhnfl365k0hgbozksbw4y64q.hdf5"

    celeba_hq_128x128_train_female = "https://caltech.box.com/shared/static/1equ12ge7c4qg1u7h4x5fvqaijwxbur1.hdf5"
    celeba_hq_128x128_train_male = "https://caltech.box.com/shared/static/zmvhpql2uazqlpl31c5kysx4plbbbx91.hdf5"
    celeba_hq_128x128_val_male = "https://caltech.box.com/shared/static/q6pr6ch603e9p0ugbkkg2gvt7l7ne189.hdf5"
    celeba_hq_128x128_val_female = "https://caltech.box.com/shared/static/8yqzybwqcgt2s9azxahi4zycvjs1x7jn.hdf5"

    celeba_hq_256x256_train_female = "https://caltech.box.com/shared/static/9oqdtqu0zugbwh78rxgj1wdwnuocxbih.hdf5"
    celeba_hq_256x256_train_male = "https://caltech.box.com/shared/static/49eskrayxpfm9xcj6nfxakm6iz4sw8y0.hdf5"
    celeba_hq_256x256_val_male = "https://caltech.box.com/shared/static/wdnwwv75gs7g9b8pj604jyr9vnj76smb.hdf5"
    celeba_hq_256x256_val_female = "https://caltech.box.com/shared/static/rzadbhcgsp3xh8uhv2s9tf5yj353e6qj.hdf5"

    register(DataDep(
        DEPNAME,
        """
        Dataset: High-quality version of CelebA dataset.
        Authors: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
        Website: https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training
        Disclaimer: CelebAHQ dataset may contain bias. For more information please check Tensorflow.org 
        [here](https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study).

        References:
        [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://arxiv.org/abs/1710.10196)
        """,
        [celeba_hq_32x32_train_female, celeba_hq_32x32_train_male, celeba_hq_32x32_val_male, celeba_hq_32x32_val_female, 
        celeba_hq_64x64_train_female, celeba_hq_64x64_train_male, celeba_hq_64x64_val_male, celeba_hq_64x64_val_female, 
        celeba_hq_128x128_train_female, celeba_hq_128x128_train_male, celeba_hq_128x128_val_male, celeba_hq_128x128_val_female, 
        celeba_hq_256x256_train_female, celeba_hq_256x256_train_male, celeba_hq_256x256_val_male, celeba_hq_256x256_val_female],
    ))
end

"""
    CelebAHQ(; split=:train, resolution=32, dir=nothing)
High-quality version of CelebA dataset.
Authors: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
Website: https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training
"""
struct CelebAHQ <: MLDatasets.UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    resolution::Int
    gender::Symbol
    features::Array{}
end

CelebAHQ(; split=:train, resolution=32, gender=:female, Tx=Float32, dir=nothing) = CelebAHQ(Tx, resolution, split, gender; dir)
CelebAHQ(split::Symbol; kws...) = CelebAHQ(; split, kws...)
CelebAHQ(resolution::Symbol; kws...) = CelebAHQ(; resolution, kws...)
CelebAHQ(Tx::Type; kws...) = CelebAHQ(; Tx, kws...)

function CelebAHQ(Tx::Type, resolution::Int, split::Symbol, gender::Symbol; dir=nothing)
    DEPNAME = "CelebAHQ"

    @assert split ∈ (:train, :test)
    @assert gender ∈ (:male, :female)

    if resolution in 2 .^ (5:8)
        newsplit = split == :test ? :val : split
        HDF5FILE = "celeba_hq_$(resolution)x_$(resolution)_$(newsplit)_$(gender).hdf5"
    else
        error("$resolution x $resolution not supported for CelebA-HQ dataset.")
    end

    # local path extraction
    features_path = MLDatasets.datafile(DEPNAME, HDF5FILE, dir)

    # loading
    fid = h5open(features_path, "r")
    features = read(fid)["imgs"]
    close(fid)

    # useful side information
    metadata = Dict{String,Any}()
    metadata["n_observations"] = size(features)[end]

    return CelebAHQ(metadata, split, resolution, gender, Tx.(features))
end
