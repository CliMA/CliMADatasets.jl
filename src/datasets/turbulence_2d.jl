function __init__turbulence_2d()
    DEPNAME = "Turbulence2D"
    LORES_TRAINFILE = "en_ewt-ud-train.conllu"
    LORES_TESTFILE = "en_ewt-ud-test.conllu"
    HIRES_TRAINFILE = "en_ewt-ud-train.conllu"
    HIRES_TESTFILE = "en_ewt-ud-test.conllu"

    register(DataDep(
        DEPNAME,
        """
        Dataset: Universal Dependencies - English Dependency Treebank Universal Dependencies English Web Treebank
        Authors: Climate Modeling Alliance
        Website: https://github.com/UniversalDependencies/UD_English-EWT
        The files are available for download at the github
        repository linked above. Note that using the data
        responsibly and respecting copyright remains your
        responsibility. Copyright and License is discussed in
        detail on the Website.
        """,
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/" .* [HIRES, LORES],
        "4f645242cc985ca59e744e3aebfb3d9e86507babd773ff3226c0bfeb3d0a52c3"
    ))
end

"""
    Turbulence2D(; split=:train, resolution=:high, dir=nothing)
    UD_English(split=; [dir])
A Gold Standard Universal Dependencies Corpus for
English, built over the source material of the
English Web Treebank LDC2012T13
(https://catalog.ldc.upenn.edu/LDC2012T13).
The corpus comprises 254,825 words and 16,621 sentences, 
taken from five genres of web media: weblogs, newsgroups, emails, reviews, and Yahoo! answers. 
See the LDC2012T13 documentation for more details on the sources of the sentences. 
The trees were automatically converted into Stanford Dependencies and then hand-corrected to Universal Dependencies. 
All the basic dependency annotations have been single-annotated, a limited portion of them have been double-annotated, 
and subsequent correction has been done to improve consistency. Other aspects of the treebank, such as Universal POS, 
features and enhanced dependencies, has mainly been done automatically, with very limited hand-correction.
Authors: Climate Modeling Alliance
"""
struct Turbulence2D <: UnsupervisedDataset
    metadata::Dict{String,Any}
    split::Symbol
    features::Array{}
end

Turbulence2D(; split=:train, resolution=:high, Tx=Float32, dir=nothing) = Turbulence2D(Tx, resolution, split; dir)
Turbulence2D(split::Symbol; kws...) = Turbulence2D(; split, kws...)
Turbulence2D(resolution::Symbol; kws...) = Turbulence2D(; resolution, kws...)
Turbulence2D(Tx::Type; kws...) = Turbulence2D(; Tx, kws...)

function Turbulence2D(Tx::Type, resolution::Symbol, split::Symbol; dir=nothing)
    DEPNAME = "Turbulence2D"
    LORES_TRAINFILE = "en_ewt-ud-train.conllu"
    LORES_TESTFILE = "en_ewt-ud-test.conllu"
    HIRES_TRAINFILE = "en_ewt-ud-train.conllu"
    HIRES_TESTFILE = "en_ewt-ud-test.conllu"

    @asser resolution ∈ [:high, :low]
    @assert split ∈ [:train, :test]

    if split == :train
        if resolution == :high
            FILE = HIRES_TRAINFILE
        elseif resolution == :low 
            FILE = LORES_TRAINFILE
        else
            error()
        end
    elseif split == :test
        if resolution == :high
            FILE = HIRES_TESTFILE
        elseif resolution == :low 
            FILE = LORES_TESTFILE
        else
            error()
        end
    else
        error()
    end


    path = datafile(DEPNAME, FILE, dir)

    doc = []
    sent = []
    lines = open(readlines, path)
    for line in lines
        line = chomp(line)
        if length(line) == 0
            length(sent) > 0 && push!(doc, sent)
            sent = []
        elseif line[1] == '#' # comment line
            continue
        else
            items = Vector{String}(Base.split(line, '\t'))
            push!(sent, items)
        end
    end
    length(sent) > 0 && push!(doc, sent)
    T = typeof(doc[1][1])
    features = Vector{Vector{T}}(doc)

    metadata = Dict{String,Any}("n_observations" => length(features))
    return Turbulence2D(metadata, split, features)
end