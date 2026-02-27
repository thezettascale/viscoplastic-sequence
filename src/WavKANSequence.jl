module WavKANSequence

using Lux
using Lux: Training
using NNlib
using Optimisers
using Random: AbstractRNG
using Reactant: @compile, AutoEnzyme

include("utils.jl")
include("config.jl")
include("wavelets/wavelets.jl")
include("layers/layers.jl")
include("models/models.jl")
include("pipeline/pipeline.jl")

export load_config, create_model, get_visco_loader, train_epoch
export loss_fcn, BIC, log_csv
export MinMaxNormaliser, UnitGaussianNormaliser, encode, decode
export RNOConfig, KANRNOConfig, TransformerConfig, KANTransformerConfig
export AutoEnzyme

end
