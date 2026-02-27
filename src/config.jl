using ConfParser

abstract type ModelConfig end


struct RNOConfig <: ModelConfig
    n_hidden::Int
    num_layers::Int
    activation::String
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function RNOConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    return RNOConfig(
        parse(Int, retrieve(conf, "Architecture", "n_hidden")),
        parse(Int, retrieve(conf, "Architecture", "num_layers")),
        retrieve(conf, "Architecture", "activation"),
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


struct KANRNOConfig <: ModelConfig
    n_hidden::Int
    num_layers::Int
    activation::String
    wavelet_names::Vector{String}
    norm::Bool
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function KANRNOConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    num_layers = parse(Int, retrieve(conf, "Architecture", "num_layers"))
    wav_keys = ["wav_one", "wav_two", "wav_three", "wav_four", "wav_five", "wav_six"]
    wavelet_names = [retrieve(conf, "Architecture", k) for k in wav_keys[1:(num_layers + 1)]]
    return KANRNOConfig(
        parse(Int, retrieve(conf, "Architecture", "n_hidden")),
        num_layers,
        retrieve(conf, "Architecture", "activation"),
        wavelet_names,
        parse(Bool, retrieve(conf, "Architecture", "norm")),
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


struct TransformerConfig <: ModelConfig
    d_model::Int
    nhead::Int
    dim_feedforward::Int
    dropout::Float32
    num_encoder_layers::Int
    num_decoder_layers::Int
    max_len::Int
    activation::String
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function TransformerConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    return TransformerConfig(
        parse(Int, retrieve(conf, "Architecture", "d_model")),
        parse(Int, retrieve(conf, "Architecture", "nhead")),
        parse(Int, retrieve(conf, "Architecture", "dim_feedforward")),
        parse(Float32, retrieve(conf, "Architecture", "dropout")),
        parse(Int, retrieve(conf, "Architecture", "num_encoder_layers")),
        parse(Int, retrieve(conf, "Architecture", "num_decoder_layers")),
        parse(Int, retrieve(conf, "Architecture", "max_len")),
        retrieve(conf, "Architecture", "activation"),
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


struct KANTransformerConfig <: ModelConfig
    d_model::Int
    nhead::Int
    dim_feedforward::Int
    dropout::Float32
    num_encoder_layers::Int
    num_decoder_layers::Int
    max_len::Int
    activation::String
    encoder_wavelet_names::Vector{String}
    decoder_wavelet_names::Vector{String}
    output_wavelet::String
    norm::Bool
    learning_rate::Float32
    step_rate::Int
    gamma::Float32
    min_lr::Float32
    batch_size::Int
    num_epochs::Int
    p::Float32
end

function KANTransformerConfig(path::String)
    conf = ConfParse(path)
    parse_conf!(conf)
    n_enc = parse(Int, retrieve(conf, "Architecture", "num_encoder_layers"))
    n_dec = parse(Int, retrieve(conf, "Architecture", "num_decoder_layers"))
    enc_keys = ["wav_one", "wav_two", "wav_three", "wav_four", "wav_five", "wav_six", "wav_seven", "wav_eight"]
    dec_keys = ["wav_one", "wav_two", "wav_three"]
    return KANTransformerConfig(
        parse(Int, retrieve(conf, "Architecture", "d_model")),
        parse(Int, retrieve(conf, "Architecture", "nhead")),
        parse(Int, retrieve(conf, "Architecture", "dim_feedforward")),
        parse(Float32, retrieve(conf, "Architecture", "dropout")),
        n_enc, n_dec,
        parse(Int, retrieve(conf, "Architecture", "max_len")),
        retrieve(conf, "Architecture", "activation"),
        [retrieve(conf, "EncoderWavelets", k) for k in enc_keys[1:n_enc]],
        [retrieve(conf, "DecoderWavelets", k) for k in dec_keys[1:n_dec]],
        retrieve(conf, "OutputWavelet", "wav"),
        parse(Bool, retrieve(conf, "Architecture", "norm")),
        parse(Float32, retrieve(conf, "Optimizer", "learning_rate")),
        parse(Int, retrieve(conf, "Optimizer", "step_rate")),
        parse(Float32, retrieve(conf, "Optimizer", "gamma")),
        parse(Float32, retrieve(conf, "Optimizer", "min_lr")),
        parse(Int, retrieve(conf, "Dataloader", "batch_size")),
        parse(Int, retrieve(conf, "Pipeline", "num_epochs")),
        parse(Float32, retrieve(conf, "Loss", "p")),
    )
end


function load_config(model_name::String)
    configs = Dict(
        "RNO" => () -> RNOConfig("config/rno.ini"),
        "KAN_RNO" => () -> KANRNOConfig("config/kan_rno.ini"),
        "Transformer" => () -> TransformerConfig("config/transformer.ini"),
        "KAN_Transformer" => () -> KANTransformerConfig("config/kan_transformer.ini"),
    )
    return configs[model_name]()
end
