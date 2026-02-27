include("src/WavKANSequence.jl")

using .WavKANSequence
using HyperTuning
using Random
using Lux
using Lux: Training
using Optimisers
using MLDataDevices: reactant_device

const WAVELET_LIST = ["MexicanHat", "DerivativeOfGaussian", "Morlet", "Shannon", "Meyer"]
const ACTIVATIONS = ["relu", "selu", "leakyrelu", "swish", "gelu"]

model_name = length(ARGS) >= 1 ? ARGS[1] : "RNO"
dev = reactant_device()

function run_trial(cfg::ModelConfig, trial)
    train_loader, test_loader = get_visco_loader(cfg.batch_size; dev = dev)
    input_size = size(first(train_loader)[2], 1)
    model = create_model(cfg, input_size)
    ps, st = Lux.setup(Lux.default_rng(), model)
    ps, st = ps |> dev, st |> dev
    train_state = Training.TrainState(model, ps, st, Optimisers.Adam(cfg.learning_rate))
    loss_fn(yp, y) = loss_fcn(yp, y; p = cfg.p)
    test_loss = 0.0
    for epoch in 1:(cfg.num_epochs)
        train_state, _, test_loss = train_epoch(train_state, train_loader, test_loader, loss_fn, model, epoch, cfg)
        report_value!(trial, test_loss)
        should_prune(trial) && return
    end
    test_loss < 100 && report_success!(trial)
    return test_loss
end

function rno_objective(trial)
    Random.seed!(get_seed(trial))
    @suggest n_hidden in trial
    @suggest n_layers in trial
    @suggest activation in trial
    @suggest b_size in trial
    @suggest learning_rate in trial
    @suggest gamma_val in trial
    @suggest step_rate in trial
    cfg = RNOConfig(n_hidden, n_layers, activation, Float32(learning_rate), step_rate, Float32(gamma_val), 1.0f-8, b_size, 50, 2.0f0)
    return run_trial(cfg, trial)
end

function kan_rno_objective(trial)
    Random.seed!(get_seed(trial))
    @suggest n_hidden in trial
    @suggest n_layers in trial
    @suggest activation in trial
    @suggest b_size in trial
    @suggest learning_rate in trial
    @suggest gamma_val in trial
    @suggest step_rate in trial
    @suggest wav_one in trial; @suggest wav_two in trial; @suggest wav_three in trial
    @suggest wav_four in trial; @suggest wav_five in trial; @suggest wav_six in trial
    @suggest layer_norm in trial
    wavelet_names = [wav_one, wav_two, wav_three, wav_four, wav_five, wav_six][1:n_layers]
    cfg = KANRNOConfig(
        n_hidden, n_layers, activation, wavelet_names, layer_norm,
        Float32(learning_rate), step_rate, Float32(gamma_val), 1.0f-4, b_size, 15, 2.0f0
    )
    return run_trial(cfg, trial)
end

function transformer_objective(trial)
    Random.seed!(get_seed(trial))
    @suggest d_model in trial; @suggest nhead in trial; @suggest dim_feedforward in trial
    @suggest dropout in trial; @suggest num_encoder_layers in trial
    @suggest num_decoder_layers in trial; @suggest max_len in trial
    @suggest activation in trial; @suggest b_size in trial
    @suggest learning_rate in trial; @suggest gamma_val in trial; @suggest step_rate in trial
    cfg = TransformerConfig(
        d_model, nhead, dim_feedforward, Float32(dropout),
        num_encoder_layers, num_decoder_layers, max_len, activation,
        Float32(learning_rate), step_rate, Float32(gamma_val), 1.0f-4, b_size, 50, 2.0f0
    )
    return run_trial(cfg, trial)
end

function kan_transformer_objective(trial)
    Random.seed!(get_seed(trial))
    @suggest d_model in trial; @suggest nhead in trial; @suggest dim_feedforward in trial
    @suggest dropout in trial; @suggest num_encoder_layers in trial
    @suggest num_decoder_layers in trial; @suggest max_len in trial
    @suggest activation in trial; @suggest b_size in trial
    @suggest learning_rate in trial; @suggest gamma_val in trial; @suggest step_rate in trial
    @suggest encoder_wav_one in trial; @suggest encoder_wav_two in trial
    @suggest encoder_wav_three in trial; @suggest encoder_wav_four in trial
    @suggest encoder_wav_five in trial; @suggest encoder_wav_six in trial
    @suggest encoder_wav_seven in trial; @suggest encoder_wav_eight in trial
    @suggest decoder_wav_one in trial; @suggest decoder_wav_two in trial
    @suggest decoder_wav_three in trial; @suggest output_wavelet in trial; @suggest norm in trial
    enc_wavs = [
        encoder_wav_one, encoder_wav_two, encoder_wav_three, encoder_wav_four,
        encoder_wav_five, encoder_wav_six, encoder_wav_seven, encoder_wav_eight,
    ][1:num_encoder_layers]
    dec_wavs = [decoder_wav_one, decoder_wav_two, decoder_wav_three][1:num_decoder_layers]
    cfg = KANTransformerConfig(
        d_model, nhead, dim_feedforward, Float32(dropout),
        num_encoder_layers, num_decoder_layers, max_len, activation,
        enc_wavs, dec_wavs, output_wavelet, norm,
        Float32(learning_rate), step_rate, Float32(gamma_val), 1.0f-3, b_size, 15, 2.0f0
    )
    return run_trial(cfg, trial)
end

spaces = Dict(
    "RNO" => (
        rno_objective, Scenario(
            n_hidden = 2:20, n_layers = 2:5, activation = ACTIVATIONS,
            b_size = 1:20, learning_rate = (1.0e-4 .. 1.0e-1), gamma_val = (0.1 .. 0.9), step_rate = 10:40,
            verbose = true, max_trials = 50, pruner = MedianPruner(; start_after = 5, prune_after = 10)
        ),
    ),
    "KAN_RNO" => (
        kan_rno_objective, Scenario(
            n_hidden = 2:90, n_layers = 2:6, activation = ACTIVATIONS,
            wav_one = WAVELET_LIST, wav_two = WAVELET_LIST, wav_three = WAVELET_LIST,
            wav_four = WAVELET_LIST, wav_five = WAVELET_LIST, wav_six = WAVELET_LIST,
            b_size = 1:20, learning_rate = (1.0e-5 .. 1.0e-1), gamma_val = (0.5 .. 0.9), step_rate = 10:40,
            layer_norm = [false, false],
            verbose = true, max_trials = 100, pruner = MedianPruner()
        ),
    ),
    "Transformer" => (
        transformer_objective, Scenario(
            d_model = range(64, 192; step = 2), nhead = 1:20, dim_feedforward = 500:1200,
            dropout = (0.1 .. 0.9), num_encoder_layers = 2:8, num_decoder_layers = 1:3,
            max_len = 1000:5000, activation = ACTIVATIONS,
            b_size = 1:20, learning_rate = (1.0e-4 .. 1.0e-1), gamma_val = (0.1 .. 0.9), step_rate = 10:40,
            verbose = true, max_trials = 50, pruner = MedianPruner(; start_after = 5, prune_after = 10)
        ),
    ),
    "KAN_Transformer" => (
        kan_transformer_objective, Scenario(
            d_model = range(10, 60; step = 2), nhead = 1:10, dim_feedforward = 300:500,
            dropout = (0.1 .. 0.9), num_encoder_layers = 2:3, num_decoder_layers = [1, 1],
            encoder_wav_one = WAVELET_LIST, encoder_wav_two = WAVELET_LIST,
            encoder_wav_three = WAVELET_LIST, encoder_wav_four = WAVELET_LIST,
            encoder_wav_five = WAVELET_LIST, encoder_wav_six = WAVELET_LIST,
            encoder_wav_seven = WAVELET_LIST, encoder_wav_eight = WAVELET_LIST,
            decoder_wav_one = WAVELET_LIST, decoder_wav_two = WAVELET_LIST,
            decoder_wav_three = WAVELET_LIST, output_wavelet = WAVELET_LIST,
            max_len = 251:450, activation = ACTIVATIONS,
            b_size = 1:10, learning_rate = (1.0e-6 .. 1.0e-1), gamma_val = (0.5 .. 0.9), step_rate = 10:40,
            norm = [false, false],
            verbose = true, max_trials = 100, pruner = MedianPruner()
        ),
    ),
)

objective_fn, space = spaces[model_name]
HyperTuning.optimize(objective_fn, space)
display(top_parameters(space))
