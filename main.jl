include("src/WavKANSequence.jl")

using .WavKANSequence
using Random
using Lux
using Lux: Training
using Reactant
using Optimisers
using ProgressBars
using JLD2
using MLDataDevices: reactant_device, cpu_device

const NUM_REPETITIONS = 5

model_name = length(ARGS) >= 1 ? ARGS[1] : "RNO"
cfg = load_config(model_name)
dev = reactant_device()

train_loader, test_loader = get_visco_loader(cfg.batch_size; dev = dev)
input_size = size(first(train_loader)[2], 1)
rng = Lux.default_rng()

log_dir = joinpath("logs", model_name)
model_dir = joinpath(log_dir, "trained_models")
mkpath(model_dir)

for rep in 1:NUM_REPETITIONS
    Random.seed!(rep)
    model = create_model(cfg, input_size)
    ps, st = Lux.setup(rng, model)
    ps = ps |> dev
    st = st |> dev

    train_state = Training.TrainState(model, ps, st, Optimisers.Adam(cfg.learning_rate))

    log_file = joinpath(log_dir, "repetition_$rep.csv")
    open(log_file, "w") do f
        write(f, "Epoch,Time (s),Train Loss,Test Loss,BIC\n")
    end

    loss_fn(y_pred, y) = loss_fcn(y_pred, y; p = cfg.p)
    start_time = time()

    for epoch in ProgressBar(1:(cfg.num_epochs))
        train_state, tl, vl = train_epoch(train_state, train_loader, test_loader, loss_fn, model, epoch, cfg)
        bic = BIC(model, size(first(train_loader)[2], 1), vl)
        log_csv(epoch, tl, vl, bic, time() - start_time, log_file)
    end

    ps_cpu = train_state.parameters |> cpu_device()
    jldsave(joinpath(model_dir, "model_$rep.jld2"); ps = ps_cpu, model_name = model_name)
end
