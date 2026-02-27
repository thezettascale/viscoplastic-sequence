include("src/WavKANSequence.jl")

using .WavKANSequence
using Lux
using Reactant
using JLD2
using Plots; pythonplot()
using MLDataDevices: reactant_device, cpu_device

model_name = length(ARGS) >= 1 ? ARGS[1] : "KAN_RNO"
cfg = load_config(model_name)
dev = reactant_device()

train_loader, test_loader = get_visco_loader(1; dev = dev)
input_size = size(first(train_loader)[2], 1)

model = create_model(cfg, input_size)
rng = Lux.default_rng()
_, st = Lux.setup(rng, model)

model_file = joinpath("logs", model_name, "trained_models", "model_1.jld2")
ps = JLD2.load(model_file, "ps") |> dev
st = st |> dev

epsi_first, sigma_first = first(test_loader)
num_samples = size(epsi_first, 1)

model_compiled = @compile model((epsi_first, sigma_first), ps, Lux.testmode(st))
predicted_stress, _ = model_compiled((epsi_first, sigma_first), ps, Lux.testmode(st))
predicted_stress = copy(predicted_stress) |> cpu_device()
epsi_first = epsi_first |> cpu_device()
sigma_first = sigma_first |> cpu_device()

delay = 30
anim = @animate for i in 1:(num_samples + delay)
    if i <= num_samples && i <= delay
        epsi = epsi_first[1:i, 1]
        sigma = sigma_first[1:i, 1]
        pred_epsi = [NaN]
        pred_sigma = [NaN]
    elseif i <= num_samples && i > delay
        epsi = epsi_first[1:i, 1]
        sigma = sigma_first[1:i, 1]
        pred_epsi = epsi_first[1:(i - delay), 1]
        pred_sigma = predicted_stress[1:(i - delay), 1]
    else
        epsi = epsi_first[1:num_samples, 1]
        sigma = sigma_first[1:num_samples, 1]
        pred_epsi = epsi_first[1:(i - delay), 1]
        pred_sigma = predicted_stress[1:(i - delay), 1]
    end
    plot(
        [epsi, pred_epsi], [sigma, pred_sigma],
        title = "$model_name Prediction", xlabel = "Strain", ylabel = "Stress",
        color = [:blue :red], label = ["True" "$model_name Predicted"]
    )
    xlims!(0, 1)
    ylims!(0, 1)
end

mkpath("figures")
gif(anim, "figures/$(model_name)_visco_prediction.gif"; fps = 15)
