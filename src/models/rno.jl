struct RNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_chain, :hidden_chain)}
    output_chain::O
    hidden_chain::H
    dt::Float32
    T::Int
    n_hidden::Int
end

function RNO(cfg::RNOConfig, input_dim::Int, output_dim::Int, input_size::Int)
    phi = get_activation(cfg.activation)
    hidden_units = fill(cfg.n_hidden, cfg.num_layers)
    layer_output = [input_dim + output_dim + cfg.n_hidden; hidden_units; output_dim]
    layer_hidden = [cfg.n_hidden + output_dim; hidden_units; cfg.n_hidden]

    out_layers = Any[Lux.Dense(layer_output[i] => layer_output[i + 1], phi) for i in 1:(length(layer_output) - 2)]
    push!(out_layers, Lux.Dense(layer_output[end - 1] => layer_output[end]))
    hid_layers = Any[Lux.Dense(layer_hidden[i] => layer_hidden[i + 1], phi) for i in 1:(length(layer_hidden) - 2)]
    push!(hid_layers, Lux.Dense(layer_hidden[end - 1] => layer_hidden[end]))

    dt = Float32(1 / (input_size - 1))
    return RNO(Lux.Chain(out_layers...), Lux.Chain(hid_layers...), dt, input_size, cfg.n_hidden)
end

function (m::RNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]

    y = reshape(y_true[1, :], 1, bs)
    hidden = similar(x, m.n_hidden, bs) .* 0.0f0

    st_out = st.output_chain
    st_hid = st.hidden_chain

    for t in 2:(m.T)
        xt = reshape(x[t, :], 1, :)
        xprev = reshape(x[t - 1, :], 1, :)

        h0 = similar(x, m.n_hidden, bs) .* 0.0f0
        h, st_hid = m.hidden_chain(vcat(xprev, hidden), ps.hidden_chain, st_hid)
        hidden = (h .* m.dt) .+ h0

        out, st_out = m.output_chain(vcat(xprev, (xprev .- xt) ./ m.dt, hidden), ps.output_chain, st_out)
        y = vcat(y, out)
    end

    return y, (output_chain = st_out, hidden_chain = st_hid)
end
