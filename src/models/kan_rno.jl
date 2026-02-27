struct KANRNO{O, H} <: Lux.AbstractLuxContainerLayer{(:output_layers, :hidden_layers)}
    output_layers::O
    hidden_layers::H
    dt::Float32
    T::Int
    n_hidden::Int
end

function _make_named_layers(layers::Vector)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return NamedTuple{names}(Tuple(layers))
end

function KANRNO(cfg::KANRNOConfig, input_dim::Int, output_dim::Int, input_size::Int)
    hidden_units = fill(cfg.n_hidden, cfg.num_layers)
    layer_output = [input_dim + output_dim + cfg.n_hidden; hidden_units; output_dim]
    layer_hidden = [cfg.n_hidden + output_dim; hidden_units; cfg.n_hidden]

    out_layers = [
        KANdense(layer_output[i], layer_output[i + 1], cfg.wavelet_names[i], cfg.activation; norm = cfg.norm)
            for i in 1:(length(layer_output) - 1)
    ]
    hid_layers = [
        KANdense(layer_hidden[i], layer_hidden[i + 1], cfg.wavelet_names[i], cfg.activation; norm = cfg.norm)
            for i in 1:(length(layer_hidden) - 1)
    ]

    dt = Float32(1 / (input_size - 1))
    return KANRNO(_make_named_layers(out_layers), _make_named_layers(hid_layers), dt, input_size, cfg.n_hidden)
end

function (m::KANRNO)(input, ps, st)
    x, y_true = input
    bs = size(x)[end]

    y = reshape(y_true[1, :], 1, bs)
    hidden = similar(x, m.n_hidden, bs) .* 0.0f0

    st_out = st.output_layers
    st_hid = st.hidden_layers

    for t in 2:(m.T)
        xt = reshape(x[t, :], 1, :)
        xprev = reshape(x[t - 1, :], 1, :)

        h0 = similar(x, m.n_hidden, bs) .* 0.0f0
        h = vcat(xprev, hidden)
        for (k, layer) in pairs(m.hidden_layers)
            h, st_hid_k = layer(h, ps.hidden_layers[k], st_hid[k])
            st_hid = merge(st_hid, NamedTuple{(k,)}((st_hid_k,)))
        end
        hidden = (h .* m.dt) .+ h0

        output = vcat(xprev, (xprev .- xt) ./ m.dt, hidden)
        for (k, layer) in pairs(m.output_layers)
            output, st_out_k = layer(output, ps.output_layers[k], st_out[k])
            st_out = merge(st_out, NamedTuple{(k,)}((st_out_k,)))
        end
        y = vcat(y, output)
    end

    return y, (output_layers = st_out, hidden_layers = st_hid)
end
