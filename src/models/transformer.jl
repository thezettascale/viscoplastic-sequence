struct Transformer{PE, E, D, O} <: Lux.AbstractLuxContainerLayer{(:pe, :encoder, :decoder, :output_layer)}
    pe::PE
    encoder::E
    decoder::D
    output_layer::O
end

function Transformer(cfg::TransformerConfig)
    pe = PositionalEncoding(cfg.d_model, cfg.max_len)
    enc = _make_named_layers(
        [
            EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation)
                for _ in 1:(cfg.num_encoder_layers)
        ]
    )
    dec = _make_named_layers(
        [
            DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation)
                for _ in 1:(cfg.num_decoder_layers)
        ]
    )
    return Transformer(pe, enc, dec, Lux.Dense(cfg.d_model => 1))
end

function (m::Transformer)(input, ps, st)
    pred, new_st = _transformer_forward(m, input, ps, st)
    pred = (NNlib.hardtanh(pred) .* 0.5f0) .+ 0.5f0
    return pred, new_st
end
