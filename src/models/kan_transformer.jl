struct KANTransformer{PE, E, D, O} <: Lux.AbstractLuxContainerLayer{(:pe, :encoder, :decoder, :output_layer)}
    pe::PE
    encoder::E
    decoder::D
    output_layer::O
end

function KANTransformer(cfg::KANTransformerConfig)
    pe = PositionalEncoding(cfg.d_model, cfg.max_len)

    enc = _make_named_layers(
        [
            begin
                    wn = cfg.encoder_wavelet_names[i]
                    factory(in_d, out_d) = KANdense(in_d, out_d, wn, cfg.activation; norm = cfg.norm, is_2d = true)
                    EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation; kan_factory = factory)
                end for i in 1:(cfg.num_encoder_layers)
        ]
    )

    dec = _make_named_layers(
        [
            begin
                    wn = cfg.decoder_wavelet_names[i]
                    factory(in_d, out_d) = KANdense(in_d, out_d, wn, cfg.activation; norm = cfg.norm, is_2d = true)
                    DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation; kan_factory = factory)
                end for i in 1:(cfg.num_decoder_layers)
        ]
    )

    out = KANdense(cfg.d_model, 1, cfg.output_wavelet, cfg.activation; norm = cfg.norm, is_2d = true)
    return KANTransformer(pe, enc, dec, out)
end

function (m::KANTransformer)(input, ps, st)
    return _transformer_forward(m, input, ps, st)
end
