include("rno.jl")
include("kan_rno.jl")
include("transformer.jl")
include("kan_transformer.jl")

function _transformer_forward(m, input, ps, st)
    src_raw, tgt_raw = input
    src, st_pe = m.pe(src_raw, ps.pe, st.pe)
    tgt, st_pe = m.pe(tgt_raw, ps.pe, st_pe)

    st_enc = st.encoder
    memory = src
    for k in keys(m.encoder)
        memory, st_enc_k = m.encoder[k](memory, ps.encoder[k], st_enc[k])
        st_enc = merge(st_enc, NamedTuple{(k,)}((st_enc_k,)))
    end

    st_dec = st.decoder
    for k in keys(m.decoder)
        tgt, st_dec_k = m.decoder[k]((tgt, memory), ps.decoder[k], st_dec[k])
        st_dec = merge(st_dec, NamedTuple{(k,)}((st_dec_k,)))
    end

    pred, st_o = m.output_layer(tgt, ps.output_layer, st.output_layer)
    pred = reshape(pred, size(pred, 2), size(pred, 3))
    return pred, (pe = st_pe, encoder = st_enc, decoder = st_dec, output_layer = st_o)
end

function create_model(cfg::RNOConfig, input_size::Int)
    return RNO(cfg, 1, 1, input_size)
end

function create_model(cfg::KANRNOConfig, input_size::Int)
    return KANRNO(cfg, 1, 1, input_size)
end

function create_model(cfg::TransformerConfig, ::Int)
    return Transformer(cfg)
end

function create_model(cfg::KANTransformerConfig, ::Int)
    return KANTransformer(cfg)
end
