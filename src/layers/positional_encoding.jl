struct PositionalEncoding <: Lux.AbstractLuxLayer
    d_model::Int
    max_len::Int
end

function Lux.initialstates(::AbstractRNG, l::PositionalEncoding)
    pe = zeros(Float32, l.d_model, l.max_len)
    position = collect(Float32, 1:(l.max_len))
    div_term = exp.(-log(10000.0f0) .* collect(Float32, 1:2:(l.d_model)) ./ l.d_model)
    div_term = reshape(div_term, 1, length(div_term))
    pe[1:2:end, :] = transpose(sin.(position .* div_term))
    pe[2:2:end, :] = transpose(cos.(position .* div_term))
    return (pe_vector = pe,)
end

function (l::PositionalEncoding)(x, ps, st)
    x3 = reshape(x, 1, size(x, 1), size(x, 2))
    encoding = repeat(st.pe_vector[:, 1:size(x3, 2)], 1, 1, size(x3, 3))
    return x3 .+ encoding, st
end
