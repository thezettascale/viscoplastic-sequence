struct MorletWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::MorletWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims), gamma = Float32[5.0f0])
end

function (l::MorletWavelet)(x, ps, st)
    y = batch_mul(cos.(ps.gamma .* x), exp.(-x .^ 2 .* 0.5f0))
    return node_mul(y, ps.weights), st
end
