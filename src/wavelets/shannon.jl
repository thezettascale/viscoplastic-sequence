struct ShannonWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::ShannonWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims),)
end

function (l::ShannonWavelet)(x, ps, st)
    y = batch_mul(sinc.(x .* Float32(2pi)), cos.(x .* Float32(pi / 3))) .* 2.0f0
    return node_mul(y, ps.weights), st
end
