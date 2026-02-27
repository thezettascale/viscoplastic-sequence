const MEXICAN_HAT_NORM = Float32(2 / sqrt(3 * sqrt(pi)))

struct MexicanHatWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::MexicanHatWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims),)
end

function (l::MexicanHatWavelet)(x, ps, st)
    y = batch_mul(x .^ 2 .- 1.0f0, exp.(x .^ 2 .* -0.5f0)) .* MEXICAN_HAT_NORM
    return node_mul(y, ps.weights), st
end
