const DOG_NORM = Float32(1 / sqrt(2 * pi))

struct DoGWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::DoGWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims),)
end

function (l::DoGWavelet)(x, ps, st)
    y = batch_mul(x, exp.(x .* -0.5f0)) .* DOG_NORM
    return node_mul(y, ps.weights), st
end
