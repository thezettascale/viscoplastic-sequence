struct MeyerWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::MeyerWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims),)
end

function (l::MeyerWavelet)(x, ps, st)
    omega = abs.(x)
    y = batch_mul(sin.(omega .* Float32(pi)), meyer_aux(omega))
    return node_mul(y, ps.weights), st
end
