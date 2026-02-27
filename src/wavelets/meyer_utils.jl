function nu(x)
    term1 = 35.0f0 .+ (-84.0f0 .* x) .+ (70.0f0 .* x .^ 2) .+ (-20.0f0 .* x .^ 3)
    return batch_mul(term1, x .^ 4)
end

function smooth_step(z, a, b)
    return (1.0f0 .+ tanh.((z .- a) ./ (b .- a))) .* 0.5f0
end

function meyer_aux(x)
    eps = 1.0f-6
    t05 = smooth_step(x, 0.5f0, 0.5f0 + eps)
    t1 = smooth_step(x, 1.0f0, 1.0f0 + eps)
    term1 = 1.0f0 .- t05
    term2 = three_mul(cos.(Float32(pi) .* nu(2.0f0 .* x .- 1.0f0) ./ 2.0f0), 1.0f0 .- t1, t05)
    return term1 + term2
end
