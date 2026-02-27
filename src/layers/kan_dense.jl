struct KANdense{W, N} <: Lux.AbstractLuxContainerLayer{(:transform, :norm_layer)}
    transform::W
    norm_layer::N
    in_dims::Int
    out_dims::Int
end

function KANdense(in_dims::Int, out_dims::Int, wavelet_name::String, base_activation::String = "relu"; norm::Bool = false, is_2d::Bool = false)
    wavelet = create_wavelet(wavelet_name, in_dims, out_dims)
    norm_layer = if norm
        is_2d ? Lux.BatchNorm(out_dims) : Lux.LayerNorm(out_dims)
    else
        Lux.WrappedFunction(identity)
    end
    return KANdense(wavelet, norm_layer, in_dims, out_dims)
end

function Lux.initialparameters(rng::AbstractRNG, l::KANdense)
    return (
        transform = Lux.initialparameters(rng, l.transform),
        norm_layer = Lux.initialparameters(rng, l.norm_layer),
        scale = ones(Float32, l.in_dims, l.out_dims),
        translation = zeros(Float32, l.in_dims, l.out_dims),
    )
end

_expand(x::AbstractArray{T, 2}, out_dims) where {T} =
    repeat(reshape(x, size(x, 1), 1, size(x, 2)), 1, out_dims, 1)
_expand(x::AbstractArray{T, 3}, out_dims) where {T} =
    repeat(reshape(x, size(x, 1), 1, size(x, 2), size(x, 3)), 1, out_dims, 1, 1)

function (l::KANdense)(x, ps, st)
    x_exp = _expand(x, l.out_dims)
    trans = repeat(ps.translation, 1, 1, size(x_exp)[3:end]...)
    sc = repeat(ps.scale, 1, 1, size(x_exp)[3:end]...)
    x_exp = (x_exp .- trans) ./ sc

    y, st_t = l.transform(x_exp, ps.transform, st.transform)

    if ndims(y) == 4
        y_perm = reshape(y, size(y, 2), size(y, 1), size(y, 3))
        y_normed, st_n = l.norm_layer(y_perm, ps.norm_layer, st.norm_layer)
        out = reshape(y_normed, size(y, 1), size(y, 2), size(y, 3))
    else
        out, st_n = l.norm_layer(y, ps.norm_layer, st.norm_layer)
    end

    return out, (transform = st_t, norm_layer = st_n)
end
