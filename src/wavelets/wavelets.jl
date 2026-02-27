include("meyer_utils.jl")
include("morlet.jl")
include("mexican_hat.jl")
include("derivative_of_gaussian.jl")
include("shannon.jl")
include("meyer.jl")

const WAVELET_MAP = Dict{String, Type}(
    "MexicanHat" => MexicanHatWavelet,
    "Morlet" => MorletWavelet,
    "DerivativeOfGaussian" => DoGWavelet,
    "Shannon" => ShannonWavelet,
    "Meyer" => MeyerWavelet,
)

create_wavelet(name::String, in_dims::Int, out_dims::Int) = WAVELET_MAP[name](in_dims, out_dims)
