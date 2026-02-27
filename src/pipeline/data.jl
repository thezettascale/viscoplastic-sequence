using MAT
using MLUtils: DataLoader
using MLDataDevices: MLDataDevices

function load_visco_data(; n_total::Int = 400, n_train::Int = 300)
    matfile = matread("data/1D_Viscoplastic_Data/viscodata_3mat.mat")
    epsi = Float32.(matfile["epsi_tol"])[1:n_total, :]
    sigma = Float32.(matfile["sigma_tol"])[1:n_total, :]
    return epsi[1:n_train, :], sigma[1:n_train, :], epsi[(n_train + 1):end, :], sigma[(n_train + 1):end, :]
end

function get_visco_loader(batch_size::Int; dev = MLDataDevices.cpu_device(), subsample::Int = 4)
    epsi_train, sigma_train, epsi_test, sigma_test = load_visco_data()
    normaliser = MinMaxNormaliser(epsi_train)
    sigma_norm = MinMaxNormaliser(sigma_train)

    # Downsample in time
    epsi_train = epsi_train[:, 1:subsample:end]
    sigma_train = sigma_train[:, 1:subsample:end]
    epsi_test = epsi_test[:, 1:subsample:end]
    sigma_test = sigma_test[:, 1:subsample:end]

    # Normalise
    epsi_train = encode(normaliser, epsi_train)
    sigma_train = encode(sigma_norm, sigma_train)
    epsi_test = encode(normaliser, epsi_test)
    sigma_test = encode(sigma_norm, sigma_test)

    # (samples, time) -> (time, batch) convention
    epsi_train = permutedims(epsi_train, [2, 1])
    sigma_train = permutedims(sigma_train, [2, 1])
    epsi_test = permutedims(epsi_test, [2, 1])
    sigma_test = permutedims(sigma_test, [2, 1])

    train_data = dev((epsi_train, sigma_train))
    test_data = dev((epsi_test, sigma_test))

    train_loader = DataLoader(train_data; batchsize = batch_size, shuffle = true)
    test_loader = DataLoader(test_data; batchsize = batch_size, shuffle = false)
    return train_loader, test_loader
end
