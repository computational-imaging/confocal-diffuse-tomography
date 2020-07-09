import torch


def interpolate(grid, lin_ind_frustrum, voxel_coords, device_id):
    """ linear interpolation for frequency-wavenumber migration
        adapted from https://github.com/vsitzmann/deepvoxels/blob/49369e243001658ccc8ba3be97d87c85273c9f15/projection.py
    """

    depth, width, height = grid.shape

    lin_ind_frustrum = lin_ind_frustrum.long()

    x_indices = voxel_coords[1, :]
    y_indices = voxel_coords[2, :]
    z_indices = voxel_coords[0, :]

    mask = ((x_indices < 0) | (y_indices < 0) | (z_indices < 0) |
            (x_indices > width-1) | (y_indices > height-1) | (z_indices > depth-1)).to(device_id)

    x0 = x_indices.floor().long()
    y0 = y_indices.floor().long()
    z0 = z_indices.floor().long()

    x0 = torch.clamp(x0, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    z0 = torch.clamp(z0, 0, depth - 1)
    z1 = (z0 + 1).long()
    z1 = torch.clamp(z1, 0, depth - 1)

    x_indices = torch.clamp(x_indices, 0, width - 1)
    y_indices = torch.clamp(y_indices, 0, height - 1)
    z_indices = torch.clamp(z_indices, 0, depth - 1)

    x = x_indices - x0.float()
    y = y_indices - y0.float()
    z = z_indices - z0.float()

    output = torch.zeros(height * width * depth).to(device_id)
    tmp1 = grid[z0, x0, y0] * (1 - z) * (1 - x) * (1 - y)
    tmp2 = grid[z1, x0, y0] * z * (1 - x) * (1 - y)
    output[lin_ind_frustrum] = tmp1 + tmp2

    output = output * (1 - mask.float())
    output = output.contiguous().view(depth, width, height)

    return output


def roll_n(X, axis, n):
    """ circular shift function """

    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(x):
    real, imag = torch.unbind(x, -1)

    if real.ndim > 3:
        dim_start = 2
    else:
        dim_start = 0

    for dim in range(dim_start, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def ifftshift(x):
    real, imag = torch.unbind(x, -1)

    if real.ndim > 3:
        dim_stop = 1
    else:
        dim_stop = -1

    for dim in range(len(real.size()) - 1, dim_stop, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def compl_mul(X, Y):
    """ complex multiplication for pytorch; real and imaginary parts are
        stored in the last channel of the arrays
        see https://discuss.pytorch.org/t/aten-cuda-implementation-of-complex-multiply/17215/2
    """

    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def conj(x):
    # complex conjugation for pytorch

    tmp = x.clone()
    tmp[:, :, :, :, :, 1] = tmp[:, :, :, :, :, 1] * -1
    return tmp


def fk(meas, width, mrange):
    """ perform f--k migration """

    device = meas.device
    meas = meas.squeeze()
    width = torch.FloatTensor([width]).to(device)
    mrange = torch.FloatTensor([mrange]).to(device)

    N = meas.size()[1]//2  # spatial resolution
    M = meas.size()[0]//2  # temporal resolution
    data = torch.sqrt(torch.clamp(meas, 0))

    M_grid = torch.arange(-M, M).to(device)
    N_grid = torch.arange(-N, N).to(device)
    [z, x, y] = torch.meshgrid(M_grid, N_grid, N_grid)
    z = (z.type(torch.FloatTensor) / M).to(device)
    x = (x.type(torch.FloatTensor) / N).to(device)
    y = (y.type(torch.FloatTensor) / N).to(device)

    # pad data
    tdata = data

    # fourier transform
    if tdata.ndim > 3:
        tdata = fftshift(tdata.fft(3))
    else:
        tdata = fftshift(tdata.rfft(3, onesided=False))

    tdata_real, tdata_imag = torch.unbind(tdata, -1)

    # interpolation coordinates
    z_interp = torch.sqrt(abs((((N * mrange) / (M * width * 4))**2) *
                              (x**2 + y**2) + z**2))
    coords = torch.stack((z_interp.flatten(), x.flatten(), y.flatten()), 0)
    lin_ind = torch.arange(z.numel()).to(device)
    coords[0, :] = (coords[0, :] + 1) * M
    coords[1, :] = (coords[1, :] + 1) * N
    coords[2, :] = (coords[2, :] + 1) * N

    # run interpolation
    tvol_real = interpolate(tdata_real, lin_ind, coords, device)
    tvol_imag = interpolate(tdata_imag, lin_ind, coords, device)
    tvol = torch.stack((tvol_real, tvol_imag), -1)

    # zero out redundant spectrum
    x = x[:, :, :, None]
    y = y[:, :, :, None]
    z = z[:, :, :, None]
    tvol = tvol * abs(z) / torch.clamp(torch.sqrt(abs((((N * mrange) / (M * width * 4))**2) *
                                       (x**2 + y**2)+z**2)), 1e-8)
    tvol = tvol * (z > 0).type(torch.FloatTensor).to(device)

    # inverse fourier transform and crop
    tvol = ifftshift(tvol).ifft(3).squeeze()
    geom = tvol[:, :, :, 0]**2 + tvol[:, :, :, 1]**2
    geom = geom[None, None, :, :, :]

    return geom
