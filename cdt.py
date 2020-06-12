#!/usr/bin/python
'''
Code for "Three-dimensional imaging through scattering media based on confocal diffuse tomography"
David B. Lindell and Gordon Wetzstein

See README file in this directory for instructions on how to setup and run the code

'''

import argparse
import h5py
import os
import time
import numpy as np
from numpy.fft import ifftn, fftn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as TorchF
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='letter_s',
                    choices=['cones', 'letter_s', 'letters_ut', 'mannequin', 'letter_t'],
                    help='name of scene to reconstruct, should be one of {cones, letter_s, letters_ut, mannequin, letter_t}')
parser.add_argument('--gpu_id', type=int, default=0, help='index of which GPU to run on, default=0')
parser.add_argument('--pause', type=int, default=5, help='how long to display figure, default=5 (seconds)')
opt = parser.parse_args()
print('Confocal diffuse tomography reconstruction')
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

# check to see if GPU is available
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# some helper functions
def interpolate(grid, lin_ind_frustrum, voxel_coords, device_id=device):
    # linear interpolation for frequency-wavenumber migration
    # adapted from https://github.com/vsitzmann/deepvoxels/blob/49369e243001658ccc8ba3be97d87c85273c9f15/projection.py

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
    # circular shift function

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
    # complex multiplication for pytorch; real and imaginary parts are
    # stored in the last channel of the arrays
    # see https://discuss.pytorch.org/t/aten-cuda-implementation-of-complex-multiply/17215/2

    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


class CDTReconstruction():
    # class to define scattering parameters and perform
    # reconstruction using confocal diffuse tomography

    def __init__(self):

        # set hyper parameters
        if opt.scene == 'letter_s':
            self.snr = 2e5  # SNR parameter for Wiener deconvolution
            self.scan_size = 0.6  # size of scanned area
            self.size_calibration = 1.06  # calibrated scanned area scaling for reconstruction
        elif opt.scene == 'mannequin':
            self.snr = 8e5
            self.scan_size = 0.7
            self.size_calibration = 0.88
        elif opt.scene == 'letters_ut':
            self.snr = 2e5
            self.scan_size = 0.7
            self.size_calibration = 1.0
        elif opt.scene == 'letter_t':
            self.snr = 0.6e5
            self.scan_size = 0.7
            self.size_calibration = 0.91
        elif opt.scene == 'cones':
            self.snr = 2e6
            self.scan_size = 0.7
            self.size_calibration = 1.0
        else:
            raise ValueError('Unexpected input to scene parameter. Allowed scenes are {letter_s, mannequin, letters_ut, letter_t, cones}')

        # physical parameters
        self.c = 3e8
        self.mu_a = 7
        self.mu_s = 242
        self.D = 1 / (3 * (self.mu_a + self.mu_s))

        # volume dimensions
        self.Nx = 32
        self.Ny = 32
        self.Nz = 128
        self.xmin = -self.size_calibration * self.scan_size / 2
        self.xmax = self.size_calibration * self.scan_size / 2
        self.ymin = -self.size_calibration * self.scan_size / 2
        self.ymax = self.size_calibration * self.scan_size / 2
        self.zmin = 0
        self.zmax = 2  # maximum path length in hidden volume (meters)

        self.x = np.linspace(self.xmin, self.xmax, self.Nx)
        self.y = np.linspace(self.ymin, self.ymax, self.Ny)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz)
        self.X, self.Z, self.Y = np.meshgrid(self.x, self.z, self.y)

        # laser position
        self.xl = 0
        self.yl = 0
        self.zl = 0

        # diffuser positioning
        self.xd = np.linspace(2*self.xmin, 2*self.xmax, 2*self.Nx)[None, :, None]
        self.yd = np.linspace(2*self.ymin, 2*self.ymax, 2*self.Ny)[None, None, :]
        self.t = np.linspace(0, 2*self.zmax, 2*self.Nz) / self.c
        self.t = self.t[:, None, None]
        self.zd = 0.0254  # thickness of diffuser

        # set diffusion kernel
        self.diffusion_fpsf = []
        self.setDiffusionKernel(self.c, self.t, self.xl, self.yl, self.zl,
                                self.xd, self.yd, self.zd, self.D, self.mu_a)

    def setDiffusionKernel(self, v, t, xl, yl, zl, xd, yd, zd, D, mu_a):
        # set the diffusion kernel using the diffusion equation

        t[0, :] = 1
        diff_kernel = v / (4*np.pi*D*v*t)**(3/2) * \
            np.exp(-np.abs((xd-xl)**2 + (yd - yl)**2 + (zd-zl)**2) /
                   (4*D*v*t)) * np.exp(-mu_a*v*t)
        diff_kernel[0, :, :] = 0

        psf = diff_kernel
        diffusion_psf = psf / np.sum(psf)
        diffusion_psf = np.roll(diffusion_psf, -xd.shape[1]//2, axis=1)
        diffusion_psf = np.roll(diffusion_psf, -yd.shape[2]//2, axis=2)
        diffusion_psf = fftn(diffusion_psf) * fftn(diffusion_psf)
        diffusion_psf = abs(ifftn(diffusion_psf))

        # convert to pytorch and take fft
        self.diffusion_fpsf = torch.from_numpy(diffusion_psf.astype(np.float32)).to(device)[None, None, :, :, :]
        self.diffusion_fpsf = self.diffusion_fpsf.rfft(3, onesided=False)
        return

    def fk(self, meas, width, mrange):
        # perform f--k migration

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
        tvol_real = interpolate(tdata_real, lin_ind, coords)
        tvol_imag = interpolate(tdata_imag, lin_ind, coords)
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

    def conj(self, x):
        # complex conjugation for pytorch

        tmp = x.clone()
        tmp[:, :, :, :, :, 1] = tmp[:, :, :, :, :, 1] * -1
        return tmp

    def AT(self, x):
        # wrapper function for f--k migration

        return self.fk(x, 2*self.xmax, 2*self.zmax)

    def M(self, x):
        # trimming function

        return x[:, :, :self.Nz, :self.Nx, :self.Ny]

    def MT(self, x):
        # padding function

        return TorchF.pad(x, (0, self.Ny, 0, self.Nx, 0, self.Nz))

    def run(self):
        # run confocal diffuse tomography reconstruction

        with h5py.File('./data/' + opt.scene + '.mat', 'r') as f:
            meas = np.array(f['meas']).transpose(2, 1, 0)
        f.close()

        # trim scene to 1 meter along the z-dimension
        # and downsample to ~50 ps time binning
        b = meas[:417, :, :]
        downsampled = np.zeros((self.Nz, 32, 32))
        for i in range(meas.shape[1]):
            for j in range(meas.shape[2]):
                x = np.linspace(0, 1, self.Nz)
                xp = np.linspace(0, 1, 417)
                yp = b[:, i, j].squeeze()
                downsampled[:, i, j] = np.interp(x, xp, yp)
        b = downsampled
        b /= np.max(b)  # normalize to 0 to 1

        # initialize pytorch arrays
        b = torch.from_numpy(b).to(device)[None, None, :, :, :].float()
        x = torch.zeros(b.size()[0], 1, 2*self.Nz, 2*self.Nx, 2*self.Ny).to(device)

        # construct inverse psf for Wiener filtering
        tmp = compl_mul(self.diffusion_fpsf, self.conj(self.diffusion_fpsf))
        tmp = tmp + 1/self.snr
        invpsf = compl_mul(self.conj(self.diffusion_fpsf), 1/tmp)

        # measure inversion runtime
        if device.type == 'cpu':
            start = time.time()
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # pad measurements
        x = self.MT(b)

        # perform deconvolution
        x_deconv = compl_mul(x.rfft(3, onesided=False), invpsf).ifft(3)[:, :, :, :, :, 0]

        # confocal inverse filter
        x = self.AT(x_deconv)

        # measure elapsed time
        if device.type == 'cpu':
            stop = time.time()
            print('Elapsed time: %.02f ms' % (1000 * (stop - start)))
        else:
            end.record()
            torch.cuda.synchronize()
            print('Elapsed time: %.02f ms' % (start.elapsed_time(end)))

        # plot results
        x_npy = x.cpu().data.numpy().squeeze()[:self.Nz, :self.Nx, :self.Ny]
        b_npy = b.cpu().data.numpy().squeeze()

        # trim any amplified noise at the very end of the volume
        x_npy[-15:, :, :] = 0

        plt.suptitle('Measurements and reconstruction')
        plt.subplot(231)
        plt.imshow(np.max(b_npy, axis=0), cmap='gray', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.subplot(232)
        plt.imshow(np.max(b_npy, axis=1), aspect=(self.xmax-self.xmin)/(self.zmax/3e8*1e9), cmap='gray',
                   extent=[self.xmin, self.xmax, self.zmax/3e8*1e9, self.zmin])
        plt.xlabel('x (m)')
        plt.ylabel('t (ns)')
        plt.subplot(233)
        plt.imshow(np.max(b_npy, axis=2), aspect=(self.ymax-self.ymin)/(self.zmax/3e8*1e9), cmap='gray',
                   extent=[self.ymin, self.ymax, self.zmax/3e8*1e9, self.zmin])
        plt.xlabel('y (m)')
        plt.ylabel('t (ns)')

        plt.subplot(234)
        plt.imshow(np.max(x_npy, axis=0), cmap='gray', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.subplot(235)
        plt.imshow(np.max(x_npy, axis=1), aspect=(self.xmax-self.xmin)/(self.zmax/2), cmap='gray',
                   extent=[self.xmin, self.xmax, self.zmax/2, self.zmin])
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.subplot(236)
        plt.imshow(np.max(x_npy, axis=2), aspect=(self.ymax-self.ymin)/(self.zmax/2), cmap='gray',
                   extent=[self.ymin, self.ymax, self.zmax/2, self.zmin])
        plt.xlabel('y (m)')
        plt.ylabel('z (m)')
        plt.tight_layout()

        plt.pause(opt.pause)


def main():
    cdt = CDTReconstruction()
    cdt.run()
    return


if __name__ == '__main__':
    main()
