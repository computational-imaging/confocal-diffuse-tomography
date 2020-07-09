'''
Code for "Three-dimensional imaging through scattering media based on confocal diffuse tomography"
David B. Lindell and Gordon Wetzstein

See README file in this directory for instructions on how to setup and run the code
'''

import h5py
import time
import numpy as np
from numpy.fft import ifftn, fftn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as TorchF
from utils import fk, compl_mul, conj


class CDTReconstruction():
    # class to define scattering parameters and perform
    # reconstruction using confocal diffuse tomography

    def __init__(self, scene, mu_s=None, zd=None, pause=5, device=torch.device('cuda:0')):

        self.device = device
        self.scene = scene
        self.pause = pause

        # set hyper parameters
        if scene == 'letter_s':
            self.snr = 1e4  # SNR parameter for Wiener deconvolution
            self.scan_size = 0.6  # size of scanned area
            self.size_calibration = 1.06  # calibrated scanned area scaling for reconstruction
            self.exposure_time = 60 / 32**2  # per pixel exposure time, seconds
        elif scene == 'mannequin':
            self.snr = 2e4
            self.scan_size = 0.7
            self.size_calibration = 0.87
            self.exposure_time = 720 / 32**2
        elif scene == 'letters_ut':
            self.snr = 1e4
            self.scan_size = 0.7
            self.size_calibration = 1.0
            self.exposure_time = 600 / 32**2
        elif scene == 'letter_t':
            self.snr = 2.5e3
            self.scan_size = 0.7
            self.size_calibration = 1.0
            self.exposure_time = 3600 / 32**2
        elif scene == 'cones':
            self.snr = 2e4
            self.scan_size = 0.7
            self.size_calibration = 1.0
            self.exposure_time = 400 / 32**2
        elif scene == 'resolution_50':
            self.snr = 1.5e4
            self.scan_size = 0.7
            self.size_calibration = 1.02
            self.exposure_time = 80 / 32**2
        elif scene == 'resolution_70':
            self.snr = 1.5e4
            self.scan_size = 0.7
            self.size_calibration = 1.04
            self.exposure_time = 80 / 32**2
        elif 'letter_u' in scene:
            self.snr = 5e3
            self.scan_size = 0.7
            self.size_calibration = 1.0
            self.exposure_time = 60 / 32**2
        else:
            raise ValueError('Unexpected input to scene parameter.')

        # physical parameters
        # found by minimizing model fit error to calibration data
        self.c0 = 3e8
        self.n = 1.12
        self.c = self.c0/self.n
        self.mu_a = 0.53
        self.mu_s = 262
        self.ze = 0.0036

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

        # allow optional override of these parameters
        if zd:
            self.zd = zd
        if mu_s:
            self.mu_s = mu_s

        # set diffusion kernel
        self.diffusion_fpsf = []
        self.setDiffusionKernel(self.c, self.t, self.xl, self.yl, self.zl,
                                self.xd, self.yd, self.zd, self.ze,
                                self.mu_s, self.mu_a)

    def setDiffusionKernel(self, v, t, xl, yl, zl, xd, yd, zd, ze, mu_s, mu_a):

        '''
        Returns the diffusion model for a slab with finite thickness given by
        Michael S. Patterson, B. Chance, and B. C. Wilson,
        "Time resolved reflectance and transmittance for the noninvasive
        measurement of tissue optical properties,"
        Appl. Opt. 28, 2331-2336 (1989)
        '''

        t[0, :] = 1
        d = zd - zl
        z0 = 1 / mu_s
        D = 1 / (3 * (mu_a + mu_s))
        rho = np.sqrt((xd-xl)**2 + (yd - yl)**2)

        # Photon migration through a turbid slab described by a model
        # based on diffusion approximation.
        # https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-19-4587
        n_dipoles = 20
        ii = np.arange(-n_dipoles, n_dipoles+1)[None, None, :]
        z1 = d * (1 - 2 * ii) - 4*ii*ze - z0
        z2 = d * (1 - 2 * ii) - (4*ii - 2)*ze + z0

        dipole_term = z1 * np.exp(-(z1**2) / (4*D*v*t)) - \
            z2 * np.exp(-(z2**2) / (4*D*v*t))

        dipole_term = np.sum(dipole_term, axis=-1)[..., None]  # sum over dipoles

        diff_kernel = (4*np.pi*D*v)**(-3/2) * t**(-5/2) \
            * np.exp(-mu_a * v * t - rho**2 / (4*D*v*t)) \
            * dipole_term

        psf = diff_kernel

        diffusion_psf = psf / np.sum(psf)
        diffusion_psf = np.roll(diffusion_psf, -xd.shape[1]//2, axis=1)
        diffusion_psf = np.roll(diffusion_psf, -yd.shape[2]//2, axis=2)
        diffusion_psf = fftn(diffusion_psf) * fftn(diffusion_psf)
        diffusion_psf = abs(ifftn(diffusion_psf))

        # convert to pytorch and take fft
        self.diffusion_fpsf = torch.from_numpy(diffusion_psf.astype(np.float32)).to(self.device)[None, None, :, :, :]
        self.diffusion_fpsf = self.diffusion_fpsf.rfft(3, onesided=False)
        return

    def AT(self, x):
        # wrapper function for f--k migration

        return fk(x, 2*self.xmax, 2*self.zmax)

    def M(self, x):
        # trimming function

        return x[:, :, :self.Nz, :self.Nx, :self.Ny]

    def MT(self, x):
        # padding function

        return TorchF.pad(x, (0, self.Ny, 0, self.Nx, 0, self.Nz))

    def run(self):
        # run confocal diffuse tomography reconstruction

        with h5py.File('./data/' + self.scene + '.mat', 'r') as f:
            meas = np.array(f['meas']).transpose(2, 1, 0)
        f.close()

        # trim scene to 1 meter along the z-dimension
        # and downsample to ~50 ps time binning from 16 ps
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
        b = torch.from_numpy(b).to(self.device)[None, None, :, :, :].float()
        x = torch.zeros(b.size()[0], 1, 2*self.Nz, 2*self.Nx, 2*self.Ny).to(self.device)

        # construct inverse psf for Wiener filtering
        tmp = compl_mul(self.diffusion_fpsf, conj(self.diffusion_fpsf))
        tmp = tmp + 1/self.snr
        invpsf = compl_mul(conj(self.diffusion_fpsf), 1/tmp)

        # measure inversion runtime
        if self.device.type == 'cpu':
            start = time.time()
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # pad measurements
        x = self.MT(b)

        # perform f-k migration on measurements
        x_fk = self.AT(x)

        # perform deconvolution
        x_deconv = compl_mul(x.rfft(3, onesided=False), invpsf).ifft(3)[:, :, :, :, :, 0]

        # confocal inverse filter
        x = self.AT(x_deconv)

        # measure elapsed time
        if self.device.type == 'cpu':
            stop = time.time()
            print('Elapsed time: %.02f ms' % (1000 * (stop - start)))
        else:
            end.record()
            torch.cuda.synchronize()
            print('Elapsed time: %.02f ms' % (start.elapsed_time(end)))

        # plot results
        x_npy = x.cpu().data.numpy().squeeze()[:self.Nz, :self.Nx, :self.Ny]
        b_npy = b.cpu().data.numpy().squeeze()
        x_deconv_npy = x_deconv.cpu().data.numpy().squeeze()[:self.Nz, :self.Nx, :self.Ny]
        x_fk_npy = x_fk.cpu().data.numpy().squeeze()[:self.Nz, :self.Nx, :self.Ny]

        # trim any amplified noise at the very end of the volume
        x_npy[-15:, :, :] = 0

        if self.pause > 0:
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

            plt.pause(self.pause)

        # return measurements, deconvolved meas, reconstruction
        return b_npy, x_fk_npy, x_deconv_npy, x_npy
