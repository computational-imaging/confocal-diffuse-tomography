import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cpu')
torch.set_default_dtype(torch.float64)


class DiffusionModel(nn.Module):
    def __init__(self, us_init=150., ua_init=0., n_init=1.0, t0_init=0., datadir='./data'):
        super().__init__()
        self.bin_size = 8e-12
        self.t = torch.arange(1, 65536)*self.bin_size
        self.t = self.t[self.t < 35e-9]
        self.c0 = 3e8
        self.c = self.c0 / n_init
        self.thicknesses = torch.arange(1, 8.5, 0.5)
        self.datadir = datadir

        self.t = self.t.to(device)
        self.thicknesses = self.thicknesses.to(device)

        # get diffuse reflectance
        self.R = self.calculate_reflection_coeff(n_init)

        # all values are scaled for numerical accuracy so optimized value should be ~0 to 10
        self.us_scale = 100.
        self.t0_scale = 1e9

        self.us_init = us_init / self.us_scale  # originally m^-1
        self.ua_init = ua_init  # m^-1
        self.n_init = n_init  # mm
        self.t0_init = t0_init  # ns

        self.laser, _ = self.load_impulse_response()
        self.data = self.preprocess_data()

        # optimizable parameters
        self.ua = nn.Parameter(torch.Tensor([self.ua_init])).to(device)
        self.us = nn.Parameter(torch.Tensor([self.us_init])).to(device)
        self.n = torch.Tensor([self.n_init]).to(device)
        self.t0 = nn.Parameter(self.t0_init * torch.ones_like(self.thicknesses)).to(device)

    def fresnel(self, n, theta_i):
        n0 = 1.
        with np.errstate(invalid='ignore'):
            theta_t = np.arcsin(n*np.sin(theta_i) / n0)
            R = 0.5 * (np.sin(theta_i - theta_t)**2 / np.sin(theta_i + theta_t)**2 +
                       np.tan(theta_i - theta_t)**2 / np.tan(theta_i + theta_t)**2)
            R[theta_i == 0] = (n - n0)**2 / (n + n0)**2
            R[np.arcsin(n0/n) < theta_i] = 1.
        return R

    def calculate_reflection_coeff(self, n1):
        '''
        calculate reflection coefficient given the refractive indices of the
        two materials. This is derived in

        Zhu, J. X., D. J. Pine, and D. A. Weitz.
        "Internal reflection of diffusive light in random media."
        Physical Review A 44.6 (1991): 3948.
        '''
        # integrate to calculate c1 and c2
        theta = np.linspace(0., np.pi/2, 501)

        c1 = abs(np.trapz(self.fresnel(n1, theta)*np.sin(theta)*np.cos(theta), theta))
        theta = -np.linspace(-np.pi/2, 0., 501)
        c2 = abs(np.trapz(self.fresnel(n1, theta)*np.sin(theta)*np.cos(theta)**2, theta))

        R = (3*c2 + 2*c1) / (3*c2 - 2*c1 + 2)
        return R

    def load_impulse_response(self):
        N = 129

        with h5py.File(self.datadir + '/capture_direct.mat', 'r') as f:
            data = np.array(f['out']).astype(np.float64)
            data = torch.from_numpy(data).to(device)

            # subtract dc component
            data = data[data > 0]
            data = data - torch.median(data)
            data = torch.clamp(data, 0.)

            shift = torch.where(data > 100)[0][0].int()
            data = data[shift:shift+N].squeeze()
            data = data / torch.sum(data)

            # we need to flip manually since pytorch
            # convolution is implemented as correlation
            data = torch.flip(data, dims=(0,))

            return data[None, None, :], shift

    def preprocess_data(self):
        data = []
        for idx, thickness in enumerate(self.thicknesses):
            switcher = {
                    1.0: self.datadir + '/capture_1_0_inch.mat',
                    1.5: self.datadir + '/capture_1_5_inch.mat',
                    2.0: self.datadir + '/capture_2_0_inch.mat',
                    2.5: self.datadir + '/capture_2_5_inch.mat',
                    3.0: self.datadir + '/capture_3_0_inch.mat',
                    3.5: self.datadir + '/capture_3_5_inch.mat',
                    4.0: self.datadir + '/capture_4_0_inch.mat',
                    4.5: self.datadir + '/capture_4_5_inch.mat',
                    5.0: self.datadir + '/capture_5_0_inch.mat',
                    5.5: self.datadir + '/capture_5_5_inch.mat',
                    6.0: self.datadir + '/capture_6_0_inch.mat',
                    6.5: self.datadir + '/capture_6_5_inch.mat',
                    7.0: self.datadir + '/capture_7_0_inch.mat',
                    7.5: self.datadir + '/capture_7_5_inch.mat',
                    8.0: self.datadir + '/capture_8_0_inch.mat'}

            d = 0.0254 * thickness
            with h5py.File(switcher[thickness.item()], 'r') as f:
                foam = np.array(f['out']).astype(np.float64)
                foam = torch.from_numpy(foam).to(device)

            # subtract dc component
            foam = foam[foam > 0]
            foam = foam - torch.median(foam)
            foam = torch.clamp(foam, 0.)

            # find calibrated shift to start of scattering layer
            laser, shift = self.load_impulse_response()

            # time bins for data
            propagation_distance = d

            # subtract the distance measured to the SPAD using the hardware setup
            shift = -shift + propagation_distance / 3e8 / self.bin_size

            foam = torch.roll(foam, shift.int().item())
            foam = foam[:len(self.t)]
            foam = foam / torch.max(foam)

            # align peaks to initial guess of model coefficients
            model = self.forward(d, self.ua_init, self.us_init)
            model = F.conv1d(model, laser, padding=(laser.shape[-1]))
            max_vals, _ = torch.max(model, dim=2, keepdim=True)
            model = model / max_vals
            model_idx = torch.argmax(model)
            foam_idx = torch.argmax(foam)
            shift = model_idx - foam_idx
            foam = torch.roll(foam, shift.int().item())
            foam = F.pad(foam, (0, laser.shape[-1]+1))

            data.append(foam[None, None, :])

        data = torch.cat(data, dim=0)
        return data

    def forward(self, d, ua, us, t0=0.):
        '''
        Returns the diffusion model for a slab with finite thickness given by
        Michael S. Patterson, B. Chance, and B. C. Wilson,
        "Time resolved reflectance and transmittance for the noninvasive
        measurement of tissue optical properties,"
        Appl. Opt. 28, 2331-2336 (1989)

        parameters:
        d -  thickness
        ua - absorption coefficient
        us - reduced scattering coefficient
        ze - extrapolation distance
        '''

        t = self.t[None, :, None]
        c = self.c
        us = us * self.us_scale
        t0 = t0 / self.t0_scale
        ze = 2/3 * 1/us * (1 + self.R) / (1 - self.R)
        tshift = torch.clamp(t-t0, 8e-12)

        z0 = 1 / us
        D = 1 / (3 * (ua + us))

        # Photon migration through a turbid slab described by a model
        # based on diffusion approximation.
        # https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-19-4587
        n_dipoles = 20
        ii = torch.arange(-n_dipoles, n_dipoles+1)[None, None, :].to(device)
        z1 = d * (1 - 2 * ii) - 4*ii*ze - z0
        z2 = d * (1 - 2 * ii) - (4*ii - 2)*ze + z0

        dipole_term = z1 * torch.exp(-(z1**2) / (4*D*c*(tshift))) - \
            z2 * torch.exp(-(z2**2) / (4*D*c*(tshift)))

        dipole_term = torch.sum(dipole_term, dim=-1, keepdim=True)  # sum over dipoles

        model = (4*np.pi*D*c)**(-3/2) * torch.clamp(t-t0, 1e-14)**(-5/2) \
            * torch.exp(-ua * c * (t-t0)) \
            * dipole_term
        model = model.squeeze()

        return model[None, None, :]
