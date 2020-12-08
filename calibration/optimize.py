import numpy as np
import torch
from diffusion_model import DiffusionModel
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import configargparse

torch.set_num_threads(2)
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)

p = configargparse.ArgumentParser()

p.add_argument('--experiment_name', type=str, required=True,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--lr', type=float, default=1e-2, help='learning rate. default=1e-2')
p.add_argument('--num_iters', type=int, default=5000,
               help='Number of iterations to run.')
p.add_argument('--convergence', type=float, default=1e-6,
               help='Number of iterations to run.')
p.add_argument('--steps_til_summary', type=int, default=50,
               help='Iterations until tensorboard summary is saved.')
p.add_argument('--us_init', type=float, default=150,
               help='Initial reduced scattering coefficient value. (default 150 1/m).')
p.add_argument('--ua_init', type=float, default=0,
               help='Initial absorption coefficient value. (default 0 1/m).')
p.add_argument('--n_init', type=float, default=1.0,
               help='Value for refractive index of scattering media. (default 1.5).')


p.add_argument('--logging_root', type=str, default='./log', help='root for logging')
opt = p.parse_args()

for arg in vars(opt):
    print(arg, getattr(opt, arg))


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_summary(writer, m, mse, model, data, total_steps):
    plt.switch_backend('agg')
    fig = plt.figure()

    t = np.arange(1, model.shape[-1]+1) * m.bin_size
    for i in range(model.shape[0]):
        plt.plot(t, model[i, :].detach().cpu().numpy().squeeze())
        plt.xlim([0, 10e-9])
        plt.ylim([0, 1.1])

    plt.gca().set_prop_cycle(None)
    for i in range(data.shape[0]):
        plt.plot(t, data[i, :].detach().cpu().numpy().squeeze(), '.', markersize=1)
        plt.xlim([0, 10e-9])
        plt.ylim([0, 1.1])

    writer.add_figure('plots', fig, global_step=total_steps)
    writer.add_scalar('us', m.us * m.us_scale, total_steps)
    writer.add_scalar('ua', m.ua, total_steps)


def optimize():
    log_dir = opt.logging_root
    cond_mkdir(log_dir)

    summaries_dir = os.path.join(log_dir, opt.experiment_name)
    cond_mkdir(summaries_dir)

    m = DiffusionModel(us_init=opt.us_init, ua_init=opt.ua_init,
                       n_init=opt.n_init)
    m.to(device)

    optim = torch.optim.Adam(m.parameters(), lr=opt.lr, amsgrad=True)

    writer = SummaryWriter(summaries_dir)

    converged = False
    converged_eps = opt.convergence
    prev_loss = 1e6
    for ii in range(opt.num_iters):

        loss, model, data = objective(m)
        optim.zero_grad()

        # write summary
        writer.add_scalar('mse', loss, ii)
        if not ii % opt.steps_til_summary:
            write_summary(writer, m, loss, model, data, ii)
            print(f'{ii}: {loss.detach().cpu().numpy():03f}')

        loss.backward()
        optim.step()

        # values should be non-negative
        def clamp_nonnegative(m):
            m.ua.data = torch.clamp(m.ua.data, min=0)
            m.us.data = torch.clamp(m.us.data, min=0)

        m.apply(clamp_nonnegative)

        if torch.abs(loss - prev_loss) < converged_eps:
            converged = True
            break

        prev_loss = loss.clone()

    out = {'us': m.us.detach().cpu().numpy().squeeze().item() * m.us_scale,
           'ua': m.ua.detach().cpu().numpy().squeeze().item(),
           't0': m.t0.detach().cpu().numpy().squeeze(),
           'us_init': opt.us_init,
           'ua_init': opt.ua_init,
           'n_init': opt.n_init,
           'mse': loss.detach().cpu().numpy().squeeze().item(),
           'iters': ii,
           'converged': converged,
           'converged_eps': converged_eps}
    np.save(os.path.join(summaries_dir, 'out.npy'), out)


def objective(m: DiffusionModel):

    # get predicted diffusion model
    model = []
    for idx, thickness in enumerate(m.thicknesses):
        d = 0.0254 * thickness
        model.append(m.forward(d, m.ua, m.us, m.t0[idx]))
    model = torch.cat(model, dim=0)

    # convolve model with measured impulse response
    model = F.conv1d(model, m.laser, padding=(m.laser.shape[-1]))
    max_vals, _ = torch.max(model, dim=2, keepdim=True)
    model = model / max_vals

    # get data
    data = m.data.clone()

    # calculate error
    mse = torch.sum((model - data)**2)
    return mse, model, data


optimize()
