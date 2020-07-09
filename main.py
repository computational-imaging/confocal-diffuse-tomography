import os
import torch
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
from cdt_reconstruction import CDTReconstruction
cudnn.benchmark = True

scene_choices = ['cones', 'letter_s', 'letters_ut', 'mannequin',
                 'letter_t', 'resolution_50', 'resolution_70'] +\
                [f'letter_u_{x}' for x in np.arange(50, 82, 2)]

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='letter_s',
                    choices=scene_choices,
                    help='name of scene to reconstruct.')
parser.add_argument('--cpu', action='store_true', default=False, help='Force run on CPU, default=False')
parser.add_argument('--gpu_id', type=int, default=0, help='index of which GPU to run on, default=0')
parser.add_argument('--pause', type=int, default=5, help='how long to display figure, default=5 (seconds)')
opt = parser.parse_args()
print('Confocal diffuse tomography reconstruction')
print('\n'.join(["\t%s: %s" % (key, value) for key, value in vars(opt).items()]))

# check to see if GPU is available
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu_id)
if opt.cpu:
    device = torch.device('cpu')
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cdt = CDTReconstruction(opt.scene, pause=opt.pause, device=device)
    cdt.run()
    return


if __name__ == '__main__':
    main()
