import sys

# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = '/ghome/group03/mcv-m6-2023-team6/week4/deq_flow/code_v_1_0/core_'
# to CUDA\vX.Y\bin
# os.environ['PATH'] = r'path\to\your\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin' + ';' + os.environ['PATH']

# SET REPO PATH
sys.path.append(repoRoot)

import argparse
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

from deq_flow.code_v_1_0.core_.deq import get_model


# from deq_flow.code_v_1_0.core.utils.utils import InputPadder


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


@torch.no_grad()
def flow_deqflow(image1, image2, colType=1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true', help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse', action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true', help="Enable warm start.")

    parser.add_argument('--name', default='deq-flow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output_path', help="output path for evaluation")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--large', action='store_true', help='use large model')
    parser.add_argument('--huge', action='store_true', help='use huge model')
    parser.add_argument('--gigantic', action='store_true', help='use gigantic model')

    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])

    parser.add_argument('--eval_interval', type=int, default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int, default=20000, help="saving interval")
    parser.add_argument('--time_interval', type=int, default=500, help="timing interval")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--schedule', type=str, default="onecycle", help="learning rate schedule")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0,
                        help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--wnorm', action='store_true', help="use weight normalization")
    parser.add_argument('--f_solver', default='anderson', type=str, choices=['anderson', 'broyden'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str, choices=['anderson', 'broyden'],
                        help='backward solver to use')
    parser.add_argument('--f_thres', type=int, default=36, help='forward pass solver threshold')
    parser.add_argument('--b_thres', type=int, default=36, help='backward pass solver threshold')
    parser.add_argument('--stop_mode', type=str, default="abs", help="fixed-point convergence stop mode")
    parser.add_argument('--eval_factor', type=float, default=1.5,
                        help="factor to scale up the f_thres at test for better precision.")

    parser.add_argument('--sliced_core', action='store_true', help="use the sliced core implementation.")
    parser.add_argument('--ift', action='store_true', help="use implicit differentiation.")
    parser.add_argument('--safe_ift', action='store_true', help="use a safer function for IFT to avoid segment fault.")
    parser.add_argument('--n_losses', type=int, default=1,
                        help="number of loss terms (uniform spaced, 1 + fixed point correction).")
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help="indexing for fixed point correction.")
    parser.add_argument('--sup_all', action='store_true', help="supervise all the trajectories by Phantom Grad.")
    parser.add_argument('--phantom_grad', type=int, nargs='+', default=[1], help="steps of Phantom Grad")
    parser.add_argument('--tau', type=float, default=1.0, help="damping factor for unrolled Phantom Grad")

    parser.add_argument('--sradius_mode', action='store_true', help="monitor spectral radius during validation")

    args = parser.parse_args()

    args.viz = True
    args.name = 'deq-flow-H-kitti'
    args.stage = 'kitti'
    args.viz_set = ['kitti']
    args.restore_ckpt = '/ghome/group03/mcv-m6-2023-team6/week4/deq_flow/deq-flow-H-kitti.pth'
    args.gpus = [0]
    args.wnorm = True
    args.f_thres = 36
    args.f_solver = 'broyden'
    args.huge = True

    DEQFlow = get_model(args)
    model = nn.DataParallel(DEQFlow(args), device_ids=args.gpus)

    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()

    model = model.module
    model.eval()

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    padder = InputPadder(image1.shape, mode='kitti')
    image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

    start = time.time()
    _, flow_pr, _ = model(image1, image2)
    end = time.time()

    flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

    return flow, end - start
