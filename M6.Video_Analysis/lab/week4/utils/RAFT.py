import os
import sys

# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = '/ghome/group03/mcv-m6-2023-team6/week4/RAFT'
# to CUDA\vX.Y\bin
# os.environ['PATH'] = r'path\to\your\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin' + ';' + os.environ['PATH']

# SET REPO PATH
sys.path.append(repoRoot)
sys.path.append(os.path.join(repoRoot, 'core'))
# sys.path.append(os.path.join(repoRoot, 'core/utils'))

import argparse
import numpy as np
import torch
import time

from core.raft import RAFT
from core.utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(img):
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def flow_raft(image1, image2, model='/ghome/group03/mcv-m6-2023-team6/week4/RAFT/models/raft-kitti.pth', colType=1,
              small=True, mixed_precision=True, alternate_corr=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default=model)
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        shape = image1.shape
        image1 = load_image(image1)
        image2 = load_image(image2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        # Resize flow_up to original image1 size
        flow_up = torch.nn.functional.interpolate(flow_up, size=shape[:2], mode='bilinear', align_corners=True)
        end = time.time()

    return flow_up.cpu().numpy().squeeze().transpose(1, 2, 0), end - start
