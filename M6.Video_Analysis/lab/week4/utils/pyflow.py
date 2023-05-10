import sys
import time

import numpy as np

# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = '/ghome/group03/mcv-m6-2023-team6/week4/pyflow'
# to CUDA\vX.Y\bin
# os.environ['PATH'] = r'path\to\your\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin' + ';' + os.environ['PATH']

# SET REPO PATH
sys.path.append(repoRoot)

import pyflow as pyflow


def flow_pyflow(img_prev, img_next, colType=0):
    img_prev = img_prev.astype(float) / 255.
    img_next = img_next.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = colType  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    if colType == 0:
        img_prev = np.expand_dims(img_prev, axis=2)
        img_next = np.expand_dims(img_next, axis=2)

    start = time.time()
    u, v, _ = pyflow.coarse2fine_flow(
        img_prev, img_next, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    end = time.time()

    return flow, end - start
