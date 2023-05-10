import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# RAFT
# Clone Repo
# https://github.com/princeton-vl/RAFT
# run ./download_models.sh to download pretrained models
# Set path to RAFT in utils/RAFT.py
from utils.RAFT import flow_raft
# DEQ-FLOW
# CLone Repo
# https://github.com/locuslab/deq-flow
# Add path sys
from utils.deq_flow_ import flow_deqflow
# LITEFLOWNET
# CLone Repo
# https://github.com/sniklaus/pytorch-liteflownet
# pip install cupy-cuda12x
from utils.liteflownet_pytorch import flow_liteflownet
# MASKFLOWNET
# Clone Repo
# https://github.com/microsoft/MaskFlownet
# Set path to MaskFlownet in utils/maskflow.py
from utils.maskflow import maskflownet
from utils.optical_flow import compute_errors, flow_read, HSVOpticalFlow2, opticalFlow_arrows
# PYFLOW
# Clone Repo
# https://github.com/pathak22/pyflow
# Set path to pyflow in utils/pyflow.py
from utils.pyflow import flow_pyflow


def flow_LK(img_prev, img_next, colType=0):
    if colType == 1:
        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
        img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take all pixels
    height, width = img_prev.shape[:2]
    p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

    start = time.time()
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_next, p0, None, **lk_params)
    end = time.time()

    p0 = p0.reshape((height, width, 2))
    p1 = p1.reshape((height, width, 2))
    st = st.reshape((height, width))

    flow = p1 - p0
    flow[st == 0] = 0

    return flow, end - start


estimate_flow = {
    'PyFlow': flow_pyflow,
    'LK': flow_LK,
    'MaskFlowNet': maskflownet,
    'RAFT': flow_raft,
    'LiteFlowNet': flow_liteflownet,
    'DEQ-Flow': flow_deqflow
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_path', type=str, default="/ghome/group03/dataset/OpticalFlow/data_stereo_flow/",
                        help='path to ground truth file for optical flow')

    parser.add_argument('--frames_path', type=str, default="/ghome/group03/dataset/OpticalFlow/frames/colored_0/",
                        help='path to folder containing the images to estimate the optical flow')

    parser.add_argument('--results_path', type=str, default='Results/Task1_2/',
                        help='path to save results in a csv file')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Output path for the results
    output_path = os.path.join(current_dir, args.results_path)

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_10 = np.array(Image.open(os.path.join(args.frames_path, '000045_10.png')))
    img_11 = np.array(Image.open(os.path.join(args.frames_path, '000045_11.png')))

    methods = ['PyFlow', 'LK', 'MaskFlowNet', 'RAFT', 'LiteFlowNet']
    # methods = ['DEQ-Flow']

    results = []

    flow_gt = flow_read(os.path.join(args.gt_path, '000045_10.png'))

    # perform grid using the multiple combinations of the parameters using product show progress in tqdm
    for method in methods:
        print('.................Estimating flow for method: {}....................'.format(method))
        output_path_method = os.path.join(output_path, method)

        flow, runtime = estimate_flow[method](img_10, img_11, colType=1)

        msen, pepn = compute_errors(flow, flow_gt, threshold=3, save_path=output_path_method + '/')

        # visualize_flow
        if args.visualize:
            opticalFlow_arrows(img_10, flow_gt, flow, save_path=output_path_method + '/')
            HSVOpticalFlow2(flow, save_path=output_path_method + '/')

        results.append([method, msen, pepn, runtime])

    df = pd.DataFrame(results, columns=['method', 'msen', 'pepn', 'runtime'])

    print(df)

    df.to_csv(output_path + 'results.csv', index=False)
