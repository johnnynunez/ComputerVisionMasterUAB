import argparse
import os
import time
from itertools import product

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.optical_flow import compute_errors, flow_read, HSVOpticalFlow2, opticalFlow_arrows


def estimate_block_flow(block_size, distance_type, blocks_pos, ref_img, curr_img):
    tlx_ref = blocks_pos['tlx_ref']
    tly_ref = blocks_pos['tly_ref']
    init_tlx_curr = blocks_pos['init_tlx_curr']
    init_tly_curr = blocks_pos['init_tly_curr']
    end_tlx_curr = blocks_pos['end_tlx_curr']
    end_tly_curr = blocks_pos['end_tly_curr']

    if distance_type == 'NCC':
        corr = cv2.matchTemplate(
            curr_img[init_tly_curr:end_tly_curr + block_size, init_tlx_curr:end_tlx_curr + block_size],
            ref_img[tly_ref:tly_ref + block_size, tlx_ref:tlx_ref + block_size],
            method=cv2.TM_CCORR_NORMED)

        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        flow_x = x + init_tlx_curr - tlx_ref
        flow_y = y + init_tly_curr - tly_ref

    else:

        min_dist = np.inf
        for y_curr in range(init_tly_curr, end_tly_curr):
            for x_curr in range(init_tlx_curr, end_tlx_curr):
                wind_ref = ref_img[tly_ref:tly_ref + block_size, tlx_ref:tlx_ref + block_size]
                wind_curr = curr_img[y_curr:y_curr + block_size, x_curr:x_curr + block_size]

                if distance_type == 'SAD':
                    dist = np.sum(np.abs(wind_ref - wind_curr))
                elif distance_type == 'SSD':
                    dist = np.sum((wind_ref - wind_curr) ** 2)
                else:
                    raise ValueError('This distance is unknown')

                if dist < min_dist:
                    min_dist = dist
                    flow_x = x_curr - tlx_ref
                    flow_y = y_curr - tly_ref

    return [flow_x, flow_y]


def estimate_flow(motion_type, N, P, step, distance_type, ref_img, curr_img):
    h, w = ref_img.shape
    flow = np.zeros(shape=(h, w, 2))

    for tly_ref in range(0, h - N, step):
        for tlx_ref in range(0, w - N, step):
            blocks_pos = {
                'tlx_ref': tlx_ref,
                'tly_ref': tly_ref,
                'init_tlx_curr': max(tlx_ref - P, 0),
                'init_tly_curr': max(tly_ref - P, 0),
                'end_tlx_curr': min(tlx_ref + P, w - N),
                'end_tly_curr': min(tly_ref + P, h - N)
            }

            flow[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N, :] = estimate_block_flow(N, distance_type, blocks_pos,
                                                                                    ref_img, curr_img)

    if motion_type == 'backward':
        flow = -flow

    return flow


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--block_size', type=int, nargs='+', default=[16],
                        help='size of the square blocks in which the image is divided (N)')

    parser.add_argument('--search_area', type=int, nargs='+', default=[32],
                        help='number of pixels in every direction to define the search area (P)')

    parser.add_argument('--step_size', type=int, nargs='+', default=[8])

    parser.add_argument('--motion_type', type=str, nargs='+', default=['forward'],
                        help='motion type to use: forward or backward')

    parser.add_argument('--distance_type', type=str, nargs='+', default=['NCC'],
                        help='distance metric to compare the blocks: SAD, SSD, NCC')

    parser.add_argument('--gt_path', type=str, default="/ghome/group03/dataset/OpticalFlow/data_stereo_flow/",
                        help='path to ground truth file for optical flow')

    parser.add_argument('--frames_path', type=str, default="/ghome/group03/dataset/OpticalFlow/frames/",
                        help='path to folder containing the images to estimate the optical flow')

    parser.add_argument('--results_path', type=str, default='./Results/Task1_1/task1_1_results.csv',
                        help='path to save results in a csv file')
    parser.add_argument('--visualize', type=bool, default=False)

    args = parser.parse_args()

    img_10 = np.array(Image.open(os.path.join(args.frames_path, '000045_10.png')))
    img_11 = np.array(Image.open(os.path.join(args.frames_path, '000045_11.png')))

    results = []

    # perform grid using the multiple combinations of the parameters using product show progress in tqdm
    for motion_type, N, P, step, distance_type in product(args.motion_type, args.block_size, args.search_area,
                                                          args.step_size, args.distance_type):
        print(
            'Estimating flow for motion_type: {}, N: {}, P: {}, step_size: {}, distance_type: {}'.format(motion_type, N,
                                                                                                         P, step,
                                                                                                         distance_type))

        if motion_type == 'forward':
            ref_image = img_10
            curr_image = img_11

        elif motion_type == 'backward':
            ref_image = img_11
            curr_image = img_10

        else:
            raise ValueError("Invalid motion type. Possible: 'forward' or 'backward'")

        start = time.time()
        flow = estimate_flow(motion_type, N, P, step, distance_type, ref_image, curr_image)
        end = time.time()
        flow_gt = flow_read(os.path.join(args.gt_path, '000045_10.png'))

        msen, pepn = compute_errors(flow, flow_gt, threshold=3, save_path='./Results/Task1_1/')

        # visualize_flow
        if args.visualize:
            opticalFlow_arrows(img_10, flow_gt, flow, save_path='./Results/Task1_1/')
            HSVOpticalFlow2(flow, save_path='./Results/Task1_1/')

        results.append([motion_type, distance_type, N, P, step, msen, pepn, end - start])

    df = pd.DataFrame(results,
                      columns=['motion_type', 'distance_type', 'block_size', 'search_area', 'Step_size', 'msen', 'pepn',
                               'runtime'])

    print(df)

    # check if the folder exists, if not create it
    if not os.path.exists(os.path.dirname(args.results_path)):
        os.makedirs(os.path.dirname(args.results_path))

    df.to_csv(args.results_path, index=False)
