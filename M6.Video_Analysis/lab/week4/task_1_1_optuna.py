import argparse
import gc
import os
import time

import cv2
import numpy as np
import optuna
from PIL import Image
from optuna.samplers import TPESampler

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


def estimate_flow(N, P, step, distance_type, ref_img, curr_img):
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

    return flow


def objective(trial, valid_combinations):
    valid_combination = trial.suggest_categorical('valid_combination', valid_combinations)
    ref_image = np.array(Image.open(os.path.join(args.frames_path, '000045_10.png')))
    curr_image = np.array(Image.open(os.path.join(args.frames_path, '000045_11.png')))

    results = []

    # Parse the valid_combination string to extract parameter values
    valid_combination_dict = eval(valid_combination)
    block_size = valid_combination_dict['block_size']
    search_area = valid_combination_dict['search_area']
    step_size = valid_combination_dict['step_size']
    distance_type = valid_combination_dict['distance_type']

    print(valid_combination_dict)

    start = time.time()
    flow = estimate_flow(block_size, search_area, step_size, distance_type, ref_image, curr_image)
    end = time.time()
    flow_gt = flow_read(os.path.join(args.gt_path, '000045_10.png'))
    msen, pepn = compute_errors(flow, flow_gt, threshold=3, save_path='./Results/Task1_1/')

    print('MSEN: ', msen)
    print('PEPN: ', pepn)
    print('Time: ', end - start)

    if args.savePlots:
        opticalFlow_arrows(img_10, flow_gt, flow, save_path='./Results/Task1_1/',
                           name='bs_{}_sa_{}_ss_{}_dt_{}'.format(block_size, search_area, step_size, distance_type))
        HSVOpticalFlow2(flow, save_path='./Results/Task1_1/',
                        name='bs_{}_sa_{}_ss_{}_dt_{}'.format(block_size, search_area, step_size, distance_type))

    return msen, pepn


###########################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--block_size', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128],
                        help='size of the square blocks in which the image is divided (N)')

    parser.add_argument('--search_area', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128],
                        help='number of pixels in every direction to define the search area (P)')

    parser.add_argument('--step_size', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128])

    parser.add_argument('--distance_type', type=str, nargs='+', default=['NCC', 'SAD', 'SSD'],
                        help='distance metric to compare the blocks: SAD, SSD, NCC')

    parser.add_argument('--gt_path', type=str, default="/ghome/group03/dataset/OpticalFlow/data_stereo_flow/",
                        help='path to ground truth file for optical flow')

    parser.add_argument('--frames_path', type=str, default="/ghome/group03/dataset/OpticalFlow/frames/",
                        help='path to folder containing the images to estimate the optical flow')

    parser.add_argument('--results_path', type=str, default='./Results/Task1_1/task1_1_results.csv',
                        help='path to save results in a csv file')
    parser.add_argument('--savePlots', type=bool, default=True)

    args = parser.parse_args()

    img_10 = np.array(Image.open(os.path.join(args.frames_path, '000045_10.png')))
    img_11 = np.array(Image.open(os.path.join(args.frames_path, '000045_11.png')))

    results = []

    # random, grid search all of you want sampler https://optuna.readthedocs.io/en/stable/reference/samplers/index.html

    # define the range of values for the hyperparameters to search
    block_sizes = args.block_size
    search_areas = args.search_area
    step_sizes = args.step_size
    similarities = args.distance_type

    # Create a list to store all the valid combinations
    valid_combinations = []

    # Loop through all possible combinations of parameters
    for block_size in block_sizes:
        for search_area in search_areas:
            for step_size in step_sizes:
                for similarity in similarities:
                    # Check if search_area >= block_size and step_size <= block_size
                    if search_area >= block_size and step_size <= block_size and step_size >= block_size / 2:
                        # Create a dictionary to store the valid combination of parameters
                        valid_combination_dict = {'block_size': block_size, 'search_area': search_area,
                                                  'step_size': step_size, 'distance_type': similarity}
                        # Convert the dictionary to a string representation
                        valid_combination_str = str(valid_combination_dict)
                        # Append the valid combination to the list
                        valid_combinations.append(valid_combination_str)

    print('Number of valid combinations: ', len(valid_combinations))

    sampler = TPESampler(seed=42)
    gc.collect()
    try:
        study = optuna.load_study(study_name="OPTICALFLOW", storage="sqlite:///bbdd.db")
    except:
        study = optuna.create_study(
            study_name="OPTICALFLOW",
            directions=["minimize", "minimize"],
            sampler=sampler,
            storage="sqlite:///bbdd.db",
        )
        study.optimize(lambda trial: objective(trial, valid_combinations), n_trials=100, n_jobs=64, gc_after_trial=True)

    df = study.trials_dataframe()
    df.to_csv("opticalFlow_grid.csv")
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_lowest_error = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_lowest_error.number}")
    print(f"\tparams: {trial_with_lowest_error.params}")
    print(f"\tvalues: {trial_with_lowest_error.values}")

    gc.collect()
