# Clone the repo
# https://github.com/krasserm/perceiver-io.git

# CREATE AN ENVIRONMENT BECAUSE IT STILL NOT WORK WITH TORCH 2.0
# conda env create -f environment.yml
# conda activate perceiver-io


import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from perceiver.data.vision.optical_flow import OpticalFlowProcessor
from perceiver.model.vision.optical_flow import convert_config, OpticalFlow
from transformers import AutoConfig

from utils.optical_flow import compute_errors, flow_read, HSVOpticalFlow2, opticalFlow_arrows


def perceiver_io(img1, img2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load pretrained model configuration from the Hugging Face Hub
    config = AutoConfig.from_pretrained("deepmind/optical-flow-perceiver")
    # Convert configuration, instantiate model and load weights
    model = OpticalFlow(convert_config(config)).eval().to(device)

    # Create optical flow processor
    processor = OpticalFlowProcessor(patch_size=tuple(config.train_size))

    frame_pair = (img1, img2)

    start = time.time()
    optical_flow = processor.process(model, image_pairs=[frame_pair], batch_size=1, device=device).numpy()[0]
    end = time.time()

    return optical_flow, end - start


estimate_flow = {
    'Perceiver-IO': perceiver_io,
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

    methods = ['Perceiver-IO']

    results = []

    flow_gt = flow_read(os.path.join(args.gt_path, '000045_10.png'))

    # perform grid using the multiple combinations of the parameters using product show progress in tqdm
    for method in methods:
        print('.................Estimating flow for method: {}....................'.format(method))
        output_path_method = os.path.join(output_path, method)

        flow, runtime = estimate_flow[method](img_10, img_11)

        msen, pepn = compute_errors(flow, flow_gt, threshold=3, save_path=output_path_method + '/')

        # visualize_flow
        if args.visualize:
            opticalFlow_arrows(img_10, flow_gt, flow, save_path=output_path_method + '/')
            HSVOpticalFlow2(flow, save_path=output_path_method + '/')

        results.append([method, msen, pepn, runtime])

    df = pd.DataFrame(results, columns=['method', 'msen', 'pepn', 'runtime'])

    print(df)

    df.to_csv(output_path + 'results_perceiver.csv', index=False)
