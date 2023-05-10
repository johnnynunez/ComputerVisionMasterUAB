# python main.py -r task_1 -m Gaussian -p 0.25 -c gray -a 5 
# OUTPUT: ./results/task_1_Gaussian/
# {5.0: [0.4039613269069583, 0.27754962987557735]}

# python main.py -r task_2 -m AdaptiveGaussian -p 0.25 -c gray -a 5 --rho 0.05
# OUTPUT: ./results/task_2_AdaptiveGaussian/
# {5.0: {0.05: [0.5030393154694073, 0.2718995737835347]}}

# python main.py -r task_3 -m SOTA -p 0.25 -c gray -e MOG2 -a 0
# OUTPUT: ./results/task_3_SOTA/
# {0.0: [0.7081791718896054, 0.3892347646742538]}

# python main.py -r task_4 -m Gaussian -p 0.25 -c RGB -a 5
# {5.0: [0.3666943288806497, 0.2923283266021336]}


# To do grid search
# task_1
# python main.py -r task_1 -m Gaussian -p 0.25 -c gray -g True -a 5 10


import argparse
import os

import yaml

from models import Gaussian, AdaptiveGaussian, SOTA
from utils.rendering import rendering_video
from utils.util import visualizeTask1, visualizeTask2, visualizeTask4

TOTAL_FRAMES_VIDEO = 2141


def main(cfg):
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_path, "results", f'{cfg["run_name"]}_{cfg["run_mode"]}')
    os.makedirs(output_path, exist_ok=True)
    print(
        f"Run Name: {cfg['run_name']} - Run Mode: {cfg['run_mode']} - Colorspace: {cfg['colorspaces']} - Alphas :{cfg['alphas']} - Rhos: {cfg['rhos']} Output Path: {output_path}")
    print("----------------------------------------")

    frames_modelling = int(TOTAL_FRAMES_VIDEO * cfg["percentatge"])

    alpha_list = cfg["alphas"]
    rho_list = cfg["rhos"]
    colorspace_list = cfg["colorspaces"]

    dic = {}

    for alpha in alpha_list:

        if cfg["run_mode"] == "Gaussian":
            print("Gaussian Function")
            print("----------------------------------------")

            if cfg["run_name"] == "task_4":
                dic[alpha] = {}

                for colorspace in colorspace_list:
                    model = Gaussian(cfg['paths']['video_path'], frames_modelling, alpha=float(alpha),
                                     colorspace=colorspace,
                                     checkpoint=f"{colorspace}_{cfg['percentatge']}")
                    map, iou = rendering_video(cfg, model, frames_modelling, output_path,
                                               cfg['paths']['annotations_path'])
                    dic[alpha][colorspace] = [map, iou]

                    print("Done for colorspace = ", colorspace)
                    print("----------------------------------------")

            elif cfg["run_name"] == "task_1":
                model = Gaussian(cfg['paths']['video_path'], frames_modelling, alpha=float(alpha), colorspace="gray",
                                 checkpoint=f"gray_{cfg['percentatge']}")
                map, iou = rendering_video(cfg, model, frames_modelling, output_path, cfg['paths']['annotations_path'])
                dic[alpha] = [map, iou]

            else:
                raise ValueError("Invalid run name")


        elif cfg["run_mode"] == "AdaptiveGaussian":
            dic[alpha] = {}
            for rho in rho_list:

                for colorspace in colorspace_list:
                    if colorspace == "gray":
                        print("Adaptive Gaussian Function")
                        print("----------------------------------------")
                        model = AdaptiveGaussian(cfg['paths']['video_path'], frames_modelling, p=float(rho),
                                                 alpha=float(alpha),
                                                 colorspace=colorspace, checkpoint=f"{colorspace}_{cfg['percentatge']}")

                        map, iou = rendering_video(cfg, model, frames_modelling, output_path,
                                                   cfg['paths']['annotations_path'])
                        dic[alpha][rho] = [map, iou]

                print("Done for rho = ", rho)
                print("----------------------------------------")

            print("Done for all rhos")
            print("----------------------------------------")

        elif cfg["run_mode"] == "SOTA":
            model = SOTA(cfg['paths']['video_path'], frames_modelling, checkpoint=None, method=cfg['sota_method'])
            map, iou = rendering_video(cfg, model, frames_modelling, output_path, cfg['paths']['annotations_path'])

        else:
            raise ValueError("Invalid run mode")

        print("Done for alpha = ", alpha)
        print("----------------------------------------")

    print("Done for all alphas")
    print(dic)
    if cfg['grid']:
        if cfg['run_name'] == 'task_1':
            visualizeTask1(dic, output_path)
        elif cfg['run_name'] == 'task_2':
            visualizeTask2(dic, output_path)
        elif cfg['run_name'] == 'task_4':
            visualizeTask4(dic, output_path)
    print("----------------------------------------")


if __name__ == "__main__":
    # check ffmepg in your system

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--run_mode", required=True, type=str, help="Gaussian Modelling")
    parser.add_argument("-r", "--run_name", required=True, type=str, help="Run Folder Name")
    parser.add_argument("--config", default="configs/config.yml")
    parser.add_argument("-s", "--save", default=True, type=bool, help="Save the video or not")
    parser.add_argument("-d", "--display", default=False, type=bool, help="Show the video or not")
    parser.add_argument("-p", "--percentatge", required=True, default=False, type=float,
                        help="Percentatge of video to use background")
    parser.add_argument("-e", "--sota_method", default="MOG", type=str,
                        help="SOTA method to use (MOG, MOG2, LSBP, KNN, GMG)")
    parser.add_argument("-a", "--alpha", default=5, nargs="+", type=float, help="Alpha Thresholding")
    parser.add_argument("--rho", default=0.05, nargs="+", type=float, help="Rho Thresholding")
    parser.add_argument("-c", "--colorspaces", nargs='+', default="gray", type=str,
                        help="Colorspace to use (gray, RGB, YCRCB, HSV, YUV)")
    parser.add_argument("-g", "--grid", default=False, type=bool, help="Show the grid or not")

    args = parser.parse_args()

    # get the path of this file
    path = os.path.dirname(os.path.realpath(__file__))
    path_config = os.path.join(path, args.config)

    with open(path_config) as f:
        config = yaml.safe_load(f)

    config["run_mode"] = args.run_mode
    config["run_name"] = args.run_name
    config["save"] = args.save
    config["display"] = args.display
    config["percentatge"] = args.percentatge
    config["sota_method"] = args.sota_method
    config["alphas"] = args.alpha
    config["rhos"] = args.rho
    config["colorspaces"] = args.colorspaces
    config["grid"] = args.grid

    main(config)
