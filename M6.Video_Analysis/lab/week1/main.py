import argparse
import os

import yaml

from utils.metrics import generate_noisy_boxes
from utils.rendering import rendering_video


def main(cfg):
    os.makedirs(f"runs/{cfg['run_name']}/", exist_ok=True)
    print(f"Run Name: {cfg['run_name']}")

    gt_boxes, total = load_from_xml_rendering(cfg["paths"]["annotations_path"])
    gt = [gt_boxes, total]

    if cfg["noisy"]:
        print(f"Run Mode: {'Noisy'}")
        predicted_boxes = generate_noisy_boxes(
            gt_boxes,
            del_prob=cfg['noisy_values']['del_prob'],
            gen_prob=cfg['noisy_values']['gen_prob'],
            mean=cfg['noisy_values']['mean'],
            std=cfg['noisy_values']['std'],
        )

    else:
        print(f"Run Mode: {cfg['run_mode']}")
        predicted_boxes = load_from_txt_rendering(cfg["detections"][args.run_mode])

    rendering_video(
        f"runs/{cfg['run_name']}/",
        gt,
        predicted_boxes,
        cfg["paths"]["video_path"],
        save=cfg["save"],
        display=cfg["display"],
    )

    print("Done!")
    print("----------------------------------------")


if __name__ == "__main__":
    # check ffmepg in your system

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--run_mode", required=True, type=str, help="Yolo, RCNN or SSD")
    parser.add_argument("-r", "--run_name", required=True, type=str, help="Run Folder Name")
    parser.add_argument("-c", "--config", default="configs/config.yml")
    parser.add_argument("-s", "--save", default=True, type=bool, help="Save the video or not")
    parser.add_argument("-d", "--display", default=False, type=bool, help="Show the video or not")
    parser.add_argument("-n", "--noisy", default=False, type=bool, help="Noisy or not")
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
    config["noisy"] = args.noisy

    main(config)
