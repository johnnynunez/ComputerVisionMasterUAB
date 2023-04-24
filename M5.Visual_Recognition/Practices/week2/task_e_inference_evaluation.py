# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse

# import some common libraries
import os
import random

import cv2
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from formatDataset import get_kitti_dicts, register_kitti_dataset

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task E: Inference&Evaluation')
    parser.add_argument('--network', type=str, required=True, help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the model, .pth file')
    args = parser.parse_args()

    # --------------------------------- OUTPUT --------------------------------- #
    output_path = os.path.join(current_path, f'Results/Task_e_inference/{args.network}/')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #
    kitti_metadata = register_kitti_dataset("val")  # DUBTE DE SI HEM D'UTILITZAR VAL O TRAIN
    dataset_dicts = get_kitti_dicts("val")

    # --------------------------------- MODEL --------------------------------- #
    model = args.modelpath

    # Get configuration of the model path and load the weights
    cfg = get_cfg()
    cfg.merge_from_file(model)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    cfg.DATASETS.TEST = ("kitti_val",)

    predictor = DefaultPredictor(cfg)

    # --------------------------------- INFERENCE --------------------------------- #
    output_inference = os.path.join(output_path, 'inference/')
    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=kitti_metadata, scale=1.2)  #  !!!!!!!!!!!!!!!
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(output_inference + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1])

    # --------------------------------- EVALUATION --------------------------------- #
    output_evaluation = os.path.join(output_path, 'evaluation/')
    evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir=output_evaluation)
    val_loader = build_detection_test_loader(cfg, "kitti_val")

    print(inference_on_dataset(predictor.model, val_loader, evaluator))
