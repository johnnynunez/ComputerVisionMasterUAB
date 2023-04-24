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
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from formatDataset import get_kitti_dicts

# import some common detectron2 utilities
from detectron2 import model_zoo

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task C: Inference')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    args = parser.parse_args()

    # --------------------------------- OUTPUT --------------------------------- #
    output_path = os.path.join(current_path, f'Results/Task_c/{args.network}/')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #
    # Register dataset
    classes = [
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'bus',
        'truck',
        'traffic light',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush',
    ]

    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts(subset, pretrained=True))
        MetadataCatalog.get(f"kitti_{subset}").set(thing_classes=classes)

    dataset_dicts = get_kitti_dicts("val")

    # --------------------------------- MODEL --------------------------------- #
    if args.network == 'faster_RCNN':
        model = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    elif args.network == 'mask_RCNN':
        model = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    else:
        print('Network not found')
        exit()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    # --------------------------------- INFERENCE --------------------------------- #
    for d in random.sample(dataset_dicts, 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1])
