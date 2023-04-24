import random

import cv2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from pycocotools.coco import COCO

setup_logger()

# import some common libraries
import argparse

import numpy as np
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# import some common detectron2 utilities
from detectron2 import model_zoo

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task D')
    parser.add_argument('--network', type=str, default='mask_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()

    # Register dataset
    """classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"""

    classes = {
        0: '__background__',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',
        69: 'microwave',
        70: 'oven',
        71: 'toaster',
        72: 'sink',
        73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush',
    }

    # Register ms coco dataset val
    register_coco_instances(
        "MSCOCO_val", {}, "/ghome/group03/annotations/instances_val2017.json", "/ghome/group03/val2017"
    )

    # Register ms coco dataset test
    register_coco_instances(
        "MSCOCO_test", {}, "/ghome/group03/annotations/image_info_test2017.json", "/ghome/group03/test2017"
    )

    # Config
    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = 'Results/Task_d/faster_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = 'Results/Task_d/mask_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("MSCOCO_test",)

    # Predictor
    predictor = DefaultPredictor(cfg)

    # Evaluator
    evaluator = COCOEvaluator("MSCOCO_val", cfg, False, output_dir=output_path)

    # Evaluate the model
    val_loader = build_detection_test_loader(cfg, "MSCOCO_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # dataset_dicts = get_ooc_dicts('val', pretrained=True)
    dataset_dicts = DatasetCatalog.get("MSCOCO_val")
    coco = COCO("/ghome/group03/annotations/instances_val2017.json")

    for d in random.sample(dataset_dicts, 300):
        if len(d["annotations"]) > 0:
            im = cv2.imread(d["file_name"])

            # read GT mask for the biggest bounding box
            d["annotations"][0]["image_id"] = d["image_id"]
            mask = coco.annToMask(d["annotations"][0])

            # write the image at the output path
            cv2.imwrite(output_path + d["file_name"].split('/')[-1], im)
            outputs = predictor(im)

            classes_im = outputs["instances"].to("cpu").pred_classes.tolist()
            print(classes_im)

            a = np.where(mask != False)

            # Duplicate the image and put the pixels outside the bounding box to black
            im2b = np.copy(im)
            im2b[: np.min(a[0]), :] = 0  # set top rows to black
            im2b[np.max(a[0]) + 1:, :] = 0  # set bottom rows to black
            im2b[:, : np.min(a[1])] = 0  # set left columns to black
            im2b[:, np.max(a[1]) + 1:] = 0  # set right columns to black

            # Duplicate the image and put the pixels outside the mask to black
            im2c = np.copy(im)
            im2c[np.where(mask == False)] = 0

            # Duplicate the image and put the pixels outside the mask to black and add random noise to the pixels outside the bounding box
            im2d = np.copy(im)
            im2d[np.where(mask == False)] = 0
            im2d[: np.min(a[0]), :, :] = np.random.randint(
                low=0, high=256, size=(np.min(a[0]), im.shape[1], 3), dtype=np.uint8
            )
            im2d[np.max(a[0]) + 1:, :, :] = np.random.randint(
                low=0, high=256, size=(im.shape[0] - np.max(a[0]) - 1, im.shape[1], 3), dtype=np.uint8
            )
            im2d[:, : np.min(a[1]), :] = np.random.randint(
                low=0, high=256, size=(im.shape[0], np.min(a[1]), 3), dtype=np.uint8
            )
            im2d[:, np.max(a[1]) + 1:, :] = np.random.randint(
                low=0, high=256, size=(im.shape[0], im.shape[1] - np.max(a[1]) - 1, 3), dtype=np.uint8
            )

            # compute the outputs for each of the 4 images and save them

            try:
                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(output_path + "pred_" + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

                outputs2b = predictor(im2b)
                v = Visualizer(im2b[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs2b["instances"].to("cpu"))
                cv2.imwrite(output_path + "bBB_" + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

                outputs2c = predictor(im2c)
                v = Visualizer(im2c[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs2c["instances"].to("cpu"))
                cv2.imwrite(output_path + "bM_" + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

                outputs2d = predictor(im2d)
                v = Visualizer(im2d[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs2d["instances"].to("cpu"))
                cv2.imwrite(output_path + "_noise_" + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

                print("Processed image: " + d["file_name"].split('/')[-1])

            except:
                print("No detection on image: " + d["file_name"].split('/')[-1])
