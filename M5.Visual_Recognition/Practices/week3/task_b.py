import random

import cv2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

setup_logger()

# import some common libraries
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import seaborn as sns
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

# import some common detectron2 utilities
from detectron2 import model_zoo

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task B')
    parser.add_argument('--network', type=str, default='mask_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()

    # Register dataset
    """
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    """

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

    """
    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"coco2017_{subset}", lambda subset=subset: get_coco_dicts(subset, pretrained=True))
        MetadataCatalog.get(f"coco2017_{subset}").set(thing_classes=list(classes.values()))
    """
    register_coco_instances(
        "coco2017_test", {}, "/ghome/group03/annotations/image_info_test2017.json", "/ghome/group03/test2017"
    )
    register_coco_instances(
        "coco2017_val", {}, "/ghome/group03/annotations/instances_val2017.json", "/ghome/group03/val2017"
    )

    # Config
    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = 'Results/Task_b/faster_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = 'Results/Task_b/mask_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("coco2017_test",)

    # Predictor
    predictor = DefaultPredictor(cfg)

    # --------------------------------- INFERENCE --------------------------------- #

    co_ocurrence_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    dataset_dicts_val = DatasetCatalog.get('coco2017_val')  # get_coco_dicts('val', pretrained=True)
    dataset_dicts_test = DatasetCatalog.get('coco2017_test')

    for d in random.sample(dataset_dicts_val, len(dataset_dicts_val)):
        im = cv2.imread(d["file_name"])
        """
        outputs = predictor(im)
        print("Processed image: " + d["file_name"].split('/')[-1])

        # For each predicted object in the image:
        for object_class1 in outputs["instances"].to("cpu").pred_classes.tolist():
        """

        for obj1 in d['annotations']:
            object_class1 = obj1['category_id']

            if object_class1 != 0:  # Avoid background
                # We look for all the other objects that also appear in that image:
                for obj2 in d['annotations']:
                    object_class2 = obj2['category_id']

                    if object_class2 != 0:  # Avoid background
                        if object_class1 != object_class2:
                            co_ocurrence_matrix[object_class1][object_class2] = (
                                    co_ocurrence_matrix[object_class1][object_class2] + 1
                            )

    co_ocurrence_matrix = np.where(co_ocurrence_matrix == 0, np.inf, co_ocurrence_matrix)
    co_ocurrence_matrix = 1 / (co_ocurrence_matrix + 1e-7)
    mask = np.nonzero(np.sum(co_ocurrence_matrix, axis=0) > 15)[0]
    co_ocurrence_matrix = co_ocurrence_matrix[np.ix_(mask, mask)]

    top_classes_names = np.array(list(classes.values()))[mask].tolist()
    top_classes_ids = np.array(list(classes.keys()))[mask].tolist()

    fig, ax = plt.subplots(figsize=(15, 15), dpi=200)
    ax.set_title(
        "(1/Co-ocurrence) matrix values for the COCO objects \n(top " + str(len(mask)) + " (1/co-ocurrences))", size=20
    )
    heatmap = sns.heatmap(
        co_ocurrence_matrix,
        annot=False,
        linewidth=0.5,
        xticklabels=top_classes_names,
        yticklabels=top_classes_names,
        ax=ax,
    )
    heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), rotation=0, fontsize=15)
    heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), rotation=45, fontsize=15)
    fig.savefig("task_b_heatmap.png")

    image_count = 0
    for d in random.sample(dataset_dicts_val, len(dataset_dicts_val)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        is_class = False
        for obj1 in d['annotations']:
            object_class1 = obj1['category_id']

            if object_class1 in top_classes_ids:
                is_class = True
                break

        if not is_class:
            continue

        # We look for a second image
        for d2 in random.sample(dataset_dicts_val, len(dataset_dicts_val)):
            if d["file_name"] == d2["file_name"]:
                continue

            im2 = cv2.imread(d2["file_name"])
            # outputs2 = predictor(im2)

            is_class2 = False
            class_pos = 0
            selected_obj2 = None
            for o_i, obj2 in enumerate(d2['annotations']):
                if 'segmentation' not in obj2:
                    continue

                if len(obj2['segmentation']) == 0:
                    continue

                object_class2 = obj2['category_id']

                if object_class2 in top_classes_ids:
                    is_class2 = object_class2
                    class_pos = o_i
                    selected_obj2 = obj2
                    break

            if is_class2 == False:
                continue

            class_index = is_class2

            rles = mask_utils.frPyObjects(selected_obj2['segmentation'], im2.shape[1], im2.shape[0])
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle).astype(bool)

            # mask = outputs2["instances"].to("cpu").pred_masks[class_pos]

            a = np.where(mask != False)
            try:
                im2b = im2[np.min(a[0]): np.max(a[0]) + 1, np.min(a[1]): np.max(a[1]) + 1]
                maskb = mask[np.min(a[0]): np.max(a[0]) + 1, np.min(a[1]): np.max(a[1]) + 1]
            except:
                print('Empty mask')
                continue

            im2c = np.where(np.repeat(np.expand_dims(maskb, axis=2), 3, axis=-1), im2b, 0)

            if im2c.shape[0] >= im.shape[0] or im2c.shape[1] >= im.shape[1]:
                continue

            im3 = np.copy(im)
            p0, p1 = (
                np.random.uniform(low=0, high=im.shape[0] - im2c.shape[0], size=(1)).astype(int)[0],
                np.random.uniform(low=0, high=im.shape[1] - im2c.shape[1], size=(1)).astype(int)[0],
            )
            im3[p0: p0 + im2c.shape[0], p1: p1 + im2c.shape[1]] = im2c

            im3 = np.where(im3 == 0, im, im3)

            outputs3 = predictor(im3)
            v = Visualizer(im3[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs3["instances"].to("cpu"))

            cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

            print("Processed image: " + d["file_name"].split('/')[-1])

            image_count = image_count + 1

            break

        if image_count == 5000:
            break
