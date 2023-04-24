import os.path

import cv2
import torch
import torchvision.models as models
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

# wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.init(project="resnet_50_style_transfer")

setup_logger()

# import some common libraries
import argparse

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

# import some common detectron2 utilities
from detectron2 import model_zoo

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task E')
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
        "MSCOCO_val", {}, "../../dataset/COCO/annotations/instances_val2017.json", "../../dataset/COCO/val2017"
    )

    # Register ms coco dataset test
    register_coco_instances(
        "MSCOCO_test", {}, "../../dataset/COCO/annotations/image_info_test2017.json", "../../dataset/COCO/test2017"
    )

    # Config
    cfg = get_cfg()
    current_path = os.getcwd()
    if args.network == 'faster_RCNN':
        output_path = os.path.join(current_path, 'Results/Task_e/faster_RCNN/')
        # get if the path exists, if not create it
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = os.path.join(current_path, 'Results/Task_e/mask_RCNN/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("MSCOCO_test",)
    cfg.MODEL.DEVICE = 'cpu'
    # Predictor
    predictor = DefaultPredictor(cfg)

    """
    # Evaluator
    evaluator = COCOEvaluator("MSCOCO_val", cfg, False, output_dir=output_path)

    # Evaluate the model
    val_loader = build_detection_test_loader(cfg, "MSCOCO_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    """

    # dataset_dicts = get_ooc_dicts('val', pretrained=True)
    # dataset_dicts = DatasetCatalog.get("MSCOCO_test")

    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()

    output_path = os.path.join(current_path, '../Results/Task_e/style_transfer/post_detection/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    read_path = os.path.join(current_path, '../Results/Task_e/style_transfer/')
    # read images from read_path
    image_list = [
        os.path.join(read_path, filename)
        for filename in os.listdir(read_path)
        if os.path.isfile(os.path.join(read_path, filename)) and filename.endswith('.png')
    ]

    for image in image_list:
        img_1 = image
        # TODO: POSSIBILITAT DE FER SEGMENTATION O DIRECTAMENT LA IMATGE
        # PREDICCIO DE LA IMATGE
        # img_1 to numpy array
        # conver image to cv2
        img_1 = cv2.imread(img_1)

        outputs_img1 = predictor(img_1)
        v = Visualizer(img_1[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        try:
            output_img1 = v.draw_instance_predictions(outputs_img1["instances"].to("cpu")[0])
        except:
            print("No objects detected")
            continue

        # get only segmentation and put pixels outside the mask to white

        image_name = os.path.splitext(image)[0].split('/')[-1]

        cv2.imwrite(output_path + "final_detect_" + image_name + ".png", output_img1.get_image()[:, :, ::-1])
