# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse

# import some common libraries
import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from CityAI_dataset import get_CityAI_dicts
from detectron2.evaluation import COCOEvaluator

# import some common detectron2 utilities
from detectron2 import model_zoo

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))


# Modify COCOEvaluator to compute only the AP of the bounding boxes, not the masks (we want object detection, not instance segmentation)
class MyEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._tasks = ("bbox",)


if __name__ == '__main__':

    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task 1: Inference')
    parser.add_argument('--network', type=str, default='retinaNet', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument("--save_vis", type=bool, default=False, help="Save visualizations")
    args = parser.parse_args()

    # --------------------------------- OUTPUT --------------------------------- #
    output_path = os.path.join(current_path, f'Results/Task_1_1/{args.network}/')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #

    classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
               'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    for subset in ["train", "val"]:
        DatasetCatalog.register(f"CityAI_{subset}", lambda subset=subset: get_CityAI_dicts(subset))
        MetadataCatalog.get(f"CityAI_{subset}").set(thing_classes=classes)

    dataset_dicts = get_CityAI_dicts("val")

    # --------------------------------- MODEL --------------------------------- #
    cfg = get_cfg()
    if args.network == 'faster_RCNN':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    elif args.network == 'mask_RCNN':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    elif args.network == 'retinaNet':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4
    elif args.network == 'faster_RCNN_R50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    elif args.network == 'mask_RCNN_R50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif args.network == 'retinaNet_R50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4
    elif args.network == 'faster_RCNN_R101':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    elif args.network == 'mask_RCNN_R101':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    else:
        raise ValueError('Network not implemented')

    # elif args.network == 'SSD512':
    #     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/ssd512_coco17_tpu-8.yaml"))
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/ssd512_coco17_tpu-8.yaml")
    #     cfg.MODEL.SSD.SCORE_THRESH_TEST = 0.5
    #     cfg.MODEL.SSD.NMS_THRESH_TEST = 0.4
    # elif args.network =='YoLoV5':
    #     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/yolov5s.yaml"))
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/yolov5s.yaml")
    #     cfg.MODEL.YOLOV5.SCORE_THRESH_TEST = 0.5
    #     cfg.MODEL.YOLOV5.NMS_THRESH_TEST = 0.4

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.DATASETS.TEST = ("CityAI_val",)

    predictor = DefaultPredictor(cfg)

    # Evaluator
    # evaluator = MyEvaluator("CityAI_val", cfg, False, output_dir=output_path)

    # # Evaluate the model
    # val_loader = build_detection_test_loader(cfg, "CityAI_val")

    # # Print inference results on terminal and on .csv file
    # results = inference_on_dataset(predictor.model, val_loader, evaluator)
    # print(results)

    # # Results has it format: OrderedDict([('bbox', {'AP': 36.01825017229739, 'AP50': 47.08669319247479, 'AP75': 44.17784895893762, 'APs': nan, 'APm': 20.90227500636576, 'APl': 89.35432654725776, 'AP-person': nan, 'AP-bicycle': nan, 'AP-car': 36.01825017229739, 'AP-motorcycle': nan, 'AP-bus': nan, 'AP-truck': nan, 'AP-traffic light': nan, 'AP-stop sign': nan, 'AP-parking meter': nan, 'AP-bench': nan, 'AP-bird': nan, 'AP-cat': nan, 'AP-dog': nan, 'AP-horse': nan, 'AP-sheep': nan, 'AP-cow': nan, 'AP-elephant': nan, 'AP-bear': nan, 'AP-zebra': nan, 'AP-giraffe': nan, 'AP-backpack': nan, 'AP-umbrella': nan, 'AP-handbag': nan, 'AP-tie': nan, 'AP-suitcase': nan, 'AP-frisbee': nan, 'AP-skis': nan, 'AP-snowboard': nan, 'AP-sports ball': nan, 'AP-kite': nan, 'AP-baseball bat': nan, 'AP-baseball glove': nan, 'AP-skateboard': nan, 'AP-surfboard': nan, 'AP-tennis racket': nan, 'AP-bottle': nan, 'AP-wine glass': nan, 'AP-cup': nan, 'AP-fork': nan, 'AP-knife': nan, 'AP-spoon': nan, 'AP-bowl': nan, 'AP-banana': nan, 'AP-apple': nan, 'AP-sandwich': nan, 'AP-orange': nan, 'AP-broccoli': nan, 'AP-carrot': nan, 'AP-hot dog': nan, 'AP-pizza': nan, 'AP-donut': nan, 'AP-cake': nan, 'AP-chair': nan, 'AP-couch': nan, 'AP-potted plant': nan, 'AP-bed': nan, 'AP-dining table': nan, 'AP-toilet': nan, 'AP-tv': nan, 'AP-laptop': nan, 'AP-mouse': nan, 'AP-remote': nan, 'AP-keyboard': nan, 'AP-cell phone': nan, 'AP-microwave': nan, 'AP-oven': nan, 'AP-toaster': nan, 'AP-sink': nan, 'AP-refrigerator': nan, 'AP-book': nan, 'AP-clock': nan, 'AP-vase': nan, 'AP-scissors': nan, 'AP-teddy bear': nan, 'AP-hair drier': nan, 'AP-toothbrush': nan})])
    # df = pd.DataFrame(results['bbox'], index=[0])
    # # Delete the object columns
    # df = df.drop(df.filter(regex='AP-').columns, axis=1)
    # df.to_csv(output_path + 'results.csv', index=False)
    # # print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # --------------------------------- INFERENCE --------------------------------- #
    for i, d in enumerate(dataset_dicts):
        num = d["file_name"].split('/')[-1].split('.')[0]
        if num > 1000 and num < 1500:
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)

            instances = outputs["instances"].to("cpu")
            car_instances = instances[instances.pred_classes == 2]

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(car_instances)

            if args.save_vis:
                cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

            print("Processed image: " + d["file_name"].split('/')[-1])

        print("Done!")
