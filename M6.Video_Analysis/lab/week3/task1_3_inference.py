# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse

# import some common libraries
import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
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
    parser = argparse.ArgumentParser(description='Task 1_3: Fine-tuning')
    parser.add_argument('--task', type=str, default='Task_1_4', help='Task to perform: task_1_3')
    parser.add_argument('--network', type=str, default='retinaNet', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument("--save_vis", type=bool, default=True, help="Save visualizations")
    parser.add_argument("--strategy", type=str, default='C_3', help="A, B_2, B_3, B_4, C_1, C_2, C_3, C_4")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # --------------------------------- OUTPUT --------------------------------- #

    output_path = os.path.join(current_path,
                               f'Results/{args.task}/{args.network}/{args.strategy}/{args.lr}/image_examples')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #

    # classes = ['car', 'bycicle']
    # for subset in ["train", "val", "val_subset"]:
    #     DatasetCatalog.register(f"CityAI_{subset}",lambda subset=subset: get_CityAI_dicts(subset, pretrained=False, strategy=args.strategy))
    #     MetadataCatalog.get(f"CityAI_{subset}").set(thing_classes=classes)

    # metadata = MetadataCatalog.get("CityAI_train")

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
    else:
        print('Network not found')
        exit()

    # --------------------------------- CONFIG --------------------------------- #

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4

    cfg.OUTPUT_DIR = output_path

    # --------------------------------- EVALUATION --------------------------------- #

    cfg.MODEL.WEIGHTS = '/ghome/group03/mcv-m6-2023-team6/week3/Results/Task_1_4/retinaNet/C_3/0.01/model_final.pth'
    cfg.DATASETS.TEST = ("CityAI_val",)

    predictor = DefaultPredictor(cfg)

    # --------------------------------- INFERENCE --------------------------------- #
    dataset_dicts = get_CityAI_dicts("val", pretrained=False, strategy=args.strategy)

    for i, d in enumerate(dataset_dicts):
        num = int(d["file_name"].split('/')[-1].split('.')[0])
        if num < 850 or num > 1000:
            continue

        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        instances = outputs["instances"].to("cpu")
        car_instances = instances[instances.pred_classes == 0]

        # # filter the bboxes that height < width
        # height = car_instances.pred_boxes.tensor[:,3] - car_instances.pred_boxes.tensor[:,1]
        # width = car_instances.pred_boxes.tensor[:,2] - car_instances.pred_boxes.tensor[:,0]

        # ratio = height / width

        # car_instances = car_instances[ratio < 1.25]

        # Plot the predictions using the visualizer
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        out = v.draw_instance_predictions(car_instances)

        if args.save_vis:
            cv2.imwrite(output_path + '/' + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        # Plot the GT using the visualizer that are in dataset_dicts['annotations']
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        out = v.draw_dataset_dict(d)

        if args.save_vis:
            im_out = out.get_image()[:, :, ::-1]
            # # pass the image to cv2 format
            # im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
            # cv2.line(im_out, (0, 230), (im_out.shape[1], 230), (0, 0, 255), 2)
            cv2.imwrite(output_path + '/gt' + d["file_name"].split('/')[-1], im_out)

        print("Processed image: " + d["file_name"].split('/')[-1])
