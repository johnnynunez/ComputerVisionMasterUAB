# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse

# import some common libraries
import os
from datetime import datetime

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from formatDataset import get_kitti_dicts, register_kitti_dataset

# import some common detectron2 utilities
from detectron2 import model_zoo

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # argparser

    parser = argparse.ArgumentParser(description='Task E: Finetuning')
    parser.add_argument('--name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # Get the exact time to name the experiment
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    if args.network == 'faster_RCNN':
        output_path = os.path.join(current_path, f'Results/Task_e/{dt_string}_{args.name}/faster_RCNN')
        model = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    elif args.network == 'mask_RCNN':
        output_path = os.path.join(current_path, f'Results/Task_e/{dt_string}_{args.name}/mask_RCNN')
        model = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    else:
        print('Network not found')
        exit()

    os.makedirs(output_path, exist_ok=True)

    kitty_metadata = register_kitti_dataset()

    train_dicts = get_kitti_dicts("train")
    val_dicts = get_kitti_dicts("val")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATASETS.TEST = ()
    # cfg.DATASETS.VAL = ("kitti_mots_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.OUTPUT_DIR = output_path
    cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough, but you can certainly train longer

    cfg.SOLVER.AMP.ENABLED = True

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2000  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    cfg.TEST.EVAL_PERIOD = 20

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Compute the time
    start = datetime.now()
    trainer.train()
    end = datetime.now()
    print('Time to train: ', end - start)
