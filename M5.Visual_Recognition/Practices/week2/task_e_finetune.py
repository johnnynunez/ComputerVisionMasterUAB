import copy
import random

import cv2
import torch
from detectron2.data import build_detection_train_loader
from detectron2.engine import HookBase
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer

if torch.cuda.is_available():
    print('CUDA is available!')
else:
    print('CUDA is NOT available')

from detectron2.utils.logger import setup_logger

setup_logger()

import argparse
import os

# include the utils folder in the path
import sys
from datetime import datetime as dt

import wandb
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from formatDataset import get_kitti_dicts

from detectron2 import model_zoo

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


# from utils.MyTrainer import *
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()  # takes init from HookBase
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))  # builds the dataloader from the provided cfg
        self.best_loss = float("inf")  # Current best loss, initially infinite
        self.weights = None  # Current best weights, initially none
        self.i = 0  # Something to use for counting the steps

    def after_step(self):  # after each step
        if self.trainer.iter >= 0:
            print(
                f"----- Iteration num. {self.trainer.iter} -----"
            )  # print the current iteration if it's divisible by 100

        data = next(self._loader)  # load the next piece of data from the dataloader

        with torch.no_grad():  # disables gradient calculation; we don't need it here because we're not training, just calculating the val loss
            loss_dict = self.trainer.model(data)  # more about it in the next section

            losses = sum(loss_dict.values())  #
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )  # puts these metrics into the storage (where detectron2 logs metrics)

                # save best weights
                if losses_reduced < self.best_loss:  # if current loss is lower
                    self.best_loss = losses_reduced  # saving the best loss
                    self.weights = copy.deepcopy(self.trainer.model.state_dict())  # saving the best weights


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)

        evaluator_list = [coco_evaluator]

        return DatasetEvaluators(evaluator_list)


#  https://towardsdatascience.com/train-maskrcnn-on-custom-dataset-with-detectron2-in-4-steps-5887a6aa135d

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task E: Finetuning')
    parser.add_argument('--name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    # --------------------------------- W&B --------------------------------- #
    run = wandb.init(
        sync_tensorboard=True, settings=wandb.Settings(start_method="thread", console="off"), project="M5_W2"
    )
    wandb.run.name = args.name

    # --------------------------------- OUTPUT --------------------------------- #
    now = dt.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    output_path = os.path.join(current_path, f'Results/Task_e/{dt_string}_{args.name}/{args.network}')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #
    #  Register the dataset
    classes = ['car', 'pedestrian']
    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts(subset, pretrained=False))
        print(f"Successfully registered 'kitti_{subset}'!")
        MetadataCatalog.get(f"kitti_{subset}").set(thing_classes=classes)

    kitty_metadata = MetadataCatalog.get("kitti_train")

    # --------------------------------- MODEL ----------------------------------- #
    if args.network == 'faster_RCNN':
        model = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    elif args.network == 'mask_RCNN':
        model = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    else:
        print('Network not found')
        exit()

    #  Create the config
    cfg = get_cfg()

    print(cfg)

    # get the config from the model zoo
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo

    # Model
    # cfg.MODEL_MASK_ON = True  # If we want to use the mask. Aquí no sé si per faster hauriem de posar false.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    #  cfg.MODEL.BACKBONE.NAME = 'build_resnet_fpn_backbone'
    #  cfg.MODEL.BACKBONE.FREEZE_AT = 2
    #  cfg.MODEL.RESNETS.DEPTH = 50

    # Solver
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (1000, 2000, 2500)
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    # cfg.SOLVER.AMP.ENABLED = True

    # Test
    cfg.TEST.EVAL_PERIOD = 100

    # Dataset
    cfg.DATASETS.TRAIN = ("kitti_train",)
    #  cfg.DATASETS.VAL = ("kitti_val",)
    cfg.DATASETS.TEST = ("kitti_val_subset",)  # Si es comenta això peta.
    cfg.OUTPUT_DIR = output_path

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4

    print(cfg)

    # --------------------------------- TRAINING --------------------------------- #
    trainer = MyTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])

    # Compute the time
    start = dt.now()
    trainer.train()
    end = dt.now()
    print('Time to train: ', end - start)

    # # --------------------------------- EVALUATION --------------------------------- #
    # cfg.DATASETS.TEST = ("kitti_val",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "kitti_val")

    print("-----------------Evaluation-----------------")
    print(model)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    print("--------------------------------------------")

    # --------------------------------- INFERENCE --------------------------------- #
    dataset_dicts = get_kitti_dicts('val', pretrained=True)
    for d in random.sample(dataset_dicts, 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1])

    wandb.finish()
