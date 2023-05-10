# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse

# import some common libraries
import os
import copy
import pandas as pd
import wandb
import torch
from datetime import datetime as dt

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.utils.visualizer import Visualizer
from detectron2.utils import comm
from CityAI_dataset import get_CityAI_dicts
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset

# import some common detectron2 utilities
from detectron2 import model_zoo

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))


# Modify COCOEvaluator to compute only the AP of the bounding boxes, not the masks (we want object detection, not instance segmentation)
class MyEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._tasks = ("bbox",)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)

        evaluator_list = [coco_evaluator]

        return DatasetEvaluators(evaluator_list)


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()  # takes init from HookBase
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(
            build_detection_train_loader(self.cfg)
        )  # builds the dataloader from the provided cfg
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
            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )  # puts these metrics into the storage (where detectron2 logs metrics)

                # save best weights
                if losses_reduced < self.best_loss:  # if current loss is lower
                    self.best_loss = losses_reduced  # saving the best loss
                    self.weights = copy.deepcopy(
                        self.trainer.model.state_dict()
                    )  # saving the best weights


if __name__ == '__main__':

    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task 1_3: Fine-tuning')
    parser.add_argument('--task', type=str, default='Task_1_3', help='Task to perform: task_1_3')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument("--save_vis", type=bool, default=True, help="Save visualizations")
    parser.add_argument("--strategy", type=str, default='A', help="A, B_2, B_3, B_4, C_1, C_2, C_3, C_4")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # --------------------------------- W&B --------------------------------- #
    run = wandb.init(sync_tensorboard=True,
                     settings=wandb.Settings(start_method="thread", console="off"),
                     project="M6_W3")
    wandb.run.name = args.task + '_' + args.network + '_' + args.strategy + '_' + str(args.lr)

    # --------------------------------- OUTPUT --------------------------------- #
    # now = dt.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    # output_path = os.path.join(current_path, f'Results/Task_1_4/{args.network}/{dt_string}')

    output_path = os.path.join(current_path, f'Results/{args.task}/{args.network}/{args.strategy}/{args.lr}')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #

    classes = ['car']
    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"CityAI_{subset}", lambda subset=subset: get_CityAI_dicts(subset, pretrained=False,
                                                                                           strategy=args.strategy))
        MetadataCatalog.get(f"CityAI_{subset}").set(thing_classes=classes)

    metadata = MetadataCatalog.get("CityAI_train")

    # --------------------------------- MODEL --------------------------------- #
    cfg = get_cfg()
    # Write the cfg in a .txt file
    with open(os.path.join(output_path, 'config.txt'), 'w') as f:
        f.write(str(cfg))

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
    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset

    # Solver
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = (1000, 2000, 2500)
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 100

    # Test
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4

    # Dataset
    cfg.DATASETS.TRAIN = ("CityAI_train",)
    cfg.DATASETS.TEST = ("CityAI_val_subset",)
    cfg.OUTPUT_DIR = output_path

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4

    # --------------------------------- TRAINING --------------------------------- #
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])

    start = dt.now()
    trainer.train()
    end = dt.now()
    print('Training time: ', end - start)

    # --------------------------------- EVALUATION --------------------------------- #

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("CityAI_val",)

    predictor = DefaultPredictor(cfg)

    evaluator = MyEvaluator("CityAI_val", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "CityAI_val")

    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)

    df = pd.DataFrame(results['bbox'], index=[0])
    df.to_csv(output_path + '/results.csv', index=False)

    # --------------------------------- INFERENCE --------------------------------- #
    dataset_dicts = get_CityAI_dicts("val", pretrained=False, strategy=args.strategy)

    for i, d in enumerate(dataset_dicts):
        num = d["file_name"].split('/')[-1].split('.')[0]
        if num > 870 & num < 950:
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)

            instances = outputs["instances"].to("cpu")
            car_instances = instances[instances.pred_classes == 0]

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(car_instances)

            if args.save_vis:
                cv2.imwrite(output_path + '/' + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

            print("Processed image: " + d["file_name"].split('/')[-1])
