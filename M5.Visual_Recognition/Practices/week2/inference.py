# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# import some common detectron2 utilities
from detectron2 import model_zoo

if __name__ == '__main__':
    output_path = './Results/Task_c/mask_RCNN/'
    data_path = '/export/home/group03/mcv/datasets/KITTI-MOTS/testing/image_02/'

    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    os.makedirs(output_path, exist_ok=True)

    # Loop over all subdirectories in the input path
    for seq_folder, dirs, files in os.walk(data_path):
        # Create the corresponding subdirectory in the output path
        sequenceOut_path = os.path.join(output_path, os.path.relpath(seq_folder, data_path))
        os.makedirs(sequenceOut_path, exist_ok=True)

        print("Processing sequence: " + sequenceOut_path)

        # Loop over all image files in the current subdirectory
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(seq_folder, file)
                im = cv2.imread(image_path)
                outputs = predictor(im)

                outputIm_path = os.path.join(sequenceOut_path, file)

                # We can use `Visualizer` to draw the predictions on the image.
                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(outputIm_path, out.get_image()[:, :, ::-1])

                print("Processing image: " + outputIm_path)
