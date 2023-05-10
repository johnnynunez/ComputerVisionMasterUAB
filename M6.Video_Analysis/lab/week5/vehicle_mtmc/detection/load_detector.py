from pathlib import Path
import torch
import os
from ultralytics import YOLO

def load_yolo(which):
    """Load a yolo network from local repository. Download the weights there if needed."""
    if which == 'yolov8':
        model = YOLO('yolov8n.pt')
    else:
        cwd = Path.cwd()
        yolo_dir = str(Path(__file__).parent.joinpath("yolov5"))
        os.chdir(yolo_dir)
        model = torch.hub.load(yolo_dir, which, source="local")
        os.chdir(str(cwd))
    return model

