#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate m6
python3 main.py -m noisy -n True -r noisy
python3 main.py -m rcnn -r rcnn
python3 main.py -m ssd -r ssd
python3 main.py -m yolo -r yolo