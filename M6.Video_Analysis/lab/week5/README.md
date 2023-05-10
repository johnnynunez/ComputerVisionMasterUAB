
# Welcome to the Week 5 report!

In this report, we explore the Multi-Object Single Camera and Multi-Object Multi-Camera problems. For the Multi-Camera problem, we use an adapted version of this repository [vehicle_mtmc](https://github.com/regob/vehicle_mtmc)*.

**IMPORTANT**  
**This repository has been changed to support our tasks*.

Before starting with the project, please ensure the following steps are completed:

```bash
export PYTHONPATH=$("./vehicle_mtmc")
```

## Run all experiments in one command:
Run the following bash scrypt:
```
For bash:  
```bash
bash job
```


## Run the experiments using Maximum Iou Overlap tracking algorithm step by step:
#### Detections
Inference with [Yolov8](https://github.com/ultralytics/ultralytics) to get the detections
```
python inference_yolov8.py
```

#### Tracking 
Tracking using the Max Iou Overlap (MIO) alogrithm developed in [Week3](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week3): 
```bash
python pre_MTMC.py --OF 0
```
Tracking using the Max Iou Overlap with Optical Flow (MIO-OF) developed in [Week4](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week4):
```bash
python pre_MTMC.py --OF 1
```

#### Feature extraction
Extract Features from detected bounding boxes using Resnet50 for MIO:
```bash
python pre_MTMC.py --OF 0
```
and for MIO-OF
```bash
python pre_MTMC.py --OF 1
```

#### Re-Id model (MTMC)
Run the Re-Id model based on computing the similarity matrix between all tracks, check the camera compatibility and perform a priority queue using [heapq](https://docs.python.org/3/library/heapq.html). For MIO:
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s01_max_iou.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s03_max_iou.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s04_max_iou.yaml
```
For MIO-OF
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s01_max_iou_OF.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s03_max_iou_OF.yaml
```
```bash
python vehicle_mtmc/mtmc/run_mtmc.py --config AI_city/mtmc_s04_max_iou_OF.yaml
```



#### Run the experiments using DeepSort and ByteTrack tracking algorithms:

Run the following commands in the same way as [vehicle_mtmc](https://github.com/regob/vehicle_mtmc). For DeepSort:
```bash
python vehicle_mtmc/mtmc/run_express_mtmc.py --config AI_city/end2end_DeepSort_s01.yaml
```
```bash
python vehicle_mtmc/mtmc/run_express_mtmc.py --config AI_city/end2end_DeepSort_s03.yaml
```
```bash
python vehicle_mtmc/mtmc/run_express_mtmc.py --config AI_city/end2end_DeepSort_s04.yaml
```

For ByteTrack
```bash
python vehicle_mtmc/mtmc/run_express_mtmc.py --config AI_city/end2end_ByTrack_s01.yaml 
```
```bash
python vehicle_mtmc/mtmc/run_express_mtmc.py --config AI_city/end2end_ByTrack_s03.yaml 
```
```bash
python vehicle_mtmc/mtmc/run_express_mtmc.py --config AI_city/end2end_ByTrack_s04.yaml 
```


## Run Evaluation with TrackEval for individual cameras
Note that the ground truth must be in the appropriate format and the directories as well, defined in the TrackEval library. You can find the documentation for the
MOTChallenge and it's format [here](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official). 

To run multiple TrackEval's with the resulting .txt of each camera, we have created one script to manage the directories (trackeval.py). Run the following bash script:
```bash
sh job_trackEval
```

Once the directories are correctly created, run the evaluation with:
```bash
python run_mot_challenge.py --DO_PREPROC False 
```

------------
### Notes: 
+ You can find our yaml's [here](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week5/vehicle_mtmc/config/AI_city).
+ You can find HyperParameters explanations [here](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week5/vehicle_mtmc/config/defaults.py).







