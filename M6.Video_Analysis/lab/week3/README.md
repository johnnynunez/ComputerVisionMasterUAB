# Week 3

## Goal: Object Detection and tracking

Welcome to the Object Detection and Tracking using Overlap and Kalman Filter repository! This project aims to implement
a computer vision technique for detecting and tracking objects in a video. Specifically, we utilize some networks to
detect the objects in each frame and then apply overlap or the Kalman filter to track their movement over time.

Object detection and tracking is a crucial task in many applications, including surveillance, robotics, and autonomous
vehicles. By accurately detecting and tracking objects, we can gather important information about their position,
velocity, and trajectory, which can be used to make informed decisions and predictions.

In this repository, we provide the necessary code and resources to implement the overlap-based method and Kalman filter
for object detection and tracking. We also include sample videos and datasets to test and evaluate the performance of
our algorithm.

We hope that this repository will serve as a valuable resource for anyone interested in learning about object detection
and tracking using computer vision techniques. Please feel free to explore and provide feedback on our work!

## Task 1: Object detection

To execute the code, specify the desired network architecture using the "args.network" argument. This argument can take
on one of the following values:

- "faster_RCNN"
- "mask_RCNN"
- "retinaNet"

For save visualitzations ```--save_vis True```

### Task 1.1

```
python task_1_1.py --network <network_to_use>
```

### Task 1.2

For annotations you can access
here: [Annotations](https://github.com/mcv-m6-video/mcv-m6-2023-team6/tree/main/week3/Results/Task_1_2_CVAT)

### Task 1.3 & Task 1.4

```
python main.py --task Task1_3 --network <network_to_use> --save_vis <True/False> --strategy <strategy_to_use> --lr <learning_rate>
```

```
python main.py --task Task1_3_inference --network <network_to_use> --model_path <path_to_model> --save_vis <True/False> --strategy <strategy_to_use>
```

where:

- network: The network to use for object detection. It can be either 'faster_RCNN', 'mask_RCNN', or 'retinaNet'.
- model_path: The path to the model to use for object detection.
- save_vis: Whether to save visualizations or not. It can be either True or False.
- strategy: The strategy to use for object tracking. It can be one of A, B_2, B_3, B_4, C_1, C_2, C_3, C_4.

#### Some results of Task 1.4

|    Split     | Faster R-CNN | RetinaNet |
|:------------:|:------------:|:---------:|
| Strategy A 1 |     0.85     |   0.92    |
| Strategy B 2 |     0.84     |   0.84    |
| Strategy B 3 |     0.86     |   0.88    |
| Strategy B 4 |     0.87     |   0.88    |
| Strategy C 1 |     0.87     |   0.97    |
| Strategy C 2 |     0.88     |   0.88    |
| Strategy C 3 |     0.88     |   0.96    |
| Strategy C 4 |     0.88     |   0.88    |
|     Mean     |     0.87     |   0.90    |

## Task 2: Object tracking

### Task 2.1: Tracking by overlap

```
python task2_1.py -m <network to use>
```

where

- m: "faster_RCNN or retinaNet"

### Task 2.2: Tracking with Kalman filter

```
python task2_2.py
```

### Task 2.3: EvalTrack

First of all, if you want this repository with submodules, you can clone it with the following command:

``` 
git clone --recurse-submodules -j8 git://github.com/mcv-m6-video/mcv-m6-2023-team6.git
```

To eval: Put your results in MOT 1.1 Format:

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
````

And then run MOT17 benchmark by default:

```
python run_motchallenge.py --DO_PREPROC False --METRICS HOTA Identity TrackMAP
```



