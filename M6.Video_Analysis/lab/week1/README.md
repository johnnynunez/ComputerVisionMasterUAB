# Week 1

### Goal:

- Learn about the databases to be used: [AICityChallenge](https://www.aicitychallenge.org/) and [KITTI](https://www.cvlibs.net/datasets/kitti/)
- Implement evaluation metrics: Intersection over Union (IoU), Average Precision (AP), Mean Square Error in Non-Occluded
  areas, Percentage of Erroneous pixels in Non-occluded areas
- Analyze: Effect of noise additions, IoU vs Time, Optical Flow

&nbsp;

### Task 1 and 2

See Task1_Task2.ipynb

&nbsp;

### Task 3 and 4

See Task3_Task4.ipynb

&nbsp;

### Video Generation

Arguments:\
-m --run_mode : Noisy, Yolo, RCNN or SSD [noisy, rcnn, ssd, yolo]\
-r --run_name : Run Folder Name\
-c --config   : Config.yml path\
-s --save     : Save the video or not    [True, False]\
-d --display  : Show the video or not    [True, False]\
-n --noisy    : Noisy or not             [True, False]

#### Noisy generations

```
python3 main.py -m noisy -n True -r noisy
```

#### Mask R-CNN detections

```
python3 main.py -m rcnn -r rcnn
```

#### SSD512 detections

```
python3 main.py -m ssd -r ssd
```

#### YoloV3 detections

```
python3 main.py -m yolo -r yolo
```


