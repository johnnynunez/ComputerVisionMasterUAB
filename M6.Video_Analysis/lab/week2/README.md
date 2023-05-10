# Week 2

### Goal:

Background estimation

- Model the background pixels of a video sequence using a simple statistical model to classify the background /
  foreground :
    - Single Gaussian per pixel
    - Adaptive / Non-adaptive
- The statistical model will be used to preliminarily classify foreground

Comparison with more complex models

### To execute the tasks, use the main.py

Arguments:\
```-m --run_mode``` : Gaussian, AdaptiveGaussian, SOTA\
```-r --run_name``` : Run Folder Name\
```-c --config```   : config.yml path\
```-s --save ```    : Save the video or not    [True, False]\
```-d --display```  : Show the video or not    [True, False]\
```-p --percentage```: Percentage of video frames to use as background\
```e --sota_method``` : [MOG, MOG2, LSBP, KNN,GMG ]\
```a --alpha```: alpha values \
```-rho```: rho values \
```-c --colorspaces```: [gray, RGB, YCRCB, HSV, YUV]\
```-g --grid```: show the grid or not

#### Gaussian modeling

```
python main.py -r task_1 -m Gaussian -p 0.25 -c gray -a 5 
```

#### Adaptive Gaussian modeling

```
python main.py -r task_2 -m AdaptiveGaussian -p 0.25 -c gray -a 5 --rho 0.05
```

#### SOTA modeling

```
python main.py -r task_3 -m SOTA -p 0.25 -c gray -e MOG2 -a 0
```

#### Color modeling

```
 python main.py -r task_4 -m Gaussian -p 0.25 -c RGB -a 5
```
