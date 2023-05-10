# Week 4

### Task 1: Optical Flow & Multi-Target Single-Camera Tracking


#### Task 1.1: Block Matching
```bash
python task_1_1.py
```

+ Grid search to optimize hyperparameters with optuna: block size = [2, 4, 8, 16, 32, 64, 128] and search
  area =[2, 4, 8, 16, 32, 64, 128], step size = [1, 2, 4, 8, 16, 32, 64, 128], distance type = ['NCC', 'SAD', 'SSD']

```bash
python task_1_1_optuna.py
```

+ To extract visual results
```bash
python task_1_1_bbdd.ipynb
```


|         Best parameters                   | MSEN | PEPN  | 
|:-----------------------------------------:|:----:|:-----:|
|block size 16, search area 32, step 8, NCC | 2.71 | 13.67%|   


#### Task 1.2: Off-the-shelf

Clone the following repos to use each algorithm:

+ [Pyflow](https://github.com/pathak22/pyflow)
+ [Lucas Kanade](https://docs.opencv.org/3.3.1/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)
+ [MaskFlownet](https://github.com/microsoft/MaskFlownet)
+ [RAFT](https://github.com/princeton-vl/RAFT)
+ [LiteFlowNet](https://github.com/sniklaus/pytorch-liteflownet)
+ [DEQ-Flow](https://github.com/locuslab/deq-flow)

Run the following command:

```bash
python task_1_2.py
```
To check also the results from Perceiver, clone the this repo and follow the provided instructions to create perceiver-io environment
+ [Perceiver-IO](https://github.com/krasserm/perceiver-io.git)

With this environment, run the following command:

```bash
python task_1_2_perceiver.py
```

Here are the overall results:

|    Method    | MSEN | PEPN  | Runtime |
|:------------:|:----:|:-----:|:-------:|
| MaskFlowNet  | 0.28 | 0.76  |  1.54   |
|   DEQ-Flow   | 0.52 | 2.31  |  2.45   |
| LiteFlowNet  | 0.57 | 3.26  |  0.54   |
|     RAFT     | 0.59 | 1.43  |  1.67   |
| Perceiver-IO | 0.74 | 4.07  |   3.5   |
|    PyFlow    | 0.97 | 7.99  |  7.39   |
| Lucas-Kanade | 8.36 | 35.74 |  0.12   |

#### Task 1.3: Improve Tracking with Optical Flow

This task evaluates the combination of the Otical Flow and Tracking algorithm from Task 2.1
of [week 3](https://github.com/mcv-m6-video/mcv-m6-2023-team6/blob/main/week3/task2_1.py)

```bash
python task_1_3.py
```

|    Method    | HOTA % | IDF1 % |
|:------------:|:------:|:------:|
|     RAFT     | 84.27  | 87.94  |
| MaskFlowNet  | 84.26  | 87.91  |   
| LiteFlowNet  | 84.25  | 87.81  |   
| Lucas-Kanade | 84.22  | 87.78  |   

### Task 2: Multi Target Single Camera Tracking

For AI City Challenge

Using MaskFlowNet:

```bash
python task_2.py --optical_flow_method maskflownet
```

### Task 3: Multi Target Multi Camera Tracking (Optional)
```bash
python task_3.py --optical_flow_method maskflownet
```
```bash
python task_3_1.py --optical_flow_method maskflownet
```

Future work(Post Processing methods to increase performance)


