# Week 5
First, we have to generate the dictionaries that will contain the embeddings of the captions. To do so we have to run the following commands:
```
python utils/text/fastText.py
```
```
python utils/text/bert.py
```


The *main.py* file contains the pipeline for all tasks. You can choose the task to run in the parameter:

## Image-to-Text Retrieval (task a)

#### Run a sweep in wandb after adapting the sweep_config dictionary:

```
python main.py --task_a --train True --sweep True --network_image RESNET50 --network_text FastText --batch_size 64
```

#### Run train:
Using Resnet50 as a feature extractor of the anchor image:

```
python main.py --task task_a --train True --sweep False --network_image RESNET50 --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```
Using the backbone of Faster R-CNN as a feature extractor of the anchor image:
```
python main.py --task task_a --train True --sweep False --network_image fasterRCNN --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```

#### Run test:

```
python main.py --task task_a --train False --sweep False --weights_model path/to/model.pth
```


## Text-to-Image Retrieval (task b)

#### Run training examples:
Using Resnet50 as a feature extractor of the anchor image:

```
python main.py --task task_b --train True --sweep False --network_image RESNET50 --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```
Using the backbone of Faster R-CNN as a feature extractor of the anchor image:
```
python main.py --task task_b --train True --sweep False --network_image fasterRCNN --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```
#### Run test:

```
python main.py --task task_b --train False --sweep False --weights_model path/to/model.pth
```

## BERT as a text feature extractor (task c)
Same as the other tasks but using --network_text BERT
