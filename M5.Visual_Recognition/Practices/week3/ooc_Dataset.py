import os

import cv2
from tqdm import tqdm


def get_ooc_dicts(subset, pretrained=False):
    images = "/ghome/group03/mcv/datasets/out_of_context/"

    if subset == "train":
        sequences_id = os.listdir(images)

    elif subset == "val":
        sequences_id = os.listdir(images)

    elif subset == "val_subset":
        sequences_id = os.listdir(images)

    dataset_dicts = []
    idx = 1

    for seq_id in tqdm(sequences_id):
        record = {}

        filename = os.path.join(images, str(seq_id))

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        dataset_dicts.append(record)

        idx += 1

    return dataset_dicts


def get_coco_dicts(subset, pretrained=False):
    images = "/ghome/group03/test2017"

    if subset == "train":
        sequences_id = os.listdir(images)

    elif subset == "val":
        sequences_id = os.listdir(images)

    elif subset == "val_subset":
        sequences_id = os.listdir(images)

    dataset_dicts = []
    idx = 1

    for seq_id in tqdm(sequences_id):
        record = {}

        filename = os.path.join(images, str(seq_id))

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        dataset_dicts.append(record)

        idx += 1

    return dataset_dicts
