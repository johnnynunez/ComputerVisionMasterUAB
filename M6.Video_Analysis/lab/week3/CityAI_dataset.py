import os
import random

import cv2
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer
import xmltodict


def parse_xml_bb(path_xml, includeParked=True):
    """
    Input:
        - Path to xml in Pascal VOC format annotations
    Output format:
        dict[frame_id] = [{'bbox':[x1, y1, x2, y2], 'conf': 1, 'id': -1}]
    """

    with open(path_xml, 'rb') as f:
        xml_dict = xmltodict.parse(f)

    frame_dict = {}

    for track in xml_dict['annotations']['track']:

        if track['@label'] != 'car':
            continue

        track_id = int(track['@id'])

        for bbox in track['box']:

            frame = f"f_{bbox['@frame']}"

            if bbox['attribute']['#text'] == 'true' and includeParked == False:
                if frame not in frame_dict:
                    frame_dict[frame] = []
                continue

            if frame not in frame_dict:
                frame_dict[frame] = []

            frame_dict[frame].append({
                'conf': 1,
                'bbox': [float(bbox['@xtl']), float(bbox['@ytl']), float(bbox['@xbr']), float(bbox['@ybr'])],
                'id': track_id
            })

    return frame_dict


def create_splits(total_frames, strategy):
    """
    Input:
        - X: list of frames
        - k: number of splits
    Output:
        - list of k splits
    """
    # Set seed for reproducibility
    x_25 = int(total_frames * 0.25)

    if strategy == 'A':
        # Fold 1: 0-25% of the frames is train and 25-100% is val
        start = 0
        end = x_25
        train = list(range(start, end))
        val = list(range(end, total_frames))
    elif strategy == 'B_2':
        # K-Fold cross validation use K=3. Split the dataset into 3 folds
        # Fold 2: 25-50% of the frames is train and 0-25% and 50-100% is val
        start = x_25
        end = 2 * x_25
        train = list(range(start, end))
        val = list(range(0, start)) + list(range(end, total_frames))
    elif strategy == 'B_3':
        # K-Fold cross validation use K=3. Split the dataset into 3 folds
        # Fold 3: 50-75% of the frames is train and 0-50% and 75-100% is val
        start = 2 * x_25
        end = 3 * x_25
        train = list(range(start, end))
        val = list(range(0, start)) + list(range(end, total_frames))
    elif strategy == 'B_4':
        # K-Fold cross validation use K=3. Split the dataset into 3 folds
        # Fold 4: 75-100% of the frames is train and 0-75% is val
        start = 3 * x_25
        end = 4 * x_25
        train = list(range(start, end))
        val = list(range(0, start)) + list(range(end, total_frames))
    elif strategy == 'C_1':
        random.seed(1)
        # K-Fold cross validation use K=4. Split the dataset into 4 folds
        # Random split 25% train and 75% val
        list_total = list(range(total_frames))
        random.shuffle(list_total)
        train = list_total[:int(total_frames * 0.25)]
        val = list_total[int(total_frames * 0.25):]
    elif strategy == 'C_2':
        random.seed(2)
        # K-Fold cross validation use K=4. Split the dataset into 4 folds
        # Random split 25% train and 75% val
        list_total = list(range(total_frames))
        random.shuffle(list_total)
        train = list_total[:int(total_frames * 0.25)]
        val = list_total[int(total_frames * 0.25):]
    elif strategy == 'C_3':
        random.seed(3)
        # K-Fold cross validation use K=4. Split the dataset into 4 folds
        # Random split 25% train and 75% val
        list_total = list(range(total_frames))
        random.shuffle(list_total)
        train = list_total[:int(total_frames * 0.25)]
        val = list_total[int(total_frames * 0.25):]
    elif strategy == 'C_4':
        random.seed(4)
        # K-Fold cross validation use K=4. Split the dataset into 4 folds
        # Random split 25% train and 75% val
        list_total = list(range(total_frames))
        random.shuffle(list_total)
        train = list_total[:int(total_frames * 0.25)]
        val = list_total[int(total_frames * 0.25):]
    elif strategy == 'D':
        random.seed(6)
        list_total = list(range(total_frames))
        random.shuffle(list_total)
        train = list_total[:int(total_frames * 0.9)]
        val = list_total[int(total_frames * 0.9):]

    # Create a subset : 10% of the val set
    random.seed(5)
    val_subset = random.sample(val, int(len(val) * 0.1))

    return train, val, val_subset


def get_CityAI_dicts(subset, pretrained=True, strategy="A"):
    images = "/ghome/group03/dataset/AICity_data/train/S03/c010/frames"
    annotations = "/ghome/group03/dataset/ai_challenge_s03_c010-full_annotation.xml"
    gt_bb = parse_xml_bb(annotations)

    total_frames = len(os.listdir(images))

    train, val, val_subset = create_splits(total_frames, strategy)
    if subset == "train":
        list_frames = train
    elif subset == "val":
        list_frames = val
    elif subset == "val_subset":
        list_frames = val_subset
    else:
        raise ValueError("Subset must be train, val or val_subset")

    if pretrained:
        class_id = 2
    else:
        class_id = 0

    dataset_dicts = []

    for seq_id in list_frames:

        record = {}

        filename = os.path.join(images, str(seq_id) + ".jpg")

        # plot the image cv2
        im = cv2.imread(filename)  # Llegeix be les imatges

        record["file_name"] = filename
        record["image_id"] = seq_id
        record["height"] = 1080
        record["width"] = 1920

        objs = []
        gt = gt_bb[f'f_{seq_id}']

        for obj_0 in gt:
            bb = obj_0['bbox']

            # Draw the bounding box in the image
            cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

            obj = {
                "bbox": list(map(int, bb)),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_id,  # 2 is car category , and all bb detected from get_bb are cars
                "segmentation": [],
            }
            objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts


# def get_CityAI_dicts_annot(subset, pretrained=True, strategy="D"):
#     images = "/ghome/group03/dataset/AICity_data/train/S03/c010/frames"
#     annotations = "/ghome/group03/dataset/ai_challenge_s03_c010-full_annotation.xml"
#     gt_bb = parse_xml_bb(annotations)

#     total_frames = len(os.listdir(images))

#     train, val, val_subset = create_splits(total_frames, strategy)
#     if subset == "train":
#         list_frames = train
#     elif subset == "val":
#         list_frames = val
#     elif subset == "val_subset":
#         list_frames = val
#     else:
#         raise ValueError("Subset must be train, val or val_subset")


#     if pretrained:
#         class_id = 2
#     else:
#         class_id = 0

#     dataset_dicts = []

#     for seq_id in list_frames:

#         record = {}

#         filename = os.path.join(images, str(seq_id) + ".jpg"	)

#         # plot the image cv2
#         im = cv2.imread(filename)  # Llegeix be les imatges

#         record["file_name"] = filename
#         record["image_id"] = seq_id
#         record["height"] = 1080
#         record["width"] = 1920

#         objs = []
#         gt = gt_bb[f'f_{seq_id}']

#         for obj_0 in gt:
#             bb = obj_0['bbox']

#             # Draw the bounding box in the image
#             cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

#             obj = {
#                 "bbox": list(map(int, bb)),
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "category_id": class_id,  # 2 is car category , and all bb detected from get_bb are cars
#                 "segmentation": [],
#             }
#             objs.append(obj)

#         record["annotations"] = objs

#         dataset_dicts.append(record)

#     return dataset_dicts


# def get_CityAI_dicts_annot_test():
#     images = "/ghome/group03/dataset/AICity_data/AICity_data_S05_C010/validation/S05/c010/frames"

#     total_frames = len(os.listdir(images))

#     list_frames = list(range(0, 4072))

#     dataset_dicts = []

#     for seq_id in list_frames:

#         record = {}

#         filename = os.path.join(images, str(seq_id) + ".jpg"	)

#         # plot the image cv2
#         im = cv2.imread(filename)  # Llegeix be les imatges
#         height, width, channels = im.shape

#         record["file_name"] = filename
#         record["image_id"] = seq_id
#         record["height"] = height
#         record["width"] = width


#         dataset_dicts.append(record)

#     return dataset_dicts


if __name__ == "__main__":
    # frames = 10
    # strategies = ["A","B_2", "B_3", "B_4", "C_1", "C_2", "C_3", "C_4"]
    # for strat in strategies:
    #     train, val, val_subset = create_splits(frames, strategy=strat)
    #     print(f"Strategy {strat}")
    #     print(f"Train: {train}")
    #     print(f"Val: {val}")
    #     print(f"Val subset: {val_subset}")
    #     print("")

    a = get_CityAI_dicts('val')
