import copy
import random

import numpy as np
from tqdm import tqdm


# Intersection over Union (IoU)
def iou(box1, box2):
    if len(box1) > 4:
        box1 = box1[:4]
    """Return iou for a single a pair of boxes"""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)

    # respective area of ​​the two boxes
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    # overlap area
    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # IOU
    return interArea / (boxAArea + boxBArea - interArea)


# Generate noisy boxes for testing
def generate_noisy_boxes(gt_boxes, del_prob, gen_prob, mean, std, frame_shape=[1080, 1920]):
    """
    :gt_boxes: ground truth bounding boxes dict
    :del_prob: probability to delete bounding boxes
    :gen_prob: probability to generate bounding boxes
    :return: list with the noisy bounding boxes list = [[frame,x1, y1, x2, y2]]
    """
    noisy_bboxes = []
    gt_total = 0
    for frame, bboxes in gt_boxes.items():
        for bbox in bboxes:
            gt_total += 1
            if np.random.random() > del_prob:
                xtl, ytl, xbr, ybr = bbox
                noise = np.random.normal(mean, std, 4)
                noisy_bboxes.append(
                    [
                        frame,
                        xtl + noise[0],
                        ytl + noise[1],
                        xbr + noise[2],
                        ybr + noise[3],
                    ]
                )
                w = xbr - xtl
                h = ybr - ytl

        if np.random.random() <= gen_prob:
            x = np.random.randint(w, frame_shape[1] - w)
            y = np.random.randint(h, frame_shape[0] - h)
            noisy_bboxes.append([frame, x - w / 2, y - w / 2, x + w / 2, y + w / 2])

    return noisy_bboxes


def mean_IoU_restricted(gt_boxes, predicted_boxes):
    """
    :gt_boxes: ground truth bounding boxes dict
    :predicted_boxes: predicted bounding boxes
    :return: mean IOU
    """
    mIOU = 0
    count = 0
    mIOU_frame = {}
    used_pred_boxes = set()

    for gt in gt_boxes:
        for box in gt_boxes[gt]:
            iou_score = []
            for pred in predicted_boxes:
                if pred[0] == gt and tuple(pred[1:5]) not in used_pred_boxes:
                    iou_score.append(iou(box, pred[1:5]))
                else:
                    iou_score.append(0)
            if iou_score:
                id = np.argmax(iou_score)
                max_iou = iou_score[id]
                used_pred_boxes.add(tuple(predicted_boxes[id][1:5]))
                mIOU += max_iou
                count = count + 1

                # Save max iou for each frame
                if gt not in mIOU_frame:
                    mIOU_frame[gt] = []
                mIOU_frame[gt].append(max_iou)

    return mIOU / count, mIOU_frame


def mean_IoU_nonrestricted(gt_boxes, predicted_boxes):
    """
    :gt_boxes: ground truth bounding boxes dict
    :predicted_boxes: predicted bounding boxes
    :return: mean IOU
    """
    mIOU = 0
    count = 0
    mIOU_frame = {}

    for gt in gt_boxes:
        for box in gt_boxes[gt]:
            iou_score = []
            for pred in predicted_boxes:
                if pred[0] == gt:
                    iou_score.append(iou(box, pred[1:5]))
                else:
                    iou_score.append(0)
            if iou_score:
                id = np.argmax(iou_score)
                max_iou = iou_score[id]
                mIOU += max_iou
                count = count + 1

                # Save max iou for each frame
                if gt not in mIOU_frame:
                    mIOU_frame[gt] = []
                mIOU_frame[gt].append(max_iou)

    return mIOU / count, mIOU_frame


def mean_IoU_nonrestricted_2(gt_boxes, predicted_boxes):
    """
    :gt_boxes: ground truth bounding boxes dict
    :predicted_boxes: predicted bounding boxes
    :return: mean IOU
    """
    mIOU = 0
    count = 0
    mIOU_frame = {}
    # convert predicted boxes in dictionary
    predicted_boxes_dict = {}
    for pred in predicted_boxes:
        if pred[0] not in predicted_boxes_dict:
            predicted_boxes_dict[pred[0]] = []
        predicted_boxes_dict[pred[0]].append(pred[1:5])
    for gt in gt_boxes:
        for box in gt_boxes[gt]:
            iou_score = []
            predicted_box_frame = predicted_boxes_dict.get(gt, [])
            if predicted_box_frame:
                for pred in predicted_box_frame:
                    iou_score.append(iou(box, pred))
            else:
                iou_score.append(0)
            if iou_score:
                id = np.argmax(iou_score)
                max_iou = iou_score[id]
                mIOU += max_iou
                count = count + 1

                # Save max iou for each frame
                if gt not in mIOU_frame:
                    mIOU_frame[gt] = []
                mIOU_frame[gt].append(max_iou)

    return mIOU / count, mIOU_frame


# Average Precision (AP) for Object Detection
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/pascal_voc_evaluation.py
def mean_AP_Pascal_VOC(gt_boxes, N_gt, predicted_boxes, iou_th):
    """
    :gt_boxes: ground truth bounding boxes dict
    :N_gt: Total of ground truth bounding boxes
    :predicted_boxes: predicted bounding boxes
    :return: mean IOU, average precision
    """
    mIOU = 0
    tp = np.zeros(len(predicted_boxes))
    fp = np.zeros(len(predicted_boxes))
    gt_detected = copy.deepcopy(gt_boxes)

    mIOU_frame = {}
    for i in range(len(predicted_boxes)):
        frame = predicted_boxes[i][0]
        predicted = predicted_boxes[i][1:5]
        gt = gt_detected[frame]
        iou_score = []
        if len(gt) != 0:
            for b in range(len(gt)):
                iou_score.append(iou(gt[b], predicted))
            id = np.argmax(iou_score)
            max_iou = iou_score[id]
            mIOU += max_iou
            # Save max iou for each frame
            if frame not in mIOU_frame:
                mIOU_frame[frame] = []
            mIOU_frame[frame].append(max_iou)

            if max_iou > iou_th:
                if len(gt_detected[frame][id]) == 4:
                    gt_detected[frame][id].append(True)
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / np.float64(N_gt)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap  # Ull la mIoU s'ha d'agafar de la funció mean_IoU_restricted o mean_IoU_nonrestricted
    # return mIOU / len(predicted_boxes), mIOU_frame, ap


def compute_confidences_ap(gt_boxes, N_gt, predicted_boxes, N=10, iou_th=0.5):
    """
    Randomly generates the order of the bounding boxes to calculate the average precision (N times).
    Average values will be returned.
    """
    ap_scores = []

    for i in tqdm(range(N)):
        random.shuffle(predicted_boxes)
        ap = mean_AP_Pascal_VOC(gt_boxes, N_gt, predicted_boxes, iou_th)
        mIOU, _ = mean_IoU_nonrestricted_2(gt_boxes, predicted_boxes)

        ap_scores.append(ap)

    return sum(ap_scores) / len(ap_scores), mIOU
