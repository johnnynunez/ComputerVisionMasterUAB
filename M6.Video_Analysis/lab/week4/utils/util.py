import itertools

import cv2
import numpy as np
import pandas as pd
import xmltodict
from skimage import io


def load_from_xml(path):
    """

    :param path: path file

    :return: list = [[frame,x1, y1, x2, y2,confidence=-1,track_id]]
    """

    with open(path) as f:
        tracks = xmltodict.parse(f.read())["annotations"]["track"]

    gt = []
    num_iter = 0
    confidence = -1
    for track in tracks:
        track_id = track["@id"]
        label = track["@label"]
        boxes = track["box"]
        for box in boxes:
            if label == "car":
                frame = int(box["@frame"])
                frame = frame
                gt.append(
                    [frame,
                     float(box["@xtl"]),
                     float(box["@ytl"]),
                     float(box["@xbr"]),
                     float(box["@ybr"]),
                     confidence,
                     track_id
                     ]
                )
                num_iter += 1

            else:
                continue

    gt.sort(key=lambda x: x[0])
    gt = itertools.groupby(gt, key=lambda x: x[0])
    gt = {k: list(v) for k, v in gt}

    return gt


def load_from_txt(path, threshold):
    """
    :param path: path file

    :return: list = [[frame,x1, y1, x2, y2, conf]]
    """
    detections = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        if float(ll[6]) >= threshold:
            frame = int(ll[0]) - 1
            detections.append(
                [
                    frame,
                    float(ll[2]),
                    float(ll[3]),
                    float(ll[2]) + float(ll[4]),
                    float(ll[3]) + float(ll[5]),
                    float(ll[6]),
                ]
            )

    """Group the detected boxes by frame_id as a dictionary"""
    detections.sort(key=lambda x: x[0])
    detections = itertools.groupby(detections, key=lambda x: x[0])
    detections = {k: list(v) for k, v in detections}

    return detections


def iou(box1, box2, threshold=0.9):
    if len(box1) > 4:
        box1 = box1[:4]
    """Return iou for a single a pair of boxes"""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)

    if xB < xA or yB < yA:
        interArea = 0
    else:
        interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # respective area of ??the two boxes
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)

    # IOU
    iou_score = interArea / (box1Area + box2Area - interArea)

    return iou_score, iou_score >= threshold


def discard_overlaps(frame_boxes, threshold=0.9):
    discard = []
    for i in range(len(frame_boxes)):
        boxA = [frame_boxes[i][1], frame_boxes[i][2], frame_boxes[i][3], frame_boxes[i][4]]
        for j in range(len(frame_boxes)):
            boxB = [frame_boxes[j][1], frame_boxes[j][2], frame_boxes[j][3], frame_boxes[j][4]]
            if i == j:
                continue
            elif any(j in sublist for sublist in discard):
                continue
            else:
                _, score = iou(boxA, boxB, threshold)
                if score == True:
                    discard.append([i, j])

    discard.sort(key=lambda x: x[1], reverse=True)
    for d in discard:
        del frame_boxes[d[1]]

    return frame_boxes


def filter_boxes(frame_boxes, r=1.25, y=230):
    discard = []
    for i in range(len(frame_boxes)):
        h = frame_boxes[i][4] - frame_boxes[i][2]
        w = frame_boxes[i][3] - frame_boxes[i][1]
        if (frame_boxes[i][2] + h > y) and (h / w > r):
            discard.append(i)

    discard.sort(reverse=True)
    for d in discard:
        del frame_boxes[d]

    return frame_boxes


def video(det_boxes, method, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter("/export/home/group03/mcv-m6-2023-team6/week4/Results/Task1_3/" + f"{method}.mp4",
                                fourcc, fps, (1920, 1080))
    tracker_colors = {}

    for frame_id in det_boxes:
        fn = f'/export/home/group03/dataset/AICity_data/train/S03/c010/frames/{frame_id}.jpg'
        im = io.imread(fn)
        frame_boxes = det_boxes[frame_id]

        for box in frame_boxes:
            track_id = box[-1]
            if track_id not in tracker_colors:
                tracker_colors[track_id] = np.random.rand(3)
            color = tracker_colors[track_id]
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color, 2)
            cv2.putText(im, str(track_id), (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                        cv2.LINE_AA)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        video_out.write(im)
    video_out.release()


def write_csv(detections, out_path):
    df_list = []
    for frame_id in detections:
        for track in detections[frame_id]:
            width = track[3] - track[1]
            height = track[4] - track[2]
            bb_left = track[1]
            bb_top = track[2]
            df_list.append(
                pd.DataFrame({'frame': int(frame_id), 'id': int(track[-1]), 'bb_left': bb_left, 'bb_top': bb_top,
                              'bb_width': width, 'bb_height': height, 'conf': track[-2], "x": -1, "y": -1,
                              "z": -1}, index=[0]))

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=['frame'])
    df['frame'] = df['frame'] + 1

    df.to_csv(out_path, index=False)
