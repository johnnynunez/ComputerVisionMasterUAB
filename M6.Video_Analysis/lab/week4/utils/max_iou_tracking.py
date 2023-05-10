import copy
import os
import numpy as np 
from tqdm import tqdm
from PIL import Image


# INTERSECTION OVER UNION
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


def track_memory(tracked_objects,threshold=10):
    delete = []
    for idx in tracked_objects:
        if tracked_objects[idx]['memory'] == threshold:
                delete.append(idx)

    for idx in delete:
        del tracked_objects[idx]


def max_iou_tracking_withParked(det_boxes, iou_threshold=0.5):

    """ 
    MTSC: computes the maximum overlap tracking algorithm for Multi-Target Single-camera

    camera: camera being evaluated
    det_boxes: detected boxes = load_from_txt(.txt)
    iou_threshold: minimum overlap between tracked bounding boxes

    Returns: det_boxes with tracking ID.
    """
    track_id = 0
    tracked_objects = {}
    memory_check = 5

    # sequence = [seq for seq,cam in seqs.items() if camera in cam][0]
    tracked_iou = {}

    for frame_id in tqdm(det_boxes):
        # REMOVE OVERLAPPING BOUNDING BOXES 
        boxes = det_boxes[frame_id]
        frame_boxes = discard_overlaps(boxes)

        # FIRST FRAME, WE INITIALIZE THE OBJECTS ID
        if not tracked_objects:
            for j in range(len(frame_boxes)):
                # We add the tracking object ID at the end of the list  [[frame,x1, y1, x2, y2, conf, track_id]]
                frame_boxes[j].append(track_id)
                tracked_objects[f'{track_id}'] = {
                    'bbox': [frame_boxes[j][1], frame_boxes[j][2], frame_boxes[j][3], frame_boxes[j][4]],
                    'frame': frame_id, 'memory': 0, 'iou': 0}
                track_id += 1

        else:
            # FRAME N+1 WE COMPARE TO OBJECTS IN FRAME N
            for i in range(len(frame_boxes)):
                frame_boxes[i][0] = frame_id
                best_iou = 0
                track_id_best = 0
                boxA = [frame_boxes[i][1], frame_boxes[i][2], frame_boxes[i][3], frame_boxes[i][4]]

                for data in previous_tracked_objects.items():
                    id, boxB = data
                    iou_score, _ = iou(boxA, boxB['bbox'])

                    if iou_score > best_iou and iou_score >= iou_threshold:
                        best_iou = iou_score
                        track_id_best = id

                if track_id_best == 0 and best_iou == 0:
                    frame_boxes[i].append(track_id)
                    tracked_objects[f'{track_id}'] = {'bbox': boxA, 'frame': frame_id, 'memory': 0, 'iou': best_iou}
                    track_id += 1


                else:
                    if tracked_objects[f'{track_id_best}']['frame'] == frame_id:
                        # CHECK IF THERE IS AN OBJECT WITH THE SAME ID IN THAT FRAME AND CHOOSE THE ONE WITH HIGHEST IOU
                        if best_iou > tracked_objects[f'{track_id_best}']['iou']:
                            tracked_objects[f'{track_id}'] = {'bbox': tracked_objects[f'{track_id_best}']['bbox'],
                                                              'frame': frame_id, 'memory': 0, 'iou': best_iou}
                            wrong_id = [i for i, det in enumerate(frame_boxes) if det[-1] == track_id_best][0]
                            frame_boxes[wrong_id][-1] = track_id
                            track_id += 1

                            frame_boxes[i].append(track_id_best)
                            tracked_objects[f'{track_id_best}']['bbox'] = boxA
                            previous_f = tracked_objects[f'{track_id_best}']['frame']

                            # CHECK IF OBJECTS APPEAR CONSECUTIVE
                            if frame_id - previous_f == 1:
                                tracked_objects[f'{track_id_best}']['memory'] = tracked_objects[f'{track_id_best}'][
                                                                                    'memory'] + 1
                            tracked_objects[f'{track_id_best}']['frame'] = frame_id
                            tracked_objects[f'{track_id_best}']['iou'] = best_iou

                        else:
                            frame_boxes[i].append(track_id)
                            tracked_objects[f'{track_id}'] = {'bbox': boxA, 'frame': frame_id, 'memory': 0,
                                                              'iou': best_iou}
                            track_id += 1


                    else:
                        frame_boxes[i].append(track_id_best)
                        tracked_objects[f'{track_id_best}']['bbox'] = boxA
                        previous_f = tracked_objects[f'{track_id_best}']['frame']

                        # CHECK IF OBJECTS APPEAR CONSECUTIVE
                        if frame_id - previous_f == 1:
                            tracked_objects[f'{track_id_best}']['memory'] = tracked_objects[f'{track_id_best}'][
                                                                                'memory'] + 1
                        tracked_objects[f'{track_id_best}']['frame'] = frame_id
                        tracked_objects[f'{track_id_best}']['iou'] = best_iou

        if frame_id == memory_check:
            track_memory(tracked_objects)
            memory_check = memory_check + frame_id

        previous_tracked_objects = copy.deepcopy(tracked_objects)
        
    # remove key from det_boxes if the list is empty
    for key in list(det_boxes.keys()):
        if len(det_boxes[key]) == 0:
            del det_boxes[key]

    return det_boxes
