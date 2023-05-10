import argparse
import copy
import os
import pickle
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

# MASKFLOWNET
# Clone Repo
# https://github.com/microsoft/MaskFlownet
# Set path to MaskFlownet in utils/maskflow.py
# from utils.maskflow import maskflownet
# from datasets import AICityDataset
from utils.util import load_from_txt, discard_overlaps, filter_boxes, iou

from utils.max_iou_tracking import max_iou_tracking_withParked



def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_all_pkl_files(folder_path):
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    all_data = {}
    for file_name in pkl_files:
        file_path = os.path.join(folder_path, file_name)
        data = load_pkl_file(file_path)
        all_data[file_name] = data

    return all_data


def convert_pkl_to_txt(pkl_folder, txt_folder):
    all_data = read_all_pkl_files(pkl_folder)

    for file_name, data in all_data.items():
        fname = file_name.split('.')[0]
        f = open(pkl_folder+'/'+fname+'.txt','w')
        print(f"Writing content of {fname}: in txt format for TrackEval")
        for frame,dets in data.items():
            for det in dets:
                w = det[3] - det[1]
                h = det[4] - det[2]
                f.write(f'{frame},{det[-1]},{det[1]},{det[2]},{w},{h},{det[-2]},-1,-1,-1 \n')

        f.close()


def track_memory(tracked_objects):
    delete = []
    for idx in tracked_objects:
        if tracked_objects[idx]['memory'] != tracked_objects[idx]['frame']:
            if tracked_objects[idx]['memory'] <= 5:
                delete.append(idx)

    for idx in delete:
        del tracked_objects[idx]


def max_iou_tracking(path, frames_path, fps, conf_threshold=0.6, iou_threshold=0.5):
    total_time = 0.0
    total_frames = 0

    det_boxes = load_from_txt(path, threshold=conf_threshold)
    delta_t = 1 / fps

    track_id = 0
    tracked_objects = {}
    memory = 5

    for frame_id in tqdm(det_boxes):
        total_frames += 1
        start_time = time.time()
        # REMOVE OVERLAPPING BOUNDING BOXES
        boxes = det_boxes[frame_id]
        boxes = discard_overlaps(boxes)
        frame_boxes = filter_boxes(boxes)
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
            current_frame = np.array(Image.open(os.path.join(frames_path, f'{frame_id}.jpg')))
            previous_frame = np.array(Image.open(os.path.join(frames_path, f'{frame_id - 1}.jpg')))

            flow, _ = maskflownet(previous_frame, current_frame, colType=1)

            for data in previous_tracked_objects.items():
                id, boxB = data
                boxB = np.array(boxB['bbox'])

                # Optical flow estimation for each object
                flow_boxB = flow[int(boxB[1]):int(boxB[3]) + 1, int(boxB[0]):int(boxB[2]) + 1]
                flow_boxB = np.mean(flow_boxB, axis=(0, 1))

                displacement = delta_t * flow_boxB

                # UPDATE step: we add to the previous object position the motion estimated (from optical flow estimation)
                new_bbox_B = [boxB[0] + displacement[0],
                              boxB[1] + displacement[1],
                              boxB[2] + displacement[0],
                              boxB[3] + displacement[1]]

                previous_tracked_objects[id]['new_bbox'] = new_bbox_B

            for i in range(len(frame_boxes)):
                frame_boxes[i][0] = frame_id
                best_iou = 0
                track_id_best = 0
                boxA = [frame_boxes[i][1], frame_boxes[i][2], frame_boxes[i][3], frame_boxes[i][4]]

                for data in previous_tracked_objects.items():
                    id, boxB = data
                    iou_score, _ = iou(boxA, boxB['new_bbox'])

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

        if frame_id == memory:
            track_memory(tracked_objects)
            memory = memory + frame_id

        previous_tracked_objects = copy.deepcopy(tracked_objects)
        cycle_time = time.time() - start_time
        total_time += cycle_time

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    return det_boxes


def task2(args):
    # dataset = AICityDataset(args.dataset_path, args.sequences)
    for c in os.listdir(args.dataset_path + '/train/' + args.sequences):
        detections = args.dataset_path + '/train/' + args.sequences + '/' + c + '/det/det_yolo3.txt'
        frames_path = args.dataset_path + '/train/' + args.sequences + '/' + c + '/frames/'
        cap = cv2.VideoCapture(args.dataset_path + '/train/' + args.sequences + '/' + c + '/vdo.avi')
        fps = cap.get(cv2.CAP_PROP_FPS)

        tracking_boxes = max_iou_tracking(detections, frames_path, fps=fps)

        with open(f'{args.results_path}tracking_maskflownet_{c}.pkl', 'wb') as h:
            pickle.dump(tracking_boxes, h, protocol=pickle.HIGHEST_PROTOCOL)
            
def task2_withoutOF(args):
    
    # dataset = AICityDataset(args.dataset_path, args.sequences)
    for c in os.listdir(args.dataset_path + '/train/' + args.sequences):
        detections = args.dataset_path + '/train/' + args.sequences + '/' + c + '/det/det_yolo3.txt'
        frames_path = args.dataset_path + '/train/' + args.sequences + '/' + c + '/frames/'
        cap = cv2.VideoCapture(args.dataset_path + '/train/' + args.sequences + '/' + c + '/vdo.avi')
        fps = cap.get(cv2.CAP_PROP_FPS)
        

        det_boxes = load_from_txt(detections, threshold=0.6)

        tracking_boxes = max_iou_tracking_withParked(det_boxes)
        
        # If args.results_path does not exist, create it
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)

        with open(f'{args.results_path}tracking_maskflownet_{c}.pkl', 'wb') as h:
            pickle.dump(tracking_boxes, h, protocol=pickle.HIGHEST_PROTOCOL)
            
        convert_pkl_to_txt(args.results_path, args.results_path)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    # print("Assigned GPU IDs:", gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="/ghome/group03/dataset/aic19-track1-mtmc-train",
                        help='dataset')

    parser.add_argument('--sequences', type=str, default="S03", help='sequences')

    parser.add_argument('--results_path', type=str, default='/ghome/group03/mcv-m6-2023-team6/week4/Results/Task2_withoutOF/',
                        help='path to save results in a csv file')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()

    task2_withoutOF(args)

