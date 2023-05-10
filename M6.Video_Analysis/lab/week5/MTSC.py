import argparse
import os
import cv2
import pickle as pkl
import pprint

from utils import max_iou_tracking, max_iou_tracking_OF, util


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="annotate a video from multicam tracks")
    parser.add_argument("--OF", type=int)
    parser.add_argument("--output_path", type=str, default = '/ghome/group03/mcv-m6-2023-team6/week5/Results/trackings/MTSC')
    parser.add_argument("--detections_path", type=str, default = '/export/home/group03/mcv-m6-2023-team6/week5/Results/detections')
    parser.add_argument("--dataset_path", type=str, default = '/export/home/group03/dataset/aic19-track1-mtmc-train/train')
    args = parser.parse_args()
    
    use_OF = bool(args.OF)
    print(use_OF)
    if use_OF:
        print("Using: Optical Flow")
        path = f'{args.output_path}/max_iou_OF'
    else:
        print("Not Using: Optical Flow")
        path = f'{args.output_path}/max_iou'
    
    
    dataset_path = args.dataset_path
    detections_path = args.detections_path

    if use_OF:
        results_path = f'{args.output_path}/max_iou_OF/input'
    else:
        results_path = f'{args.output_path}/max_iou/input'
    

    #sequences = ["S01","S04","S03"]
    sequences = ["S03"]

    for s in sequences:
        sequence_path = os.path.join(dataset_path,s)
        
        output_path = os.path.join(results_path,s)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for c in os.listdir(sequence_path):
            det_path = os.path.join(detections_path,c+'.txt')
            frames_path = os.path.join(sequence_path,c,'frames')
            cap = cv2.VideoCapture(os.path.join(sequence_path,c,'vdo.avi'))
            fps = cap.get(cv2.CAP_PROP_FPS)
            det_boxes = util.load_from_txt(det_path)

            if use_OF:
                tracking_boxes = max_iou_tracking_OF.max_iou_tracking_withoutParked(det_boxes,frames_path,fps)
            else:
                tracking_boxes = max_iou_tracking.max_iou_tracking_withoutParked(det_boxes,frames_path,fps)
                

            with open(f'{output_path}/{c}.pkl','wb') as h:
                pkl.dump(tracking_boxes,h,protocol=pkl.HIGHEST_PROTOCOL)

        util.convert_pkl_to_txt(output_path,output_path)