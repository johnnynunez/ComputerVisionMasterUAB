from ultralytics import YOLO
import argparse
import os
from utils.util import *
import cv2
import numpy as np
import pickle

SEQS = ['S01','S04','S03']

def extract_results(frame_results,n_frame,f,confidence=0.6):
    detections_frame = frame_results[0].boxes.cpu().numpy()
    for det in detections_frame:
        if det.cls[-1] == 2 or det.cls[-1] == 7: # CAR & TRUCK Classes
            if float(det.conf[-1]) > confidence:
                w = det.data[0][2] - det.data[0][0]
                h = det.data[0][3] - det.data[0][1]
                f.write(f'{n_frame},-1,{round(det.data[0][0],3)},{round(det.data[0][1],3)},{round(w,3)},{round(h,3)},{round(det.conf[-1],3)},-1,-1,-1 \n')


def main(args):
    for seq in SEQS:
        for c in os.listdir(args.d+'/'+seq):
            f = open(args.r+'/'+c+'.txt','w')
            cd =args.d+'/'+seq+'/'+c
            frames = os.listdir(cd+'/frames')
            model = YOLO('yolov8n.pt')
            
            roi = cv2.imread(cd+'/roi.jpg')
            
            for frame in range(len(frames)):
                n_frame = frame + 1
                img = cv2.imread(cd+'/frames/'+str(n_frame)+'.jpg')
                
                img[roi == 0] = 0
                
                result = model(img)

                extract_results(result,n_frame,f)

            f.close()

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detections: Inference using yolov8')
    parser.add_argument('-d', type=str, default='/export/home/group03/dataset/aic19-track1-mtmc-train/train/', help='Dataset directory')
    parser.add_argument('-r', type=str, default='/export/home/group03/mcv-m6-2023-team6/week5/Results/detections/', help='Path to save results')

    args = parser.parse_args()

    main(args)