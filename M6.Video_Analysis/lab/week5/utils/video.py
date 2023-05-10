import os
from util import load_from_txt_video
import argparse
import cv2
import numpy as np
from skimage import io
from collections import defaultdict
from tqdm import tqdm


colors =  {1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255, 255, 0),5: (128, 0, 128),6:(0, 255, 255),7:(255, 0, 255),8:(255, 165, 0),9:(255, 20, 147),10:(165, 42, 42),11:(0, 128, 128),12:(75, 0, 130),
            13:(238, 130, 238),14:(128, 128, 0),15:(128, 0, 0),16:(255, 215, 0),17:(192, 192, 192),18:(0, 0, 128),19:(0, 255, 255),20:(255, 127, 80),21:(0, 255, 0),22:(255, 0, 255),23:(64, 224, 208),24:(245, 245, 220),25: (221, 160, 221)} 

colors = np.random.uniform(0, 255, size=(25, 3))

def video(args,fps=10.0):
    if args.cam == 'c015':
        fps = 8.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = f"{args.output}/{args.seq}_{args.cam}_2.mp4"
    frames = os.listdir(f'{args.dataset_path}/{args.seq}/{args.cam}/frames')
    frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
    # print(f"Frame path: {frames_path}")
    im = io.imread(f'{args.dataset_path}/{args.seq}/{args.cam}/frames/{frames[0]}')
    width = im.shape[0]
    height = im.shape[1]
    video_out = cv2.VideoWriter(path, fourcc, fps, (height, width))
    if not video_out.isOpened():
        print("Error: Video writer not initialized.")
    
    #get the frames in order
    frames = os.listdir(f'{args.dataset_path}/{args.seq}/{args.cam}/frames')
    frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
    
    # MODIFIED BY JOHNNY

    tracks = load_from_txt_video(args.tracking)
    
    # current path file
    for frame in tqdm(frames):
        frames_path = f'{args.dataset_path}/{args.seq}/{args.cam}/frames/{frame}'
        im = io.imread(frames_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        id = int(frame.split('.')[0])

        c = 1
        if id in tracks.keys():
            frame_boxes = tracks[id]
            for box in frame_boxes:
                track_id = box[0]

                if track_id not in colors:
                    color = colors[c]
                    c += 1

                    if c == 24:
                        c = 1
                else:
                    color = colors[track_id]

                cv2.rectangle(im, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color, 2)
                cv2.putText(im, str(track_id), (int(box[1]), int(box[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                            cv2.LINE_AA)

        video_out.write(im)
    video_out.release()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video generation')
    parser.add_argument('--dataset_path', type=str, default='/export/home/group03/dataset/aic19-track1-mtmc-train/train/', help='Dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Folder to save results')
    parser.add_argument('--seq', type=str, default='S03', help='Sequence to use')
    parser.add_argument('--cam', type=str, required=True, help='Camera within the sequence')
    parser.add_argument('--tracking', type=str, required=True, help='Text files with tracking results')

    args = parser.parse_args()

    if not os.path.exists(f'{args.output}'):
        os.makedirs(f'{args.output}') 

    video(args)


