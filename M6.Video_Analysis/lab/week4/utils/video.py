import os
import pickle

import cv2
import numpy as np
from skimage import io
from tqdm import tqdm


def video(det_boxes, method, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (1920, 1080))
    tracker_colors = {}

    # current path file
    current_path = os.path.dirname(os.path.abspath(__file__))
    for frame_id in tqdm(det_boxes):
        fn = f'../../dataset/aic19-track1-mtmc-train/train/S03/c015/frames/{frame_id}.jpg'
        im = io.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
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

        video_out.write(im)
    video_out.release()


method = 'maskflownet'
fps = 8
detections = pickle.load(open(f'./Results/Task2/tracking_maskflownet_c015.pkl', 'rb'))
video(detections, method, fps)