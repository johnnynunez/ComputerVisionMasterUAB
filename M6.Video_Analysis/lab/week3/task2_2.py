from __future__ import print_function

import cv2
import pandas as pd
from tqdm import tqdm

from sort.sort import *
from utils.util import load_from_txt, discard_overlaps, filter_boxes


def tracking(current_path, folder_det, network, thr, display):
    images = {}

    fileDetections = os.path.join(folder_det)

    colours = np.random.rand(100, 3)  # used only for display
    frame_boxes = load_from_txt(fileDetections, threshold=thr)  # load detections

    total_time = 0.0
    total_frames = 0
    out = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter("./Results/Task_2_2/" + f"task2_2_{network}_thr{int(thr * 100)}.mp4", fourcc, 10,
                                (1920, 1080))

    mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)  # create instance of the SORT tracker
    tracker_colors = {}

    for frame_id in tqdm(frame_boxes):  # all frames in the sequence

        dets = frame_boxes[frame_id]  # each box is [frame,x1, y1, x2, y2, conf]
        dets = discard_overlaps(dets)
        dets = filter_boxes(dets, r=1.25, y=230)

        # from each box we extract only the x1, y1, x2, y2
        dets = [[d[1], d[2], d[3], d[4]] for d in dets]

        total_frames += 1
        fn = current_path + f'/../../dataset/AICity_data/train/S03/c010/frames/{frame_id}.jpg'
        im = io.imread(fn)
        start_time = time.time()
        dets = np.array(dets)

        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        out.append(trackers)
        images[frame_id] = trackers

        for d in trackers:
            d = d.astype(np.uint32)
            tracker_id = d[4]
            if tracker_id not in tracker_colors:
                # generate a new random color for this tracker
                tracker_colors[tracker_id] = np.random.rand(3)
            color = tracker_colors[tracker_id]
            # color array to tuple
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            cv2.rectangle(im, (d[0], d[1]), (d[2], d[3]), color, 2)
            cv2.putText(im, str(tracker_id), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        if display:
            cv2.imshow('frame', im)
        video_out.write(im)

    video_out.release()
    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    # save trackers to data frame, not out
    # save to csv
    df_list = []
    for frame_id in images:
        for track in images[frame_id]:
            width = track[2] - track[0]
            height = track[3] - track[1]
            bb_left = track[0]
            bb_top = track[1]
            df_list.append(pd.DataFrame({'frame': int(frame_id), 'id': track[4], 'bb_left': bb_left, 'bb_top': bb_top,
                                         'bb_width': width, 'bb_height': height, 'conf': 0.5, "x": -1, "y": -1,
                                         "z": -1}, index=[0]))
    # format output for the evaluation
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=['id'])
    df['frame'] = df['frame'] + 1

    df.to_csv(f'./Results/Task_2_2/task_2_2_{network}_thr{int(thr * 100)}.csv', index=False)


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    """network = "faster_RCNN"
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.6, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.65, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.7, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.75, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.8, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.85, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.9, display=False)

    network = "retinaNet"
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.6, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.65, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.7, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.75, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.8, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.85, display=False)
    tracking( current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_A.txt",network,thr = 0.9, display=False)"""

    network = "faster_RCNN"
    tracking(current_path, f"./Results/Task1_5/{network}/A/bbox_{network}_video_A.txt", network, thr=0.75,
             display=False)
