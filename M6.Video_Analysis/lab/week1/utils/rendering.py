import itertools
import time

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from utils.metrics import (
    mean_AP_Pascal_VOC,
    mean_IoU_nonrestricted_2,
)


# Rendering Video AICity Challenge 2023


def group_by_frame(predicted_boxes):
    """Group the detected boxes by frame_id as a dictionary"""
    predicted_boxes.sort(key=lambda x: x[0])
    predicted_boxes = itertools.groupby(predicted_boxes, key=lambda x: x[0])
    predicted_boxes = {k: list(v) for k, v in predicted_boxes}
    return predicted_boxes


def rendering_video(path, annotations, predicted_boxes, video_capture, save=True, display=False):
    time_start = time.time()
    wait_time = 1
    """Create a video with the IoU score for each frame"""
    # Group the detected boxes by frame_id as a dictionary
    gt_boxes, total = annotations[0], annotations[1]
    predicted_boxes.sort(key=lambda x: x[-1], reverse=True)
    # sort by key dictionary
    gt_boxes = {k: gt_boxes[k] for k in sorted(gt_boxes)}
    predicted_boxes_group = group_by_frame(predicted_boxes)

    # Get the IoU score for each frame in format dict {frame_id: [iou_score1, iou_score2, ...]}
    AP = mean_AP_Pascal_VOC(gt_boxes, total, predicted_boxes, iou_th=0.5)
    # mIOU, mIOU_frame = mean_IoU_restricted(gt_boxes, predicted_boxes)
    mIOU, mIOU_frame = mean_IoU_nonrestricted_2(gt_boxes, predicted_boxes)
    # Get the frame_id list
    frames_id = list(mIOU_frame.keys())
    # Sort the frames list
    frames_id.sort(key=lambda x: int(x.split('_')[1]))
    frames_num = [int(frame.split('_')[1]) for frame in frames_id]
    # Get the IoU score list
    iou_scores = [np.mean(mIOU_frame[frame]) for frame in frames_id]

    # Open the video
    video_capture = cv2.VideoCapture(video_capture)
    # Get the video fps
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # Get the video width
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Get the video height
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path + "video.mp4", fourcc, fps, (width, height))

    images_plot = []
    fig = plt.figure(figsize=(5, 5))
    # Set the title
    fig.suptitle('IoU score for each frame')
    fig.tight_layout(pad=0)
    ax = plt.axes()
    # Set the x label
    ax.set_xlabel('Frame')
    # Set the y label
    ax.set_ylabel('IoU score')
    # Set the x axis range
    ax.set_xlim(0, frames_num[-1])
    # Set the y axis range
    ax.set_ylim(0, 1)
    # Create a line
    (line,) = ax.plot([], [], lw=2)
    line.set_data([], [])
    if display:
        fig.show()
    fig.canvas.draw()

    # Loop through each frame
    for i, frame_id in enumerate(frames_id):
        # Read the frame
        ret, frame = video_capture.read()
        if ret:
            # Convert the frame to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Draw the ground truth boxes
            for box in gt_boxes[frame_id]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Draw the predicted boxes
            for box in predicted_boxes_group[frame_id]:
                cv2.rectangle(frame, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
            # Draw the IoU score
            iou_score = round(iou_scores[i], 2)
            cv2.putText(
                frame, f"IoU score: {iou_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
            # put text number of frame
            cv2.putText(
                frame, f"Frame: {frames_num[i]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
            # put fps
            cv2.putText(frame, f"FPS: {fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Write the frame to the video
            out.write(frame)
            line.set_data(frames_num[:i], iou_scores[:i])
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height(physical=True)[::-1] + (3,))

            images_plot.append(image)
            if display:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(wait_time)
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    cv2.imwrite(f'save_{frames_num[i]}.png', frame)
                elif k == ord('p'):
                    wait_time = int(not (bool(wait_time)))
    print("mAP: ", AP)
    print("mIOU: ", mIOU)
    time_end = time.time()
    print("Time: ", time_end - time_start)
    if save:
        # Release the VideoWriter object
        out.release()
        # create gif matplotlib figure
        # !convert -delay 10 -loop 0 *.png animation.gif
        imageio.mimsave(path + 'iou.gif', images_plot)
    time_end = time.time()
    print("Time_Finished: ", time_end - time_start)
