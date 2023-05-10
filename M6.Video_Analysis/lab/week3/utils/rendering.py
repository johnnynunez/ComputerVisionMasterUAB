import itertools
import os
# Import the path to the week3 folder
import sys
import time

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../week3")

from util import load_from_xml_rendering, load_from_txt_rendering
from metrics import (
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


def rendering_video(path, annotations, predicted_boxes, label, video_capture, save=True, display=False):
    time_start = time.time()
    wait_time = 1
    """Create a video with the IoU score for each frame"""
    # Group the detected boxes by frame_id as a dictionary
    gt_boxes, total = annotations[0], annotations[1]

    predicted_boxes_group = []
    for i in range(len(predicted_boxes)):
        predicted_boxes[i].sort(key=lambda x: x[-1], reverse=True)
        predicted_boxes_group.append(group_by_frame(predicted_boxes[i]))

    # sort by key dictionary
    gt_boxes = {k: gt_boxes[k] for k in sorted(gt_boxes)}

    # Get the IoU score for each frame in format dict {frame_id: [iou_score1, iou_score2, ...]}
    mAP_all = []
    mIOU_all = []
    mIOU_frame_all = []
    for i in range(len(predicted_boxes)):
        AP = mean_AP_Pascal_VOC(gt_boxes, total, predicted_boxes[i], iou_th=0.5)
        # mIOU, mIOU_frame = mean_IoU_restricted(gt_boxes, predicted_boxes)
        mIOU, mIOU_frame = mean_IoU_nonrestricted_2(gt_boxes, predicted_boxes[i])

        mAP_all.append(AP)
        mIOU_all.append(mIOU)
        mIOU_frame_all.append(mIOU_frame)

    # write a csv file with the AP and mIOU for each detection method and the name of the detection 'label'
    with open(path + 'results.csv', 'w') as f:
        f.write('name,AP,mIOU\n')
        for i in range(len(predicted_boxes)):
            f.write(f'{label[i]},{mAP_all[i]},{mIOU_all[i]}\n')

    # Create a frame_id list from 535 to 2140
    frames_id = []
    frames_num = []
    for i in range(535, 2141):
        frames_id.append("f_" + str(i))
        frames_num.append(i)

    iou_scores_all = []
    for i in range(len(predicted_boxes)):
        iou_scores = [np.mean(mIOU_frame_all[i][frame]) for frame in frames_id]
        iou_scores_all.append(iou_scores)

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
    fig = plt.figure(figsize=(10, 5))
    # Set the title
    fig.suptitle('IoU score for each frame')
    fig.tight_layout(pad=0)
    ax = plt.axes()
    # Set the x label
    ax.set_xlabel('Frame')
    # Set the y label
    ax.set_ylabel('IoU score')
    # Set the x axis range
    ax.set_xlim(frames_num[0], frames_num[-1])
    # Set the y axis range
    ax.set_ylim(0, 1)
    # Create a line
    line_iou = []
    for i in range(len(predicted_boxes)):
        (line,) = ax.plot([], [], lw=2)
        line_iou.append(line)
        line_iou[i].set_label(label[i])
        line_iou[i].set_data([], [])
    # add legend outside the plot but inside the figure
    ax.legend()

    if display:
        fig.show()
    fig.canvas.draw()

    # Loop through each frame
    for i, frame_id in enumerate(frames_id):
        if i == 0:
            continue
        # Read the frame from the video starting at frame 536
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frames_num[i])
        ret, frame = video_capture.read()
        if ret:
            # Convert the frame to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Draw the ground truth boxes
            for box in gt_boxes[frame_id]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Draw the predicted boxes

            # for box in predicted_boxes_group[frame_id]:

            for box in predicted_boxes_group[-1][f'f_{frames_num[i - 1]}']:
                cv2.rectangle(frame, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
            # Draw the IoU score
            iou_score = round(iou_scores_all[-1][i], 2)
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
            for j in range(len(predicted_boxes)):
                line_iou[j].set_data(frames_num[:i], iou_scores_all[j][:i])

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

    # Plot the IoU score for each frame and each detection
    fig = plt.figure(figsize=(10, 5))
    # Set the title
    fig.suptitle('IoU score for each frame')
    fig.tight_layout(pad=0)
    ax = plt.axes()
    # Set the x label
    ax.set_xlabel('Frame')
    # Set the y label
    ax.set_ylabel('IoU score')
    # Set the x axis range
    ax.set_xlim(frames_num[0], frames_num[-1])
    # Set the y axis range
    ax.set_ylim(0, 1)
    # Create a line
    line_iou = []
    for i in range(len(predicted_boxes)):
        (line,) = ax.plot([], [], lw=2)
        line_iou.append(line)
        line_iou[i].set_label(label[i])
        line_iou[i].set_data([], [])
    # add legend outside the figure
    ax.legend()
    for i in range(len(predicted_boxes)):
        line_iou[i].set_data(frames_num, iou_scores_all[i])
    fig.savefig(path + 'iou.png')

    # print("mAP: ", AP)
    # print("mIOU: ", mIOU)
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


if __name__ == "__main__":

    output_path = '/ghome/group03/mcv-m6-2023-team6/week3/Results/Video_IoU/'
    annotations_xml = '/ghome/group03/dataset/ai_challenge_s03_c010-full_annotation.xml'
    video_path = '/ghome/group03/dataset/AICity_data/train/S03/c010/vdo.avi'
    detection_path = ['/ghome/group03/dataset/AICity_data/train/S03/c010/det/det_ssd512.txt',
                      '/ghome/group03/dataset/AICity_data/train/S03/c010/det/det_yolo3.txt',
                      '/ghome/group03/dataset/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
                      '/ghome/group03/mcv-m6-2023-team6/week3/Results/Task_1_1/faster_RCNN/A/bbox_faster_RCNN_A.txt',
                      '/ghome/group03/mcv-m6-2023-team6/week3/Results/Task_1_1/mask_RCNN/A/bbox_mask_RCNN_A.txt',
                      '/ghome/group03/mcv-m6-2023-team6/week3/Results/Task_1_1/retinaNet/A/bbox_retinaNet_A.txt']

    label = ['SSD512_given', 'YOLO3_given', 'MaskRCNN_given', 'FasterRCNN_X101', 'MaskRCNN_X101', 'RetinaNet_X101']

    # If output_path does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gt_boxes, total = load_from_xml_rendering(annotations_xml)
    gt = [gt_boxes, total]

    predicted_boxes = []
    for i in range(len(detection_path)):
        if i > 2:
            # for each detection file, delete the spaces ' ' and save it in a new file on output_path
            with open(detection_path[i], 'r') as f:
                lines = f.readlines()
                lines = [line.replace(' ', '') for line in lines]
                with open(output_path + 'det_' + label[i] + '.txt', 'w') as f:
                    f.writelines(lines)
            predicted_boxes.append(load_from_txt_rendering(output_path + 'det_' + label[i] + '.txt'))
        else:
            predicted_boxes.append(load_from_txt_rendering(detection_path[i]))

    rendering_video(
        output_path,
        gt,
        predicted_boxes,
        label,
        video_path,
        save=True,
        display=False,
    )

    for i in range(len(detection_path)):
        if i > 2:
            output_path_i = output_path + label[i] + '/'
            if not os.path.exists(output_path_i):
                os.makedirs(output_path_i)
            rendering_video(
                output_path_i,
                gt,
                [predicted_boxes[i]],
                [label[i]],
                video_path,
                save=True,
                display=False,
            )
