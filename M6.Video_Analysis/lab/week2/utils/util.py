import os
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict

""" def load_from_xml(path):

    frame_dict = defaultdict(list)
    for event, elem in ET.iterparse(path, events=('start',)):
        if elem.tag == 'track' and elem.attrib.get('label') == 'car':
            for x in (child.attrib for child in elem):
                frame = f"f_{x['frame']}"
                frame_dict[frame].append([float(x['xtl']), float(x['ytl']),
                                          float(x['xbr']), float(x['ybr'])])
    return frame_dict """


def load_from_xml(path):
    """

    :param path: path file

    :return: dict[frame_num] = [[x1, y1, x2, y2]]
    """

    with open(path) as f:
        tracks = xmltodict.parse(f.read())["annotations"]["track"]

    gt = defaultdict(list)
    num_iter = 0
    for track in tracks:
        label = track["@label"]
        boxes = track["box"]
        for box in boxes:
            if label == "car" and box['attribute']['#text'].lower() == 'false':
                frame = int(box["@frame"])
                frame = f"f_{frame}"
                gt[frame].append(
                    [
                        float(box["@xtl"]),
                        float(box["@ytl"]),
                        float(box["@xbr"]),
                        float(box["@ybr"]),
                    ]
                )
                num_iter += 1

            else:
                continue

    return gt


def load_from_txt(path):
    """
    :param path: path file

    :return: list = [[frame,x1, y1, x2, y2, conf]]
    """
    frame_list = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        frame = f"f_{int(ll[0]) - 1}"
        frame_list.append(
            [
                frame,
                float(ll[2]),
                float(ll[3]),
                float(ll[2]) + float(ll[4]),
                float(ll[3]) + float(ll[5]),
                ll[6],
            ]
        )

    return frame_list


def bounding_box_visualization(path, gt_boxes, predicted_boxes, video_capture, frame_id, iou_scores):
    n_frame = int(frame_id.split('_')[-1])
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
    res, frame = video_capture.read()
    # Draw the ground truth boxes
    for box in gt_boxes[frame_id]:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    # Draw the predicted boxes
    for box in predicted_boxes[frame_id]:
        cv2.rectangle(frame, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
    # put text mIOU of frame
    cv2.putText(
        frame,
        f"IoU score: {iou_scores[n_frame]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # put text number of frame
    cv2.putText(frame, f"Frame: {n_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(f'{path}/{frame_id}.png', frame)

    ret, frame = video_capture.read()


def noise_reduction(frame):
    # frame = cv2.erode(frame, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # frame = cv2.dilate(frame,  cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    kernel_m = 3
    kernel_g = (5, 5)
    im_median = cv2.medianBlur(frame, kernel_m)
    frame = cv2.GaussianBlur(im_median, kernel_g, 0)

    return frame


def findBBOX(mask):
    minH = 50
    maxH = 1080 / 2  # 1080--> height frame
    minW = 100  # 120
    maxW = 1920 / 2  # 1920--> width frame

    if len(mask.shape) > 2:
        # Sum over the channels to create a single channel mask
        mask = mask.sum(axis=-1)
        # Binarize to put 255 on the pixels that are not black
        mask = (mask != 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if minW < w < maxW and minH < h < maxH:
            if 0.2 < w / h < 10:
                box.append([x, y, x + w, y + h])

    return box


def visualizeTask1(dict, output_path):
    "dict: keys the value of alpha and for values [mIoU, mAP]"
    # create the output path : output_path + '/GridSearch' + time
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ouput_path_grid = os.path.join(output_path, 'GridSearch', time)

    # create the directory output 
    os.makedirs(ouput_path_grid)

    # plot mIoU scatter plot
    plt.figure()
    plt.scatter(list(dict.keys()), [x[1] for x in dict.values()], label='IoU')
    plt.xlabel('alpha')
    plt.ylabel('IoU')
    plt.title('IoU vs alpha')
    plt.xticks(list(dict.keys()))
    plt.savefig(os.path.join(ouput_path_grid, 'IoU.png'))

    # plot mAP
    plt.figure()
    plt.scatter(list(dict.keys()), [x[0] for x in dict.values()], label='mAP')
    plt.xlabel('alpha')
    plt.ylabel('mAP')
    plt.title('mAP vs alpha')
    plt.xticks(list(dict.keys()))
    plt.savefig(os.path.join(ouput_path_grid, 'mAP.png'))

    # the two metrics together
    plt.figure()
    plt.scatter(list(dict.keys()), [x[0] for x in dict.values()], label='mAP', color='orange')
    plt.scatter(list(dict.keys()), [x[1] for x in dict.values()], label='IoU', color='blue')
    plt.xlabel('alpha')
    plt.ylabel("metric's value")
    plt.title('mAP and IoU vs alpha')
    plt.xticks(list(dict.keys()))  # set x-axis ticks to dictionary keys
    plt.legend()
    plt.savefig(os.path.join(ouput_path_grid, 'mAP_IoU.png'))

    # create a table with pandas and save it as a csv file
    df = pd.DataFrame.from_dict(dict, orient='index', columns=['mAP', 'IoU'])
    df.to_csv(os.path.join(ouput_path_grid, 'task1_2.csv'))


def visualizeTask2(dict, output_path):
    """dict: keys the value of alpha and rho and for values [mIoU, mAP]
     dic[alpha][rho] = [map,iou]"""
    # create the output path : output_path + '/GridSearch' + time
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ouput_path_grid = os.path.join(output_path, 'GridSearch', time)

    # create the directory
    os.makedirs(ouput_path_grid)

    # plot mIoU scatter plot in 3D, for each alpha the rho values
    # extract the alpha, rho, and value data from the results dictionary
    alphas = []
    rhos = []
    valuesMAP = []
    valuesIoU = []
    for key, value in dict.items():
        for key2, value2 in value.items():
            alphas.append(float(key))
            rhos.append(float(key2))
            valuesMAP.append(float(value2[0]))
            valuesIoU.append(float(value2[1]))

    # create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alphas, rhos, valuesMAP)

    # set axis labels and title
    ax.set_xlabel('alpha')
    ax.set_ylabel('rho')
    ax.set_zlabel('mAP')
    ax.set_title('Grid Search Results for the mAP')
    plt.savefig(os.path.join(ouput_path_grid, 'mAP.png'))

    # plot mIoU scatter plot in 3D, for each alpha the rho values
    # create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alphas, rhos, valuesIoU)

    # set axis labels and title
    ax.set_xlabel('alpha')
    ax.set_ylabel('rho')
    ax.set_zlabel('IoU')
    ax.set_title('Grid Search Results for the IoU')
    plt.savefig(os.path.join(ouput_path_grid, 'IoU.png'))

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    X = np.array(alphas).reshape(-1, len(set(rhos)))
    Y = np.array(rhos).reshape(-1, len(set(rhos)))
    Z1 = np.array(valuesMAP).reshape(X.shape)
    Z2 = np.array(valuesIoU).reshape(X.shape)

    ax1.plot_surface(X, Y, Z1, cmap='coolwarm')
    ax2.plot_surface(X, Y, Z2, cmap='coolwarm')

    # set axis labels and titles
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('rho')
    ax1.set_zlabel('mAP')
    ax1.set_title('mAP')

    ax2.set_xlabel('alpha')
    ax2.set_ylabel('rho')
    ax2.set_zlabel('IoU')
    ax2.set_title('IoU')
    # add colorbar
    fig.colorbar(ax1.plot_surface(X, Y, Z1, cmap='coolwarm'), ax=ax1)
    fig.colorbar(ax2.plot_surface(X, Y, Z2, cmap='coolwarm'), ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(ouput_path_grid, 'task2_surfaces.png'))

    # create a table with pandas and save it as a csv file to save the mAP and IoU for each alpha and rho
    # save the dictionary of dictionaries as a csv file
    df = pd.DataFrame()

    # append columns
    df['alpha'] = alphas
    df['rho'] = rhos
    df['mAP'] = valuesMAP
    df['IoU'] = valuesIoU
    df = df[['alpha', 'rho', 'mAP', 'IoU']]

    df.to_csv(os.path.join(ouput_path_grid, 'task2.csv'))


def visualizeTask4(dict, output_path):
    """dict: keys the value of alpha and rho and for values [mIoU, mAP]
     dic[alpha][rho] = [map,iou]"""
    # create the output path : output_path + '/GridSearch' + time
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ouput_path_grid = os.path.join(output_path, 'GridSearch', time)

    # create the directory
    os.makedirs(ouput_path_grid)

    # plot mIoU scatter plot in 3D, for each alpha the rho values
    # extract the alpha, rho, and value data from the results dictionary
    alphas = []
    colors = []
    values = []
    for key, value in dict.items():
        for key2, value2 in value.items():
            alphas.append(key)
            colors.append(key2)
            values.append(value2)

    values = np.array(values).T

    # create a table with pandas and save it as a csv file to save the mAP and IoU for each alpha and rho
    # save the dictionary of dictionaries as a csv file
    df = pd.DataFrame()

    # append columns
    df['alpha'] = alphas
    df['color'] = colors
    df['mAP'] = values[0]
    df['IoU'] = values[1]
    df = df[['alpha', 'color', 'mAP', 'IoU']]

    df.to_csv(os.path.join(ouput_path_grid, 'task4.csv'))

    # # create a 3D scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(alphas, colors, values[0])

    # # set axis labels and title
    # ax.set_xlabel('alpha')
    # ax.set_ylabel('color')
    # ax.set_zlabel('mAP')
    # ax.set_title('Grid Search Results for the mAP')
    # plt.savefig(os.path.join(ouput_path_grid,'mAP.png'))

    # #plot mIoU scatter plot in 3D, for each alpha the rho values
    # # create a 3D scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(alphas, colors, values[1])

    # # set axis labels and title
    # ax.set_xlabel('alpha')
    # ax.set_ylabel('color')
    # ax.set_zlabel('IoU')
    # ax.set_title('Grid Search Results for the IoU')
    # plt.savefig(os.path.join(ouput_path_grid,'IoU.png'))


def filter_boxes(boxes, max_aspect_ratio, nms_threshold):
    # Convert frame_bbox to a NumPy array and extract the boxes and confidence scores
    # boxes = np.array([box for box in boxes if len(box) != 0])

    # Filter boxes based on aspect ratio if the length of boxes is greater than 0
    filtered_boxes = []

    for box in boxes:
        if len(box) == 0:
            continue
        else:
            aspect_ratio = (box[3] - box[1]) / (box[2] - box[0])
            if aspect_ratio > max_aspect_ratio:
                continue
            else:
                filtered_boxes.append(box)

    scores = np.ones(len(filtered_boxes))
    # Apply NMS on the filtered boxes
    nms_indices = cv2.dnn.NMSBoxes(filtered_boxes, scores, 0.0, nms_threshold)
    # print(nms_indices)

    # Set the supressed boxes to be empty
    for i in range(len(filtered_boxes)):
        if i not in nms_indices:
            filtered_boxes[i] = []

    return filtered_boxes


if __name__ == "__main__":
    # Set the parent directory of your current directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    # Set the relative path to the XML file
    relative_path = "dataset/ai_challenge_s03_c010-full_annotation.xml"

    # Get the absolute path of the XML file
    path = os.path.join(parent_dir, relative_path)

    # Print the absolute path
    print(path)
    frame_dict = load_from_xml(path)
    print(frame_dict)
