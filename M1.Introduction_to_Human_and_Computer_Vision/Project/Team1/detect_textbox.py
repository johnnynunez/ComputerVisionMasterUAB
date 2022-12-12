import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imsave
import glob
from test_submission import check_text_box_file
import pickle


def find_borders(line):
    threshold = 0.8*np.max(line)
    border = np.argwhere(np.diff(np.sign(line - threshold))).flatten()
    return border


def find_text_bounding_box(image, kernel_shape=(10, 30)):
    image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # dilated = cv2.dilate(image_bw, kernel, iterations=1)

    kernel = np.ones(kernel_shape, dtype=np.uint8)
    gradient = cv2.morphologyEx(image_bw, cv2.MORPH_GRADIENT, kernel)
    
    # There are two scenarios:  Case 1: White box & Black Text
    #                           Case 2: Black box & White Text
    # Thus, we need to calculate both: opening and closing.
    opening = cv2.morphologyEx(image_bw, cv2.MORPH_OPEN, kernel)
    opening_grad = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)
    closing_grad = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    
    # We use min to select the smooth regions of both gradients
    # (the 'big' area of textbox, given the kernel, should be quite smooth).
    smooth_regions = np.min([opening_grad, closing_grad], axis=0)
    smooth_regions = np.isclose(smooth_regions, 0)

    # Create a mask of gradient using the smooth regions and filter for the maximum value
    # (the max value should correspond to the textbox gradient, because the area and brightness are bigger).
    masked_grad = gradient[smooth_regions]
    threshold = np.max(masked_grad)*0.8
    # Remove parts of gradients below the threshold, and refine it with an opening. 
    gradient[gradient < threshold] = 0
    gradient = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)

    # Calculate vertical boundaries of remaining parts (textbox) and get top & bottom indices.
    gradient[0,:], gradient[-1,:], gradient[:,0], gradient[:,-1] = 0,0,0,0
    row_grad = np.sum(gradient, axis=1)
    borders = find_borders(row_grad)
    top, bottom = borders[0], borders[1]

    # Remove parts of image outside the calculated vertical textbox boundaries
    gradient[:max((top-gradient.shape[0]), 0)] = 0
    gradient[(bottom+gradient.shape[0]):] = 0

    # Calculate horizontal boundaries of textbox and get left & right indices.
    col_grad = np.sum(gradient[top:bottom], axis=0)
    borders = find_borders(col_grad)
    left, right = borders[0], borders[-1]

    bbox_pred = [left-10, top-20, right+10, bottom+20]
    # Snap negative bbox detected points to origin of image (0), if needed.
    bbox_pred = [0 if c < 0 else c for c in bbox_pred]
    
    return bbox_pred



def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou


def create_textbox_mask(mask_shape, bbox):
    mask = np.zeros(mask_shape, dtype=bool)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    return np.invert(mask)



def evaluate_text_bbox(images_folder, output_folder='./', se_shape = (10, 30)):
    # se_shape: Structuring element (kernel).
    bboxes_gt = np.load(os.path.join(images_folder, 'text_boxes.pkl'), allow_pickle=True)

    results = []
    for img_idx, img_file in enumerate(tqdm(sorted(map(os.path.basename, (glob.glob(f'{images_folder}/*.jpg')))))):
            # Find text bounding box.
            img = imread(os.path.join(images_folder, img_file))
            bbox_pred = find_text_bounding_box(img, kernel_shape=se_shape)

            # Save predicted masks in folder.
            mask = create_textbox_mask(img.shape[0:2], bbox_pred)
            imsave(
                os.path.join(output_folder, img_file.replace('.jpg', '.png')),
                (255*mask).astype(np.uint8),
                check_contrast=False)

            # Save image with bbox annotation, for presenting them.
            start_point, end_point = bbox_pred[0:2], bbox_pred[2:4]
            img = cv2.rectangle(img, start_point, end_point, color=(255, 0, 0), thickness=2)
            imsave(
                os.path.join(output_folder, img_file),
                img,
                check_contrast=False)

            # Load bbox GT for given image.
            bbox_gt_points = bboxes_gt[img_idx][0]
            bbox_gt = np.concatenate([bbox_gt_points[0], bbox_gt_points[2]])
            iou = bbox_iou(bbox_pred, bbox_gt)

            results.append([img_idx, img_file, bbox_pred, bbox_gt, iou])

    # Save CSV with IOU metrics for images.
    df = pd.DataFrame(results, columns=['img_idx','img_file','bbox_pred', 'bbox_gt', 'iou'])
    df.to_csv(os.path.join(output_folder, f'iou_results_{os.path.basename(images_folder)}.csv'))
    
    # Save pickle file with the list of bboxes of the text positions.
    text_boxes = df['bbox_pred'].to_list()
    pickle.dump(text_boxes, open(os.path.join(output_folder, 'text_boxes.pkl'), "wb"))
    
    
def generate_masks_test(images_folder, output_folder='./', se_shape = (10, 30)):
    results = []
    for img_idx, img_file in enumerate(tqdm(sorted(map(os.path.basename, (glob.glob(f'{images_folder}/*.jpg')))))):
            # Find text bounding box.
            img = imread(os.path.join(images_folder, img_file))
            bbox_pred = find_text_bounding_box(img, kernel_shape=se_shape)
            # Snap negative bbox detected points to origin of image (0), if needed.
            bbox_pred = [0 if c < 0 else c for c in bbox_pred]

            # Save predicted masks in folder.
            mask = create_textbox_mask(img.shape[0:2], bbox_pred)
            imsave(
                os.path.join(output_folder, img_file.replace('.jpg', '.png')),
                (255*mask).astype(np.uint8),
                check_contrast=False)

            # Save image with bbox annotation, for presenting them.
            start_point, end_point = bbox_pred[0:2], bbox_pred[2:4]
            img = cv2.rectangle(img, start_point, end_point, color=(255, 0, 0), thickness=2)
            imsave(
                os.path.join(output_folder, img_file),
                img,
                check_contrast=False)

            results.append([img_idx, img_file, bbox_pred])

    # Save CSV with IOU metrics for images.
    df = pd.DataFrame(results, columns=['img_idx','img_file','bbox_pred'])
    df.to_csv(os.path.join(output_folder, f'iou_results_{os.path.basename(images_folder)}.csv'))
    
    # Save pickle file with the list of bboxes of the text positions.
    text_boxes = df['bbox_pred'].to_list()
    pickle.dump(text_boxes, open(os.path.join(output_folder, 'text_boxes.pkl'), "wb"))

if __name__ == "__main__":
    # Single paintings perfecly aligned without background.
    evaluate_text_bbox('./data/qsd1_w5', './results/text_removal_results/qsd1_w5')
    # Single (or 2) paintings with background !!!!! --> NOT PRETENDING TO WORK
    # evaluate_text_bbox('./data/qsd2_w2', './results/text_removal_results/qsd2_w2')
