import os
from typing import List

import cv2
import numpy as np
from joblib import Parallel, delayed


def detect_edges(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    # return the edged image
    return edged


def find_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img, contours):
    tmpImg = img.copy()
    cv2.drawContours(tmpImg, contours, -1, (0, 255, 0), 2)
    return tmpImg


def rotate_frame(img, angle=0, mask=False):
    tmpImg = img.copy()
    (h, w) = img.shape[:2]
    origin = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(origin, angle, 1.0)
    rotatedImg = cv2.warpAffine(
        tmpImg, mat, (w, h), flags=cv2.INTER_NEAREST if mask else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotatedImg


def compute_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    rads = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    grades = 180 * rads / np.pi
    return grades


def get_angles(image: np.ndarray):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect edges in the image (image, threshold1, threshold 2, aperture size, L2gradient)
    img = cv2.Canny(image, 50, 100)

    # Hough Lines (image, rho, theta, threshold, minLineLength, maxLineGap)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, None, 10, img.shape[0] / 10)
    result_lines = np.zeros_like(img)
    if lines is not None:
        for line in range(0, len(lines)):
            li = lines[line][0]
            cv2.line(result_lines, (li[0], li[1]), (li[2], li[3]), (255, 255, 255), 3, cv2.LINE_AA)

    # We get the contours, hoping we get as much as the frame contours as possible
    contours, hierarchy = cv2.findContours(result_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_ = np.zeros_like(result_lines)
    cv2.drawContours(mask_, contours, -1, (255, 255, 255), thickness=4)

    # We then extract the lines again, saving their angle respect the X axis.
    lines_mask = cv2.HoughLinesP(mask_, 1, np.pi / 180, 15, None, 50, 100)

    angles_list = []
    if lines_mask is not None:
        for line in range(0, len(lines_mask)):
            l = lines_mask[line][0]
            v1 = (l[2] - l[0], l[3] - l[1])
            v2 = (1, 0)  # Is this correct?
            grades = compute_angle(v1, v2)
            angles_list.append(grades)
    angles = np.array(angles_list)
    max_, max_val = -90, 0
    current = -90
    eps = 5 # epsilon
    step = 0.1
    for i in range(int(90 / step)):
        current_count = np.where(((np.array(angles) > current - eps) & (np.array(angles) < current + eps)) | 
                                 ((np.array(angles) > current + 90 - eps) & (np.array(angles) < current + 90 + eps)))
        count = len(current_count[0])
        if count > max_val:
            max_val = count
            max_ = current
        current += step
    angles = [a for a in angles if (a > max_-eps and a < max_+eps) or (a > max_-eps+90 and a < max_+eps+90)]
    angle = angles[np.argmin(np.abs(angles))]
    if angle < 0:
        return 180.0 + angle
    return angle


def generate_cropped_angles(
        query_images_paths: List[str],
        output_dir: str,
        N_PROCESS=4,
        **kwargs,
) -> None:
    """Compute the images' mask for different color spaces and frame sizes,
    and save them in the output path.
    Args:
        query_images_paths (str): The folder where the images are located.
        output_dir (str): The folder where to save the masks."""

    def _get_angles(query_images_paths, output_dir):
        for image_filename in os.listdir(query_images_paths):
            if image_filename.endswith('.jpg'):
                output_folder = os.path.join(output_dir)
                os.makedirs(output_folder, exist_ok=True)

    Parallel(n_jobs=N_PROCESS)(
        delayed(_get_angles)(
            query_images_paths, output_dir,
        )
    )


if __name__ == '__main__':
    # read an image
    img = cv2.imread('./data/qsd1_w5_denoised/00025.jpg')
    # Detect edges:
    angle = get_angles(img)
    print(0)
