import os
import argparse
from typing import List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from skimage import io, color, feature, morphology
import cv2
import pickle
import matplotlib.pyplot as plt

import image_utils


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order.
    https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
    """
    # sort the points based on their x-coordinates
    xSorted = points[np.argsort(points[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    return np.array([tl, bl, br, tr])


def compute_angle(points: List[Tuple]) -> float:
    # Lower corners of the rectangle (bottom)
    bottom_points = sorted(points, key=lambda x: x[0], reverse=True)[:2]
    # Get the line that connects the two points
    line = np.abs(np.array(bottom_points[1]) - np.array(bottom_points[0]))
    # Get the angle of the line
    angle = np.arctan2(*line) * 180 / np.pi
    # Set the angle to be positive
    if angle < 0:
        angle = angle + 180
    return angle


def get_pca(mask: np.ndarray) -> np.ndarray:
    """
    Compute PCA of a mask.
    """
    mask = np.clip(mask, 0, 1)
    # Arrage the points of mask in a matrix
    x, y = np.where(mask == 1)
    X = np.vstack((x, y)).T
    # Calculate mean and subsctract it
    mean = np.mean(X, axis=0)
    X = X - mean
    # Calculate covariance matrix
    cov = np.cov(X, rowvar=False)
    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)
    # Find the direction along which the data
    # varies the most (principal component) and sort it
    idx = np.argsort(eigvals)[::-1]
    evecs = eigvecs[:, idx]

    return X, mean, evecs


def get_pca_points(mask: np.ndarray) -> List[Tuple]:
    """
    Returns the points of the mask that are on the PCA line.
    """
    # Compute eigenvectors from PCA
    X, mean, evecs = get_pca(mask)
    # Change points to PCA basis
    a = X @ evecs
    # Get far most left, right, top and bottom points of mask
    l, r = a[:, 0].min(), a[:, 0].max()
    b, t = a[:, 1].min(), a[:, 1].max()
    points = [[l, b], [l, t], [r, t], [r, b]]

    # Get projection of points in original basis
    proj_points = np.array(points) @ np.linalg.inv(evecs) + mean
    # Adjust points outside the image
    proj_points[:, 0] = np.clip(proj_points[:, 0], 0, mask.shape[0] - 1)
    proj_points[:, 1] = np.clip(proj_points[:, 1], 0, mask.shape[1] - 1)
    proj_points = proj_points[:, [1, 0]].astype(int)  # swap axes !!! x and y

    proj_points = order_points(proj_points)
    proj_points = [tuple(x) for x in proj_points.tolist()]
    return proj_points


def draw_mask_points_on_image_batch(images: List[np.ndarray], masks: List[np.ndarray], n: int = 4):
    for img, mask in zip(images[:n], masks[:n]):
        result = draw_mask_points_on_image(img, mask)
        plt.imshow(result)
        plt.show()


def draw_mask_points_on_image(img: np.ndarray, mask: np.ndarray, n: int = 3) -> None:
    mask = fill_connected_components_except_n_biggest(np.int8(mask), n=n)
    mask = np.uint8(mask)
    region_masks = fill_connected_component(mask)
    result = img.copy()
    # scale = int(np.max(result.shape[:2]) / np.min(result.shape[:2]))
    scale = 1
    for region in region_masks:
        points = get_pca_points(region)
        for point in points:
            result = cv2.circle(result, point, radius=0, color=(0, 255, 0), thickness=20 * scale)
        centroid = np.mean(points, axis=0, dtype=int)
        result = cv2.circle(result, centroid, radius=0, color=(255, 0, 0), thickness=20 * scale)
    return result


def get_angle_bboxes_multiple(
        masks: List[np.ndarray],
        mssg: str,
        output_dir: str = None,
        n: int = 3,
):
    """
    Returns and saves the angle of the bounding box of each mask.
    """
    results = []
    for mask in tqdm(masks, desc=mssg):
        try:
            mask = fill_connected_components_except_n_biggest(np.int8(mask), n=n)
        except Exception as e:
            print(e)
        mask = np.uint8(mask)
        try:
            region_masks = fill_connected_component(mask)
        except Exception as e:
            print(e)
        query_results = []
        for region in region_masks:
            points = get_pca_points(region)
            angle = compute_angle(points)
            query_results.append([angle, points])

        results.append(query_results)

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, 'frames.pkl')
        pickle.dump(results, open(output_file, "wb"))
        print(f"Saved results to {output_file}")

    return results


def morphology_masks(
        img: np.ndarray,
        *args,
) -> np.ndarray:
    """
    Generates a mask for the given image, using morphological operations.
    """
    image_bw = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # Find edges using Canny Edge Detector (filter based on the derivative
    # of a Gaussian in order to compute the intensity of the gradients)
    edges = feature.canny(image_bw, sigma=2)
    edges = np.float32(edges)
    # Apply closing to fill holes, apply dilation to connect edges
    closing_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((50, 50)))
    dilation_edges = cv2.dilate(closing_edges, np.ones((5, 5)))
    # Apply reconstruction by erosion to get the mask
    mar = np.ones(dilation_edges.shape)
    mar[:, [0, -1]] = 0
    mar[[0, -1], :] = 0
    mar = np.maximum(mar, dilation_edges)
    reconstruc_edges = morphology.reconstruction(mar, dilation_edges, method='erosion') * 255
    reconstruc_edges = fill_connected_components_except_n_biggest(np.int8(reconstruc_edges), n=3)
    return reconstruc_edges


def post_process_mask(mask: np.ndarray, kernel_size=5) -> np.ndarray:
    """
    Applies a morphological filter (opening + closing) to remove noise.
    """
    kernel = np.ones((kernel_size) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def erode(ffm: np.array, kernel: np.array = np.ones((2, 2), np.uint8), iterations: int = 15):
    return cv2.erode(ffm, kernel=kernel, iterations=iterations)


def toGray(tresh, cv2_command=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(tresh, cv2_command)


def fill_connected_components_except_n_biggest(mask: np.ndarray, n=3) -> np.ndarray:
    """
    Find the n biggest connected components (white blobs in the mask)
    """

    # im_with_separated_blobs is an image where each detected blob 
    # has a different pixel value ranging from 1 to nb_blobs - 1
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    # stats gives some information about the blobs
    # here, we're interested only in the size of the blobs
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component,
    # which for our applications is not adequate
    sizes = sizes[1:]
    nb_blobs -= 1

    size_idx = [(idx, size) for idx, size in zip(range(1, nb_blobs + 1), sizes)]
    size_idx = sorted(size_idx, key=lambda tup: tup[1], reverse=True)

    # output mask with just the kept components
    mask_result = np.zeros((mask.shape))
    # keep only the n=2 biggest components
    # if the n-th mask is less than 10% of the image, we discard it
    min_size = int(0.05 * mask.shape[0] * mask.shape[1])
    for i in range(min(n, nb_blobs)):
        if size_idx[i][1] >= min_size:
            mask_result[im_with_separated_blobs == size_idx[i][0]] = 255

    return mask_result


def fill_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Fills the leftmost (or rightmost) connected component (white blob in the mask).
    If there is a single conneted component, it gets filled.
    """

    nb_blobs, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    # stats gives some information about the blobs
    # here, we're interested only in the bounding boxes
    X, Y, H, W = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    # the following lines result in taking out the background which is also considered a component,
    # for our applications we don't want the background
    X, Y, H, W = X[1:], Y[1:], H[1:], W[1:]
    nb_blobs -= 1

    try:
        assert 1 <= nb_blobs <= 3, f"There should be between 1 and 3 connected components in the mask, found {nb_blobs}"
    except AssertionError as e:
        print(e)
        plt.imshow(mask)

    bboxes = []
    for i in range(nb_blobs):
        bboxes.append((X[i], Y[i], X[i] + H[i], Y[i] + W[i]))

    bboxes = sorted(bboxes, key=lambda tup: tup[0], reverse=False)  # sort bboxes by y coordinate (from left to right)
    mask_results = []
    for x1, y1, x2, y2 in bboxes:
        # output mask with just the kept component
        mask_bbox = np.zeros((mask.shape))
        mask_bbox[y1:y2, x1:x2] = 1

        mask_res = mask * mask_bbox
        mask_results.append(mask_res)

    return mask_results


## Mean background method ##

def frame_corners(mask: np.ndarray, value: int = 1):
    idx = np.where(mask == value)
    try:
        return idx[0][0], idx[0][-1], np.min(idx[1]), np.max(idx[1])
    except:
        return mask.shape[0], 0, mask.shape[1], 0


def square_mask(mask: np.ndarray) -> np.ndarray:
    """
    Converts the given mask to a square mask.
    """
    crn = frame_corners(mask)
    square_mask = np.ones(mask.shape, dtype='uint8')
    square_mask[:crn[0]] = 0
    square_mask[crn[1]:] = 0
    square_mask[:, :crn[2]] = 0
    square_mask[:, crn[3]:] = 0
    return square_mask


def background_color(image, frame_size):
    """Select the frames pixels of the images and return their 
        mean and standard deviation."""
    ch = image.shape[-1]
    frame = np.concatenate((np.reshape(image[0:frame_size + 1, :], (-1, ch)),
                            np.reshape(image[-frame_size:, :], (-1, ch)),
                            np.reshape(image[:, 0:frame_size + 1], (-1, ch)),
                            np.reshape(image[:, -frame_size:], (-1, ch))))
    return np.mean(frame, axis=(0, 1)), np.std(frame, axis=0)


def get_mask_meanBackground(image, frame_size):
    m, sd = background_color(image, frame_size)
    z = (image - m) / sd
    return abs(z) >= 1.96


def generate_mask_meanBackground(image_filename, color_space, frame_size, channels):
    """Generate the mask of a given image, using the given color space,
        frame size and channels to consider."""

    image = io.imread(image_filename)
    if color_space == 'CieLab':
        image = color.rgb2lab(image)
    elif color_space == 'YCbCr':
        image = color.rgb2ycbcr(image)
    elif color_space == 'HSV':
        image = color.rgb2hsv(image)

    # IMPORTANT. We get the mask for each channel in the color space,
    # and apply the intersection mask of each channel.
    image_masks = []
    for ch in channels:
        mask = get_mask_meanBackground(image[:, :, [ch]], frame_size)
        image_masks.append(mask)

    mask = np.logical_and(*image_masks)
    return mask


def generate_masks_meanBackground(
        query_images_paths: List[str],
        output_dir: str,
        frame_sizes=(2, 4, 10),
        color_space_channels=[('RGB', (0, 1, 2)), ('CieLab', (1, 2)), ('YCbCr', (1, 2)), ('HSV', (0, 1))],
        N_PROCESS=4,
        **kwargs,
) -> None:
    """Compute the images' mask for different color spaces and frame sizes,
    and save them in the output path.
    Args:
        query_images_paths (str): The folder where the images are located.
        output_dir (str): The folder where to save the masks."""

    def _generate_meanBackground(query_images_paths, output_dir, color_space, frame_size, channels):
        for image_filename in os.listdir(query_images_paths):
            if image_filename.endswith('.jpg'):
                mask = generate_mask_meanBackground(os.path.join(query_images_paths, image_filename), color_space,
                                                    frame_size, channels)
                mask = square_mask(mask)

                output_folder = os.path.join(output_dir, color_space, str(frame_size))
                os.makedirs(output_folder, exist_ok=True)
                io.imsave(
                    os.path.join(output_folder, image_filename.replace('.jpg', '.png')),
                    (255 * mask).astype(np.uint8),
                    check_contrast=False
                )

    Parallel(n_jobs=N_PROCESS)(
        delayed(_generate_meanBackground)(
            query_images_paths, output_dir, color_space, frame_size, channels,
        ) for color_space, channels in color_space_channels for frame_size in tqdm(frame_sizes)
    )


def generate_masks(
        query_images_paths: List[str],
        output_dir: str,
        method: str,
        N_PROCESS=4,
        **kwargs,
) -> None:
    """Given a list of query images paths (where the query images are located),
    compute their masks and save these masks in the output path.
    Args:
        query_path (str): The folder where the images are located.
        output_path (str): The folder where to save the masks.
        N_PROCESS (int)
        **kwargs
    """
    methods = {
        "meanBackground": generate_masks_meanBackground,
        "otsu": generate_masks_otsu,
        "floodFill": generate_masks_floodFill,
    }
    methods[method](query_images_paths, output_dir, N_PROCESS=N_PROCESS, **kwargs)


## Otsu's method ##

def otsus_method(img_gray, th=None):
    # create the thresholded image
    if not th:
        th = np.max(np.int8(img_gray)) + 1  # WE CAN PLAY WITH THIS

    # compute the histogram of the gray image
    hist, bin_edges = np.histogram(img_gray, bins=th)

    # normalize the histogram
    hist = hist / np.sum(hist)

    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # compute the probability of each gray level
    prob = hist / img_gray.size
    # compute the mean intensity
    mean_val = np.sum(prob * bin_mids)
    # initialize the maximum variance
    max_var = 0
    # initialize the threshold
    threshold = 0
    # loop over all possible threshold values
    for t in range(th):
        # compute the weight of the two classes
        w0 = np.sum(prob[:t])
        w1 = np.sum(prob[t:])

        # compute the means of the two classes
        m0 = np.sum(prob[:t] * bin_mids[:t]) / w0
        m1 = np.sum(prob[t:] * bin_mids[t:]) / w1

        # compute the variance between the two classes
        var = w0 * w1 * (m0 - m1) ** 2

        # check to see if the variance is larger than the maximum
        if var > max_var:
            max_var = var
            threshold = t

    # create binary image
    img_bin = img_gray < threshold

    return img_bin.astype(np.uint8)


def generate_mask_otsu(image_filename, th=256, color_space="gray", channels=(0), **kwargs):
    """Generate the mask of a given image, using the given color space,
        frame size and channels to consider."""

    image = cv2.imread(image_filename)
    if color_space == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'CieLab':
        image = color.rgb2lab(image)
    elif color_space == 'YCbCr':
        image = color.rgb2ycbcr(image)
    elif color_space == 'HSV':
        image = color.rgb2hsv(image)

    # IMPORTANT. We get the mask for each channel in the color space,
    # and apply the intersection mask of each channel.
    image_masks = []
    if color_space == 'gray':
        mask = otsus_method(image, th=th)
    else:
        for ch in channels:
            mask = otsus_method(image[:, :, [ch]], th=th)
            image_masks.append(mask)
        mask = np.logical_and(*image_masks)
    mask = _extract_mask(mask)
    return mask


def generate_masks_otsu(
        query_images_paths: List[str],
        output_dir: str,
        N_PROCESS: int = 4,
        **kwargs,
) -> List[np.ndarray]:
    """
    Computes the masks of the images in the query_images_paths folder,
    """

    def _generate_otsu(
            image_path: str,
            output_dir: str,
            **kwargs,
    ):
        mask = generate_mask_otsu(image_path, **kwargs)
        output_folder = os.path.join(output_dir, )
        os.makedirs(output_folder, exist_ok=True)
        io.imsave(
            os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '.png')),
            (mask).astype(np.uint8),
            check_contrast=False,
        )

    Parallel(n_jobs=N_PROCESS)(
        delayed(_generate_otsu)(
            img_path, output_dir, **kwargs,
        ) for img_path in tqdm(query_images_paths, desc="Generating masks with Otsu's method")
    )


def _extract_mask(tresh):
    x, y, w, h = cv2.boundingRect(tresh)
    mask = np.zeros(tresh.shape, dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return mask

def _extract_mask_roated(mask, tresh):
    rect = cv2.minAreaRect(tresh)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(mask, [box], 0, (0, 0, 255), 2)
    mask = cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
    return mask


## Floodfill method ##

def flood_fill_method(
        img: np.ndarray,
        kernel: np.array,
        iterations: int,
        loDiff: tuple,
        upDiff: tuple,
        extract_mask: bool = True
) -> np.ndarray:
    """
    Generates a Mask for each image by flood filling the background
    starting from the boundaries.

    Parameters:
    img: An array containing the image with background.

    Returns:
    mask: mask with 1 in the foreground and 0 in the background
    """
    mask = img.copy()

    # Find the Off White color.
    tmp = cv2.dilate(img, kernel, iterations=iterations)
    # cv2.imshow("dilated image", tmp)

    # Color of Off-White pixel
    offwhite = tmp[0, 0, :]
    offwhite = tuple((int(offwhite[0]), int(offwhite[1]), int(offwhite[2])))

    # Fill black pixels with off-white color
    cv2.floodFill(mask, None, seedPoint=(0, 0), newVal=offwhite)

    # Fill off-white pixels with black color
    cv2.floodFill(mask, None, seedPoint=(0, 0), newVal=0, loDiff=loDiff, upDiff=upDiff)

    # Find the black pixels in the image
    black_pixels_image = np.where(
        (img[:, :, 0] == 0) &
        (img[:, :, 1] == 0) &
        (img[:, :, 2] == 0)
    )
    # Convert the black pixels in the image to white
    mask[black_pixels_image] = [255, 255, 255]

    # Mask in binary
    # get (i, j) positions of all RGB pixels that are not black
    not_black_pixels_mask = np.where(
        (mask[:, :, 0] != 0) &
        (mask[:, :, 1] != 0) &
        (mask[:, :, 2] != 0)
    )
    # set those pixels to white
    mask[not_black_pixels_mask] = [255, 255, 255]
    if extract_mask:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = _extract_mask(mask)
    return mask


def mask_background_multiple_n_components(
        img: np.ndarray,
        kernel: np.array,
        iterations: int,
        loDiff: tuple,
        upDiff: tuple,
        n: int = 3,
        extract_mask: bool = False,
        kernel_size: int = 7,
) -> np.ndarray:
    """
    Background segmentation using flood fill method.
    Designed to work with images with two components.
    """
    # img_size = img.shape[:2]
    # from skimage.transform import resize
    # img_segm = resize(image=img, output_shape=(800, 800))
    # from skimage.filters import unsharp_mask
    # img_segm = unsharp_mask(img_segm, radius=15, amount=4, multichannel=True)
    mask = flood_fill_method(img, kernel, iterations, loDiff, upDiff, extract_mask=extract_mask)
    mask = flood_fill_method(255 - mask, kernel, iterations, loDiff, upDiff, extract_mask=extract_mask)
    # from skimage.segmentation import chan_vese
    # print("Generating mask with Chan-Vese method")
    # mask = chan_vese(
    #     color.rgb2gray(img_segm), mu=0.01, lambda1=0.8, lambda2=1, tol=1e-3,
    #     max_num_iter=80, dt=0.5, init_level_set="checkerboard",
    #     extended_output=False
    # )
    # print("Mask generated")
    # print("type(mask):", type(mask))
    # print('mask.shape:', mask.shape)
    # print("mask: ", mask)
    mask = post_process_mask(mask, kernel_size=kernel_size)
    # mask = resize(image=mask, output_shape=img_size)

    mask = color.rgb2gray(mask)
    mask = fill_connected_components_except_n_biggest(np.int8(mask), n=n)
    return mask


def mask_background_multiple_n_components_n(
        img: np.ndarray,
        kernel: np.array,
        iterations: int,
        loDiff: tuple,
        upDiff: tuple,
        n: int = 3,
        extract_mask: bool = True,
        kernel_size: int = 7,
) -> np.ndarray:
    """
    Background segmentation using flood fill method.
    Designed to work with images with two components.
    """
    mask = img.copy()
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)

    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort the max contour area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:n]
    # filter by max contour area
    if len(contours) > 1:
        max_contour_area = cv2.contourArea(contours[0]) - cv2.contourArea(contours[1])
    else:
        max_contour_area = cv2.contourArea(contours[0])
    contours = [c for c in contours if cv2.contourArea(c) >= max_contour_area / 10]

    image_copy = mask.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")
    # Fill area with white color
    mask = np.zeros(mask.shape, dtype=np.uint8)
    # black colour outside the contours
    # cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))

    # white colour on the max contours
    #cv2.fillPoly(mask, pts=[contours[0]], color=(255, 255, 255))
    cv2.drawContours(mask, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=3)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(np.uint8(closing))
    if nb_blobs == 2:
        mask = _extract_mask_roated(mask, contours[0])
    return mask


def generate_masks_multiple(
        images,
        N_PROCESS: int = 4,
        mssg: str = "Generating masks for queries with multiple paitings...",
        image_files: List[str] = None,
        output_dir: str = None,
        method: str = "flood_fill",
        n: int = 3,
        **kwargs,
) -> List[np.ndarray]:
    """
    Produces masks for each image in the list of images.
    """
    assert 0 < n < 4, "n must be between 1 and 3"
    _flood_fill_method = mask_background_multiple_n_components if n > 1 else flood_fill_method

    methods = {
        "flood_fill": _flood_fill_method,
        "morphology": morphology_masks,
        "mask_multiple_n": mask_background_multiple_n_components_n,
    }
    __method__ = methods[method]

    masks = Parallel(n_jobs=N_PROCESS)(
        delayed(__method__)(
            img, np.ones((2, 2), np.uint8), 10000, (2,) * 4, (2,) * 4, n
        ) for img in tqdm(images, desc=mssg)
    )

    if image_files is not None:
        for mask, image_file in zip(masks, image_files):
            output_folder = os.path.join(output_dir, )
            os.makedirs(output_folder, exist_ok=True)
            io.imsave(
                os.path.join(output_folder, os.path.basename(image_file).replace('.jpg', '.png')),
                (mask).astype(np.uint8),
                check_contrast=False,
            )

    return masks
