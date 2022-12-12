import os
import glob
import pickle
from typing import List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import cv2
from skimage.feature import hog, ORB, local_binary_pattern, multiblock_lbp
from skimage.color import rgb2gray
from skimage.transform import resize, integral_image
import scipy
from skimage.io import imread
from enum import Enum


class Mode(Enum):
    QUERY = 0
    IMAGE = 1


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def dct2_block(image: np.ndarray, quantize: bool) -> np.ndarray:
    image = resize(image=image, output_shape=(325, 325))
    imsize = image.shape
    dct = np.zeros(imsize)

    # Do 8x8 DCT on dct (in-place)
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct[i:(i + 8), j:(j + 8)] = dct2(image[i:(i + 8), j:(j + 8)])

    if quantize:
        # Quantize the DCT coefficients
        # keep only 4 first coefficients (two first of each axis)
        dct = np.array([dct[i:(i + 1), j:(j + 1)] for i in np.r_[:imsize[0]:8] for j in np.r_[:imsize[1]:8]]).flatten()
    return dct


def dct_batch(images: np.ndarray, mssg: str, quantize: bool, N_PROCESS=4) -> np.ndarray:
    return np.array(
        Parallel(n_jobs=N_PROCESS)(delayed(dct2_block)(img, quantize) for img in tqdm(images, desc=mssg))
    ).astype(np.float16)


def hog_image(image: np.ndarray, image_mask=None) -> np.ndarray:
    image = resize(image=image, output_shape=(350, 350))
    h_im, im = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=2 if len(image.shape) == 3 else None,
    )
    return im.astype(np.float32)


def hog_batch(images: np.ndarray, mssg: str, N_PROCESS=4) -> np.ndarray:
    features = [hog_image(img, image_mask=None) for img in tqdm(images, desc=mssg)]
    return features


def hog_multiple(files_dir, features_filename: str):
    """
    Compute HoG features for query image containing potentially up to 2 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features = []
    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query = []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            features_query.append(
                hog_image(im_query)
            )
        features.append(features_query)

    # pickle.dump(features, open(os.path.join(files_dir, features_filename), "wb"))
    return features


def lbp(image: np.ndarray, histogram: bool) -> np.ndarray:
    image = resize(image=image, output_shape=(500, 500))
    bw_image = rgb2gray(image)
    lbp_feats = local_binary_pattern(image=bw_image, P=10, R=5).astype(np.float16).flatten()
    if histogram:
        lbp_feats = np.histogram(lbp_feats, bins=16, density=True)[0]
    return lbp_feats


def lbp_block(image: np.ndarray) -> np.ndarray:
    image = resize(image=image, output_shape=(500, 500))
    bw_image = (rgb2gray(image) * 255).astype(np.uint8)
    bw_image = integral_image(bw_image)
    return np.array([multiblock_lbp(int_image=bw_image, r=3, c=3, width=int(bw_image.shape[0] / 9),
                                    height=int(bw_image.shape[0] / 9))])


def lbp_batch(images: np.ndarray, block: bool, histogram: bool, mssg: str) -> np.ndarray:
    if not block:
        return np.array(Parallel(n_jobs=-1)(delayed(lbp)(file, histogram) for file in tqdm(images, desc=mssg))).astype(
            np.float32)
    return np.array(Parallel(n_jobs=-1)(delayed(lbp_block)(file) for file in tqdm(images, desc=mssg))).astype(
        np.float32)


def orb_image(image: np.ndarray, fastThreshold=None, edgeThreshold=None, image_mask=None) -> np.ndarray:
    if fastThreshold is not None and edgeThreshold is not None:
        orb = cv2.ORB_create(nfeatures=1200, fastThreshold=fastThreshold, edgeThreshold=edgeThreshold)
    else:
        orb = cv2.ORB_create(nfeatures=1200, scoreType=cv2.ORB_FAST_SCORE)
    image = resize(image=image, output_shape=(500, 500))
    bw_image = rgb2gray(image) * 255
    bw_image = bw_image.astype(np.uint8)
    if image_mask is not None:
        image_mask = resize(image=image_mask, output_shape=(500, 500))
        base_mask = np.array(np.where(image_mask, 0, 255), dtype=np.uint8)
        keypoints, descriptors = orb.detectAndCompute(bw_image, mask=base_mask)
    else:
        keypoints, descriptors = orb.detectAndCompute(bw_image, None)
    
    if descriptors is None:
        res = orb_image(
            image + np.random.randint(-10, high=10, size=image.shape, dtype=int), fastThreshold=0, edgeThreshold=0
        )
        return res[0], res[1]
    return keypoints, descriptors


def orb_batch(images: np.ndarray, mssg: str) -> Tuple[List, List]:
    features = [orb_image(img, image_mask=None) for img in tqdm(images, desc=mssg)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, descriptors


def orb_multiple(files_dir, mssg: str, image_mask=None):
    """
    Compute ORB features for query image containing potentially up to 3 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features, keypoints = [], []
    for dir_name in tqdm(dir_names, desc=mssg):
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query, keypoints_query = [], []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            res = orb_image(im_query, image_mask=mask_query)
            keypoints_query.append(res[0])
            features_query.append(res[1])

        features.append(features_query)
        keypoints.append(keypoints_query)

    return keypoints, features


def sift_image(image: np.ndarray, image_mask=None) -> np.ndarray:
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1200)
    image = resize(image=image, output_shape=(500, 500))
    bw_image = rgb2gray(image) * 255
    bw_image = bw_image.astype(np.uint8)
    if image_mask is not None:
        image_mask = resize(image=image_mask, output_shape=(500, 500))
        base_mask = np.array(np.where(image_mask, 0, 255), dtype=np.uint8)
        keypoints, descriptors = sift.detectAndCompute(bw_image, mask=base_mask)
    else:
        keypoints, descriptors = sift.detectAndCompute(bw_image, None)
    if descriptors is None:
        res = sift_image(image + np.random.randint(-10, high=10, size=image.shape, dtype=int))
        return res[0], res[1]
    return keypoints, descriptors


def sift_batch(images: np.ndarray, mssg: str) -> Tuple[List, List]:
    features = [sift_image(img, image_mask=None) for img in tqdm(images, desc=mssg)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)


def sift_multiple(files_dir, mssg):
    """
    Compute SIFT features for query image containing potentially up to 3 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features, keypoints = [], []
    for dir_name in tqdm(dir_names, desc=mssg):
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query, keypoints_query = [], []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            res = sift_image(im_query, image_mask=mask_query)
            keypoints_query.append(res[0])
            features_query.append(res[1])

        features.append(features_query)
        keypoints.append(keypoints_query)

    return keypoints, features


def surf_image(image: np.ndarray, image_mask=None) -> np.ndarray:
    surf = cv2.xfeatures2d.SURF_create(400)
    image = resize(image=image, output_shape=(500, 500))
    bw_image = rgb2gray(image) * 255
    bw_image = bw_image.astype(np.uint8)
    if image_mask is not None:
        image_mask = resize(image=image_mask, output_shape=(500, 500))
        base_mask = np.array(np.where(image_mask, 0, 255), dtype=np.uint8)
        keypoints, descriptors = surf.detectAndCompute(bw_image, mask=base_mask)
    else:
        keypoints, descriptors = surf.detectAndCompute(bw_image, None)
    if descriptors is None:
        res = surf_image(image + np.random.randint(-10, high=10, size=image.shape, dtype=int))
        return res[0], res[1]
    return keypoints, descriptors


def surf_batch(images: np.ndarray, mssg: str) -> Tuple[List, List]:
    features = [surf_image(img, image_mask=None) for img in tqdm(images, desc=mssg)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)

def surf_multiple(files_dir, mssg):
    """
    Compute SURF features for query image containing potentially up to 3 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features, keypoints = [], []
    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query, keypoints_query = [], []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            res = surf_image(im_query, image_mask=mask_query)
            keypoints_query.append(res[0])
            features_query.append(res[1])

        features.append(features_query)
        keypoints.append(keypoints_query)

    return keypoints, features


def akaze_image(image: np.ndarray, image_mask=None) -> np.ndarray:
    akaze = cv2.AKAZE_create(threshold=1e-6)
    image = resize(image=image, output_shape=(500, 500))
    bw_image = rgb2gray(image) * 255
    bw_image = bw_image.astype(np.uint8)
    if image_mask is not None:
        image_mask = resize(image=image_mask, output_shape=(500, 500))
        base_mask = np.array(np.where(image_mask, 0, 255), dtype=np.uint8)
        keypoints, descriptors = akaze.detectAndCompute(bw_image, mask=base_mask)
    else:
        keypoints, descriptors = akaze.detectAndCompute(bw_image, None)
    if descriptors is None:
        res = akaze_image(image + np.random.randint(-10, high=10, size=image.shape, dtype=int))
        return res[0], res[1]
    return keypoints, descriptors


def akaze_batch(images: np.ndarray, mssg: str) -> Tuple[List, List]:
    features = [akaze_image(img, image_mask=None) for img in tqdm(images, desc=mssg)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)


def akaze_multiple(files_dir, mssg):
    """
    Compute AKAZE features for query image containing potentially up to 3 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features, keypoints = [], []
    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query, keypoints_query = [], []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            res = akaze_image(im_query, image_mask=mask_query)
            keypoints_query.append(res[0])
            features_query.append(res[1])

        features.append(features_query)
        keypoints.append(keypoints_query)

    return keypoints, features


def harris_corner_detector(image, mode=0):
    """
    Extract keypoints from image using Harris Corner Detector.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    dst = cv2.cornerHarris(image, 4, -1, 0.04)
    corners = np.argwhere(dst > dst.max() * 0.10)
    return [cv2.KeyPoint(corner[0], corner[1], 9) for corner in corners]


def harris_corner_subpixel_accuracy(image, mode=0):
    """
    Extract keypoints from image using Harris Corner Detector with subpixel accuracy.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    if mode == Mode.QUERY:
        thresh = 0.15
    elif mode == Mode.IMAGE:
        thresh = 0.10
    else:
        thresh = 0.15

    # find Harris corners
    dst = cv2.cornerHarris(image, 4, -1, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, thresh * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image, np.float32(centroids), (2, 2), (-1, -1), criteria)

    return [cv2.KeyPoint(corner[0], corner[1], 4) for corner in corners]


def harris_corner_batch(images: np.ndarray, mssg: str) -> Tuple[List, List]:
    features = [harris_corner_detector(file) for file in tqdm(images, desc=mssg)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)


def harris_corner_subpixel_batch(images: np.ndarray, mssg: str) -> Tuple[List, List]:
    features = [harris_corner_subpixel_accuracy(file) for file in tqdm(images, desc=mssg)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)


def harris_corner_multiple(files_dir, mssg):
    """
    Compute Harris Corner features for query image containing potentially up to 3 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features, keypoints = [], []
    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query, keypoints_query = [], []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            res = harris_corner_detector(im_query)
            keypoints_query.append(res)
            features_query.append(res)

        features.append(features_query)
        keypoints.append(keypoints_query)
    
    return keypoints, features


def harris_corner_subpixel_multiple(files_dir, mssg):
    """
    Compute Harris Corner features for query image containing potentially up to 3 query paintings.
    """
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    features, keypoints = [], []
    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        features_query, keypoints_query = [], []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            mask_query = imread(mask)
            res = harris_corner_subpixel_accuracy(im_query)
            keypoints_query.append(res)
            features_query.append(res)

        features.append(features_query)
        keypoints.append(keypoints_query)

    return keypoints, features