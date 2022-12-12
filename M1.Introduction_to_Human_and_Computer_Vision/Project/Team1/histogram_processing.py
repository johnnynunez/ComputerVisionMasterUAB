import numpy as np
from joblib import Parallel, delayed
from skimage import color
from skimage.color import rgb2gray
from tqdm import tqdm
import glob
from skimage.io import imread
import os
from skimage.transform import resize

import image_utils


def get_1dhistogram_features(
    image: np.ndarray,
    n_bins: int = 4,
    density: bool = True,
    mask: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """
    Returns the flattened 1d histogram of a single-channel image.
    """
    return np.histogram(image, bins=n_bins, density=density, weights=mask)[0]


def get_3dhistogram_features(
    image: np.ndarray,
    n_bins: int,
    density: bool = True,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Returns the flattened 3d histogram of an image.
    """
    if mask is not None:
        mask = mask.reshape(-1)

    hist = np.histogramdd(image.reshape(-1, 3), bins=n_bins, density=False, weights=mask)[0].flatten()
    hist = hist / (np.sum(hist)+1e-9) / ((256//n_bins)**3) if density else hist
    return hist


def get_histogram_features(
    image: np.ndarray,
    n_bins: int = 4,
    density: bool = True,
    histogram3d: bool = False,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Returns the histogram features for a given image.
    """
    if histogram3d:
        return [get_3dhistogram_features(image, n_bins, density, mask=mask)]
    else:
        return [get_1dhistogram_features(image[:,:,i], n_bins, density, mask=mask) for i in range(0, 3)]


def generate_histogram_features(
        img: np.ndarray,
        n_bins: int = 16,
        grayscale: bool = False,
        RGB: bool = False,
        CieLab: bool = False,
        HSV: bool = False,
        YCbCr: bool = False,
        histogram3d: bool = False,
        mask: np.ndarray = None,
) -> np.ndarray:
    """
    Returns the histogram vector of an image, one histogram per channel.
    The resulting histograms are concatenated into a single vector.
    """
    assert grayscale or RGB or CieLab or HSV or YCbCr, "At least one color space must be selected"
    histograms = []

    kwargs_hist = {"n_bins": n_bins, "density": True, "histogram3d": histogram3d, "mask": mask}

    if grayscale:
        histograms.append(get_1dhistogram_features(rgb2gray(img), **kwargs_hist))  # Grayscale

    if RGB:
        histograms = histograms + get_histogram_features(img, **kwargs_hist)

    if CieLab:
        img_lab = color.rgb2lab(img)
        histograms = histograms + get_histogram_features(img_lab, **kwargs_hist)

    if HSV:
        img_hsv = color.rgb2hsv(img)
        histograms = histograms + get_histogram_features(img_hsv, **kwargs_hist)

    if YCbCr:
        img_ycbcr = color.rgb2ycbcr(img)
        histograms = histograms + get_histogram_features(img_ycbcr, **kwargs_hist)

    # Concat histograms into a single feature vector
    feature_vector = np.hstack(histograms)

    return feature_vector


def multires_local_histograms(
    img,
    n_levels: int,
    n_bins: int,
    grayscale: bool = False,
    RGB: bool = False,
    CieLab: bool = False,
    HSV: bool = False,
    YCbCr: bool = False,
    histogram3d: bool = False,
    mask: np.ndarray = None,
):
    """
    Produces a multiresolution local histogram representation of an image.
    """
    img = resize(image=img, output_shape=(600,900))
    mask = imread(mask) if type(mask) == type("") else mask

    tiles_multilevel = image_utils.tile_image_multilevel(img, n_levels=n_levels)
    if mask is not None:  # mask must be tiled in the same way as the image
        tiles_mask_multilevel = image_utils.tile_image_multilevel(mask, n_levels=n_levels)

    histograms = []
    for lvl in range(n_levels):
        for i, tiles_lvl in enumerate(tiles_multilevel[lvl]):
            mask_lvl_tile = mask if mask is None else tiles_mask_multilevel[lvl][i]
            feats = generate_histogram_features(
                img = tiles_lvl,
                n_bins=n_bins,
                grayscale=grayscale,
                RGB=RGB,
                CieLab=CieLab,
                HSV=HSV,
                YCbCr=YCbCr,
                histogram3d=histogram3d,
                mask=mask_lvl_tile,
            )
            histograms.append(feats)
    return np.hstack(histograms)


def compute_features_multiple(
    files_dir,
    n_levels, n_bins,
    grayscale = False, RGB = False, CieLab = False, HSV = False, YCbCr = False,
    histogram3d = False,
    use_mask=True
    ):
    """
    Compute features for query image containing potentially up to 2 query paintings.
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
            mask_query = imread(mask) if use_mask else None
            features_query.append(
                multires_local_histograms(
                    im_query,
                    n_levels=n_levels,
                    n_bins=n_bins,
                    grayscale=grayscale,
                    RGB=RGB,
                    CieLab=CieLab,
                    HSV=HSV,
                    YCbCr=YCbCr,
                    histogram3d=histogram3d,
                    mask=mask_query,
                )
            )
        features.append(features_query)

    return features


def generate_feature_matrix(
        dataset: np.ndarray,
        mssg: str,
        n_levels: int = 1,
        n_bins: int = 16,
        grayscale: bool = False,
        RGB: bool = False,
        CieLab: bool = False,
        HSV: bool = False,
        YCbCr: bool = False,
        histogram3d: bool = False,
        masks: np.ndarray = None,
        N_PROCESS: int = 2,
) -> np.ndarray:
    """
    For each image in the input array, it generates a feature vector.
    """
    # Multiprocessing allows to parallelize the computation
    if masks is None:
        masks = [None] * len(dataset)
    feature_matrix = Parallel(n_jobs=N_PROCESS)(
        delayed(multires_local_histograms)(
                    image, n_levels, n_bins, grayscale, RGB, CieLab, HSV, YCbCr, histogram3d, mask
                ) for image, mask in tqdm(zip(dataset, masks), desc=mssg, total=len(dataset))
        )
    return np.stack(feature_matrix)
