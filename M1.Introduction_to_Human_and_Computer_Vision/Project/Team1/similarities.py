from typing import List

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.stats.stats import pearsonr
from tqdm import tqdm

from histogram_processing import generate_feature_matrix


def euclidean_distance(a: np.array, b: np.array, mask: np.bool = None) -> float:
    """
    Return the Euclidean distance of two arrays.
    """
    return np.linalg.norm((a - b) * mask, ord=2)


def l1_distance(a: np.array, b: np.array, mask: np.bool = None) -> float:
    """
    Return the L1 distance of two arrays.
    """
    return np.linalg.norm((a - b) * mask, ord=1)


def chi_square_distance(a: np.array, b: np.array, mask: np.bool = None) -> float:
    """
    Return the Chi-square distance of two arrays.
    """
    return np.sum((((a - b) * mask) ** 2) / ((a + b) * mask + 1e-5))


def histogram_intersection(a: np.array, b: np.array) -> float:
    """
    Return the Histogram intersection (similarity) of two arrays.
    """
    return np.sum(np.minimum(a, b))


def hellinger_kernel(a: np.array, b: np.array, mask: np.bool = None) -> float:
    """
    Return the Hellinger kernel (similarity) of two arrays.
    """
    # 0 is max similarity, so we invert the scale by subtracting the max and taking abs (e.g. [0.1, 2, 10] becomes [9.9, 8, 0])
    return np.sqrt(np.sum((np.sqrt(a) * mask - np.sqrt(b) * mask) ** 2)) / np.sqrt(2)


def cosine_similarity(a: np.array, b: np.array, mask: np.bool = None) -> float:
    """
    Return the Cosine similarity of two arrays.
    """
    return np.dot(a, b) * mask / (np.linalg.norm(a * mask) * np.linalg.norm(b * mask))


def histogram_correlation(vector1: np.ndarray, vector2: np.ndarray, n_bins: int = None) -> np.ndarray:
    """
    Return histogram correlation for each channel between feature vectors
    """
    n_bins = int(len(vector1) / 4) if n_bins is None else n_bins
    corr_ch = []
    for i in range(0, len(vector1), n_bins):
        corr_ch.append(pearsonr(vector1[i:i + n_bins], vector2[i:i + n_bins]))

    return np.mean(corr_ch)


def compute_similarities(
        query_features: np.array,
        db_images_features: np.array,
        similarity_measure,
        mask: np.bool = None,
        n_bins: int = None,
) -> np.array:
    if mask is None:
        mask = np.ones_like(query_features, dtype=bool)
    if similarity_measure == histogram_correlation:
        dist = np.array([similarity_measure(query_features, features_img, n_bins)
                         for features_img in db_images_features])
    else:
        dist = np.array([similarity_measure(query_features, features_img, mask)
                         for features_img in db_images_features])
    if similarity_measure in (hellinger_kernel, euclidean_distance, l1_distance, chi_square_distance,):
        # smaller values indicate more similarity, so we invert the scale
        # by subtracting the max and taking abs (e.g. [0.1, 2, 10] becomes [9.9, 8, 0])
        dist = -(dist - np.amax(dist))
        dist = np.nan_to_num(dist)
    return dist


def compute_similarities_parallel(
        query_features,
        db_feature_matrix,
        similarity_measure,
        mssg,
        N_PROCESS,
        masks: np.bool = None,
):
    if masks is not None:
        return Parallel(n_jobs=N_PROCESS)(delayed(compute_similarities)
                                          (query_img, db_feature_matrix, similarity_measure, mask)
                                          for query_img, mask in
                                          tqdm(zip(query_features, masks), desc=mssg, total=len(query_features))
                                          )
    return Parallel(n_jobs=N_PROCESS)(delayed(compute_similarities)(
        query_img, db_feature_matrix, similarity_measure) for query_img in tqdm(query_features, desc=mssg)
                                      )


measures = {
    'euclidean_distance': euclidean_distance,
    'l1_distance': l1_distance,
    'chi_square_distance': chi_square_distance,
    'histogram_intersection': histogram_intersection,
    'hellinger_kernel': hellinger_kernel,
    'cosine_similarity': cosine_similarity,
    'histogram_correlation': histogram_correlation,
}


def compute_similarities_batch(
        query_images: List,
        dataset: List,
        similarity_measure,
        n_levels: int = 1,
        n_bins=16,
        grayscale=False,
        RGB=False,
        CieLab: bool = False,
        HSV: bool = False,
        YCbCr: bool = False,
        histogram3d: bool = False,
        masks_queries: List = None,
        masks_database: List = None,
        mssg="Computing similarities...",
        N_PROCESS: int = 2,
) -> np.array:
    similarity_measure = measures[similarity_measure]

    query_features = generate_feature_matrix(
        query_images,
        mssg="Generating features for query dataset...(with N_PROCESS = {})".format(
            N_PROCESS),
        grayscale=grayscale,
        n_levels=n_levels,
        n_bins=n_bins,
        RGB=RGB,
        CieLab=CieLab,
        HSV=HSV,
        YCbCr=YCbCr,
        histogram3d=histogram3d,
        masks=masks_queries,
        N_PROCESS=N_PROCESS,
    )
    print("query_features.shape", query_features.shape)

    db_feature_matrix = generate_feature_matrix(
        dataset=dataset,
        grayscale=grayscale,
        n_levels=n_levels,
        n_bins=n_bins,
        RGB=RGB,
        CieLab=CieLab,
        HSV=HSV,
        YCbCr=YCbCr,
        histogram3d=histogram3d,
        mssg="Generating features for BBDD dataset...(with N_PROCESS = {})".format(
            N_PROCESS),
        masks=masks_database,
        N_PROCESS=N_PROCESS,
    )
    print("db_feature_matrix.shape", db_feature_matrix.shape)

    dist = compute_similarities_parallel(query_features, db_feature_matrix, similarity_measure, mssg, N_PROCESS)

    return dist


def compute_similarities_multiple(
        features_multiple,
        db_features: np.array,
        similarity_measure: str,
        n_bins: int,
):
    # features_multiple is a list of lists of features, where each list in the first level
    # corresponds to a query image, and each list in the second level corresponds to a picture
    # within the query image (there may be up to two pictures in a query image).

    similarity_measure = measures[similarity_measure]

    sims = []
    for query_features in features_multiple:
        sims_query = []
        for pic_features in query_features:
            sims_query.append(
                compute_similarities(
                    query_features=pic_features,
                    db_images_features=db_features,
                    similarity_measure=similarity_measure,
                    n_bins=n_bins,
                )
            )
        sims.append(sims_query)

    return sims


def Flann_matcher(descriptor1, descriptor2, k=2, plot=False):
    try:
        # Fast Matcher
        index_params = dict(algorithm = 0, trees = 4)
        search_params = dict(checks=30)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Matching descriptors
        matches = matcher.knnMatch(np.float32(descriptor1), np.float32(descriptor2), k=k)
        # Delete possible false positives
        matches = [m for m,n in matches if m.distance < 0.7*n.distance]

        return len(matches)
    except Exception as e:
        print("Error in Flann_matcher.")
        print(e)
        raise Exception(e)


def BF_matcher(descriptor1, descriptor2, similarity_measure, plot = False):
    try:
        # Brute Force Matcher
        bf = cv2.BFMatcher(similarity_measure)
        if similarity_measure==cv2.NORM_HAMMING or similarity_measure==cv2.NORM_HAMMING2:
            matches = bf.knnMatch(np.asarray(descriptor1,np.uint8), np.asarray(descriptor2,np.uint8), k=2)
        else:
            matches = bf.knnMatch(np.asarray(descriptor1,np.float32), np.asarray(descriptor2,np.float32), k=2)
        # Delete possible false positives
        matches = [m for m,n in matches if m.distance < 0.7*n.distance]

        return len(matches)
    except Exception as e:
        print("Error in BF_matcher.")
        print(e)
        raise Exception(e)


def compute_matches(
    descriptor1: np.array,
    descriptor2: List[np.array],
    method: str = "BF",
    similarity_measure: str = "L2",
    k: int = 2,
    plot: bool = False,
    N_PROCESS: int = 2,
) -> int:
    if method == "BF":
        return Parallel(n_jobs=N_PROCESS)(delayed(BF_matcher)(
            descriptor1, desc2, similarity_measure, plot) for desc2 in tqdm(descriptor2, leave=False, desc="Computing matches with BF_matcher...")
        )
    elif method == "FLANN":
        return Parallel(n_jobs=N_PROCESS)(delayed(Flann_matcher)(
            descriptor1, desc2, k, plot) for desc2 in tqdm(descriptor2, leave=False, desc="Computing matches with BF_matcher...")
        )
    raise Exception("Method not recognized. Must be either 'BF' or 'FLANN'.")


def compute_matches_batch(
    descriptor1: np.array,
    descriptor2: List[np.array],
    method: str = "BF",
    similarity_measure: str = "L2",
    k: int = 2,
    plot: bool = False,
    N_PROCESS: int = 2,
) -> np.array:
    res = []
    for desc1 in descriptor1:
        if method == "BF":
            res.append( Parallel(n_jobs=N_PROCESS)(delayed(BF_matcher)(
                    desc1, desc2, similarity_measure, plot) for desc2 in tqdm(descriptor2, leave=False, desc="Computing matches with BF_matcher...")
                )
            )
        elif method == "FLANN":
            res.append(Parallel(n_jobs=N_PROCESS)(delayed(Flann_matcher)(
                    desc1, desc2, k, plot) for desc2 in tqdm(descriptor2, leave=False, desc="Computing matches with BF_matcher...")
                )
            )
    return np.array(res)


def compute_matches_multiple(
    features_multiple: List[List[np.array]],
    db_features: List[List[np.array]],
    method: str = "BF",
    similarity_measure: str = "L2",  # this is ignored when method == "FLANN"
    k: int = 2,  # only used when method == "FLANN"
    N_PROCESS: int = 2,
) -> List[List[int]]:
    # features_multiple is a list of lists of features, where each list in the first level
    # corresponds to a query image, and each list in the second level corresponds to a picture
    # within the query image (there may be up to two pictures in a query image).
    distances_BF = {"L1": cv2.NORM_L1, "L2": cv2.NORM_L2, "Hamming": cv2.NORM_HAMMING, "Hamming2": cv2.NORM_HAMMING2}
    similarity_measure = distances_BF[similarity_measure]

    i_out = 0
    sims = []
    for query_features in tqdm(features_multiple, leave=False, desc="Computing matches..."):
        i_out = i_out + 1
        sims_query = []
        for pic_features in tqdm(query_features, leave=False, desc="Computing matches for query image {}...".format(i_out)):

            n_maches = compute_matches(
                            descriptor1=pic_features,
                            descriptor2=db_features,
                            method=method,
                            similarity_measure=similarity_measure,
                            k=k,
                            N_PROCESS=N_PROCESS,
                        )
            energy = np.sqrt(np.sum(np.array(n_maches)**2))

            if energy == 0:
                sims_query.append(np.zeros_like(n_maches))
            else:
                energy = 1
                sims_query.append(np.array(n_maches) / energy)
        sims.append(sims_query)
    return sims