import os
import cv2
from typing import Dict, List
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
import numpy as np


def cluster_kmeans(imgs: List[np.array], feature_matrix: np.array, cluster_num: int):
    assert len(feature_matrix.shape) == 2

    kmeans = KMeans(cluster_num, n_init=20, max_iter=1000)
    kmeans.fit(feature_matrix)

    preds = kmeans.predict(feature_matrix)
    print("Found Clusters!")

    img_clusters = pd.DataFrame()
    img_clusters["imgs"] = [img for img in imgs]
    img_clusters["clusters"] = preds

    return img_clusters


def cluster_agglomerative(
    imgs: np.array,
    features: np.array,
    cluster_num: int,
    affinity: str
) -> Dict[str, any]:

    clustering = AgglomerativeClustering(
        n_clusters=cluster_num,
        affinity=affinity,  #  If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
        connectivity=None,
        linkage="complete",
        distance_threshold=None,
        compute_distances=False,
    )

    clustering = clustering.fit(features)

    print("Found Clusters!")

    img_clusters = pd.DataFrame()
    img_clusters["imgs"] = [img for img in imgs]
    img_clusters["clusters"] = clustering.labels_

    return img_clusters


def save_clusters(img_clusters: Dict[str, any], desc_method: str):
    print("Storing images in their respective cluster folder...")
    for cluster in set(img_clusters["clusters"].values):
        fp = "./results/" + desc_method + "_clusters"
        imgs_of_clusters = img_clusters[img_clusters["clusters"] == cluster]

        if not os.path.exists(fp):
            os.mkdir(fp)

        fp =  fp + "//" + str(cluster)
        if not os.path.exists(fp):
            os.mkdir(fp)

        for i, _ in enumerate(imgs_of_clusters["imgs"].values):
            img_meta = imgs_of_clusters.iloc[[i]]
            img_path = fp + "//" + "bbdd_" + str(img_meta.index[0]).zfill(5) + ".jpg"
            cv2.imwrite(img_path, img_meta["imgs"].values[0])
