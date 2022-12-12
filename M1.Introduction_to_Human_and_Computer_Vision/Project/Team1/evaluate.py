import os
import pickle
from typing import List
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage import io

import image_features
import similarities
from data_utils import DataHandler


def get_index_positions(list_of_elems, element):
    """
    Returns the indexes of all occurrences of the specified element within the list.
    """
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from index_pos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


"""
def get_top_k_vector(similarity_vector: np.ndarray, db_files: List[str], k: int) -> List[str]:
    # Get top K indexes of the vector (unordered)
    idx = np.argpartition(similarity_vector, -k)[-k:]
    # Then we order index in order to get the ordered top k values
    top_k = list(similarity_vector[idx])
    
    # Filter top_k by the number of matches based by threshold
    top_k = [dist for dist in top_k if dist > match_threshold]

    sorted_top = list(sorted(top_k, reverse=True))
    sorted_top = list(dict.fromkeys(sorted_top))
    sorted_index = []

    for j in sorted_top:
        sorted_index.extend(get_index_positions(top_k, j))

    idx = [idx[m] for m in sorted_index]

    return [db_files[i] for i in idx]
"""


def get_top_k_vector(
        similarity_vector: np.ndarray,
        db_files: List[str],
        k: int,
        match_threshold: int = None
) -> List[str]:
    """
    Retrieves the k most similar images to that of the query vector.
    """
    sim_files = list(zip(similarity_vector, db_files))
    sim_files.sort(key=lambda x: x[0], reverse=True)
    if match_threshold is not None:
        sim_files = [x if x[0] > match_threshold else (-1, '') for x in sim_files]
    res = [x[1] for x in sim_files[:k]]
    # print(res[0])
    #  print("res = ", res)
    return res


def get_top_k(similarity_matrix: np.ndarray, db_files: List[str], k: int, mssg: str) -> List[List[str]]:
    """
    Retrieves the k most similar images to those of the queries.
    """
    return [get_top_k_vector(vector, db_files, k) for vector in tqdm(similarity_matrix, desc=mssg)]


def get_top_k_multiple(similarities_multiple: List[List[np.array]], db_files, k: int, match_threshold: int):
    """
    Get top k images for each query image.
    """
    top_k = []
    for similarities_query in similarities_multiple:
        top_k_query = []
        for similarities_pic in similarities_query:
            top_k_query.append(
                get_top_k_vector(
                    similarity_vector=similarities_pic,
                    db_files=db_files,
                    k=k,
                    match_threshold=match_threshold)
            )
        top_k.append(top_k_query)

    return top_k


def apk(expected: List, predicted: List, k: int):
    """
    Compute the average precision @ k.

    Parameters:
    expected : list
        List of expected elements to be predicted (order doesn't matter).
    predicted : list
        A list of predicted elements (order does matter)
    k : int
        The maximum number of predicted elements

    Out:
    score : double
        The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in expected and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not expected:
        return 0.0

    return score / min(len(expected), k)


def ark(expected: List, predicted: List, k: int):
    """
    Compute the average RECALL @ k.

    Parameters:
    expected : list
        List of expected elements to be predicted (order doesn't matter).
    predicted : list
        A list of predicted elements (order does matter)
    k : int
        The maximum number of predicted elements

    Out:
    score : double
        The average RECALL at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in expected and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not expected:
        return 0.0

    return score / min(len(expected), k)


def mapk_multiple(actual: List[List], predicted: List[List], k):
    assert len(actual) == len(predicted), "Actual and predicted lists must have the same length (number of queries)."

    apks = []
    for query_actual, query_predicted in zip(actual, predicted):
        for i in range(len(query_predicted)):
            if i < len(query_actual):
                apks.append(
                    apk(
                        expected=[query_actual[i]],
                        predicted=query_predicted[i],
                        k=k
                    )
                )
            else:
                apks.append(
                    0)  # if there are more pictures to be evaluated than we have predictions for, we assign 0 mapk as a penalty
            # if there are more predictions than actual pictures, we ignore the extra predictions, without penalizing them
    return np.mean(apks)


def mapk(actual: List[List], predicted: List[List], k):
    """
    Compute the mean average precision at k.

    Parameters:
    expected : list[list]
        A list of lists of elements that are to be predicted (order doesn't matter within the lists)
    predicted : list[list]
        A list of lists of predicted elements (order matters within the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mark(actual: List[List], predicted: List[List], k):
    """
    Compute the mean average RECALL at k.

    Parameters:
    expected : list[list]
        A list of lists of elements that are to be predicted (order doesn't matter within the lists)
    predicted : list[list]
        A list of lists of predicted elements (order matters within the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    score : double
            The mean average RECALL at k over the input lists
    """
    return np.mean([ark(a, p, k) for a, p in zip(actual, predicted)])


def get_mapk(similarity_matrix: np.array, db_files: List[str], query_dataset: str, data_handler,
             out_dir: str = "../results/data/w3/"):
    eval_data = []
    for k in (1, 5, 10):
        top_k = get_top_k(
            similarity_matrix=similarity_matrix, db_files=db_files,
            k=k, mssg=f"Retrieving top {k} similar images for {query_dataset}..."
        )

        predicted_results = data_handler.store_outputs_single(
            top_k,
            f"{out_dir}/{query_dataset}.pkl",
            save=True
        )
        expected_results = pickle.load(open(f'../data/{query_dataset}/gt_corresps.pkl', "rb"))

        mapk_val = mapk(expected_results, predicted_results, k=k)

        print("MAP@{} Score: {:.4f}% ({}/{})".format(k, mapk_val * 100, int(len(predicted_results) * mapk_val),
                                                     len(predicted_results)))
        eval_data.append([k, mapk_val * 100, int(len(predicted_results) * mapk_val), len(predicted_results)])
    return eval_data


def metrics_mask(true_labels, pred_labels):
    true_labels = np.asarray(true_labels).astype(bool)
    pred_labels = np.asarray(pred_labels).astype(bool)
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    epsilon = 1e-8
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    f1_measure = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, f1_measure


def evaluate_masks(masks_gen, masks_gt):
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(len(masks_gen)):
        prec, rec, f1 = metrics_mask(masks_gen[i], masks_gt[i])
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    print(
        "Precision: {:.4f}%".format(np.mean(precisions) * 100),
        "Recall: {:.4f}%".format(np.mean(recalls) * 100),
        "F1 Score: {:.4f}%".format(np.mean(f1_scores) * 100), sep='\n'
    )
    return precisions, recalls, f1_scores


def run_experiment_meanBackground(gt_path, input_path, color_space, frame_size):
    def read_mask(path):
        return io.imread(path) // 255

    masks_path = os.path.join(input_path, color_space, str(frame_size))
    mean_precision, mean_recall, mean_f1, n = 0, 0, 0, 0
    for image_filename in sorted(os.listdir(gt_path)):
        if image_filename.endswith('.png'):
            mask_gt, mask_pred = os.path.join(gt_path, image_filename), os.path.join(masks_path, image_filename)
            mask_gt, mask_pred = read_mask(mask_gt), read_mask(mask_pred)
            precision, recall, f1_measue = metrics_mask(mask_gt, mask_pred)

            mean_precision += precision
            mean_recall += recall
            mean_f1 += f1_measue
            n += 1

    print([color_space, frame_size, mean_precision / n, mean_recall / n, mean_f1 / n])
    return [color_space, frame_size, mean_precision / n, mean_recall / n, mean_f1 / n]


# def run_experiments
def run_evaluation_masks_meanBackground(
        gt_path,
        input_path,
        output_csv_path,
        frame_sizes=(2, 4, 10),
        color_spaces=['RGB', 'CieLab', 'YCbCr', 'HSV'],
        N_PROCESS=3,
) -> None:
    eval_data = Parallel(n_jobs=N_PROCESS)(
        delayed(run_experiment_meanBackground)(
            gt_path, input_path, color_space, frame_size
        ) for color_space in color_spaces for frame_size in tqdm(frame_sizes)
    )

    dataframe = pd.DataFrame(eval_data, columns=['Color space', 'Frame size', 'Precision', 'Recall', 'F1-measure'])
    dataframe.to_csv(output_csv_path, index=False)


def run_evaluation_masks_otsu(
        gt_path,
        input_path,
        output_csv_path,
) -> None:
    def read_mask(path):
        return io.imread(path) // 255

    masks_path = input_path
    mean_precision, mean_recall, mean_f1, n = 0, 0, 0, 0
    for image_filename in sorted(os.listdir(gt_path)):
        if image_filename.endswith('.png'):
            mask_gt, mask_pred = os.path.join(gt_path, image_filename), os.path.join(masks_path, image_filename)
            mask_gt, mask_pred = read_mask(mask_gt), read_mask(mask_pred)
            precision, recall, f1_measue = metrics_mask(mask_gt, mask_pred)

            mean_precision += precision
            mean_recall += recall
            mean_f1 += f1_measue
            n += 1

    eval_data = [['Otsu', mean_precision / n, mean_recall / n, mean_f1 / n]]
    print([mean_precision / n, mean_recall / n, mean_f1 / n])
    dataframe = pd.DataFrame(eval_data, columns=['Method', 'Precision', 'Recall', 'F1-measure'])
    dataframe.to_csv(output_csv_path, index=False)


def run_experiment_save_keypoints_descriptors(db_images: np.ndarray, query_images_folder: str, save_path_=None):
    results = []
    data_handler = DataHandler()
    for descriptor_type in ('sift', 'orb', 'akaze'):
        if descriptor_type == 'sift':
            db_keypoints, db_features = image_features.sift_batch(
                images=db_images[:],
                mssg="Computing SIFT from BBDD..."
            )
            query_keypoints, query_features = image_features.sift_multiple(
                files_dir=query_images_folder,
                mssg=f"Computing SIFT from {query_images_folder}..."
            )
        elif descriptor_type == 'orb':
            db_keypoints, db_features = image_features.orb_batch(
                images=db_images[:],
                mssg="Computing ORB from BBDD..."
            )
            query_keypoints, query_features = image_features.orb_multiple(
                files_dir=query_images_folder,
                mssg=f"Computing ORB from {query_images_folder}..."
            )

        elif descriptor_type == 'surf':
            db_keypoints, db_features = image_features.surf_batch(
                images=db_images[:],
                mssg="Computing SURF from BBDD..."
            )
            query_keypoints, query_features = image_features.surf_multiple(
                files_dir=query_images_folder,
                mssg=f"Computing SURF from {query_images_folder}..."
            )
        elif descriptor_type == 'akaze':
            db_keypoints, db_features = image_features.akaze_batch(
                images=db_images[:],
                mssg="Computing AKAZE from BBDD..."
            )
            query_keypoints, query_features = image_features.akaze_multiple(
                files_dir=query_images_folder,
                mssg=f"Computing AKAZE from {query_images_folder}..."
            )
        elif descriptor_type == 'harris_corner':
            db_keypoints, db_features = image_features.harris_corner_batch(
                images=db_images[:],
                mssg="Computing HARRIS from BBDD..."
            )
            query_keypoints, query_features = image_features.harris_corner_multiple(
                files_dir=query_images_folder,
                mssg=f"Computing HARRIS from {query_images_folder}..."
            )
        elif descriptor_type == 'harris_corner_subpix':
            db_keypoints, db_features = image_features.harris_corner_subpixel_batch(
                images=db_images[:],
                mssg="Computing HARRIS SUBPIXEL from BBDD..."
            )
            query_keypoints, query_features = image_features.harris_corner_subpixel_multiple(
                files_dir=query_images_folder,
                mssg=f"Computing HARRIS SUBPIXEL from {query_images_folder}..."
            )

        save_path = os.path.join(save_path_, descriptor_type)
        # save_path = f'./results/keypoints/{descriptor_type}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        db_data = (db_keypoints, db_features)
        query_data = (query_keypoints, query_features)

        data_handler.save_object_pickle(db_data, os.path.join(save_path, 'db.pkl'))
        data_handler.save_object_pickle(query_data, os.path.join(save_path, 'query.pkl'))


def run_all_experiment_keypoints(threshold_list, k_list, query_images_folder: str, db_images: np.ndarray,
                                 keypoints_folder: str, output_csv: str, N_PROCESS=2):
    data_handler = DataHandler()
    for descriptor_type in ('orb', 'sift'):
        db_keypoints, db_features = data_handler.load_object_pickle(
            os.path.join(keypoints_folder, f'{descriptor_type}/db.pkl'))
        query_keypoints, query_features = data_handler.load_object_pickle(
            os.path.join(keypoints_folder, f'{descriptor_type}/query.pkl'))
        results = []
        for method in ('BF', 'FLANN'):
            for measure in ('L1', 'L2', 'Hamming', 'Hamming2'):
                if method == 'FLANN':
                    n_matches = similarities.compute_matches_multiple(
                        features_multiple=query_features,
                        db_features=db_features,
                        method=method,
                        similarity_measure=measure,  # this is ignored if method == "FLANN,
                        N_PROCESS=N_PROCESS
                    )
                else:
                    n_matches = similarities.compute_matches_multiple(
                        features_multiple=query_features,
                        db_features=db_features,
                        method=method,
                        similarity_measure=measure,  # this is ignored if method == "FLANN,
                        N_PROCESS=1
                    )
                for threshold in threshold_list:
                    for k in k_list:
                        top_k_multiple = get_top_k_multiple(
                            similarities_multiple=n_matches,
                            db_files=db_images,
                            k=k,
                            match_threshold=threshold)
                        # predicted_results = data_handler.store_outputs_multiple(top_k_multiple, save=False)
                        predicted_results = data_handler.store_outputs_multiple(
                            outputs=top_k_multiple,
                            save=False)
                        expected_results = pickle.load(open(os.path.join(query_images_folder, 'gt_corresps.pkl'), "rb"))
                        try:
                            mapk = mapk_multiple(expected_results, predicted_results, k=k)
                        except:
                            mapk = -1
                        results.append([descriptor_type, method, measure, threshold, k, mapk])

                        dataframe = pd.DataFrame(results,
                                                 columns=['descriptor_type', 'method', 'measure', 'threshold', 'k',
                                                          'mapk'])
                        dataframe.to_csv(output_csv, mode='a', index=False)
            if method == 'FLANN':
                break


def angular_error(pred, ground):
    if pred > 0:
        pred = 180 - pred
    elif pred < 0:
        pred = -pred
    return min(abs(ground - pred), abs(180 - ground) + pred, abs(180 - pred) + ground)


def mean_angular_error(pred, ground_truth):
    # pred = [[[alpha, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]]]
    # ground_truth = [[[alpha, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]]]
    # Compute the mean angular error between the predicted and ground truth
    # bounding boxes
    list_error = []
    for i in range(len(pred)):
        list_error.append(angular_error(pred[i][0][0], ground_truth[i][0][0]))
    return np.mean(list_error), np.std(list_error), np.median(list_error)


if __name__ == "__main__":
    import argparse

    # python mask_evaluation.py --gt-path='../data/qsd2_w1' --input-path='../data/bg_masks' --output='../results/bg_masks_distribution_method_metrics.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-path', help="The file where the ground truth data is located.",
                        required=True)
    parser.add_argument('--input-path', help="The folder where the folders with the masks are located.",
                        required=True)
    parser.add_argument('--output', help="Output file (csv).")

    args = parser.parse_args()

    run_evaluation_masks_meanBackground(
        gt_path=args.gt_path,
        input_path=args.input_path,
        output_csv_path=args.output,
    )
