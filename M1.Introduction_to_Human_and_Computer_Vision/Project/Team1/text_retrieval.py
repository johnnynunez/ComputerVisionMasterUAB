import os
import glob
import pathlib
from tqdm import tqdm
# from inspect import _ParameterKind
from typing import List
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import textdistance
from skimage.io import imread

from detect_textbox import find_text_bounding_box
from image_utils import crop_bbox
from evaluate import mapk


def text_distance(text_gt, text_pred, dist='hamming'):
    if dist == 'hamming':
        distance = textdistance.hamming
    elif dist == 'levenshtein':
        distance = textdistance.levenshtein
    elif dist == 'damerau_levenshtein':
        distance = textdistance.damerau_levenshtein
    # elif dist == "jaro_winkler":
    #     # between [1 --> exact match, 0 --> no match]
    #     distance = textdistance.jaro_winkler
    # elif dist == "strcmp95":
    #     # between [1 --> exact match, 0 --> no match]
    #     distance = textdistance.strcmp95
    else:
        raise ValueError(f'Distance "{dist}" does not exist.')
    
    return distance(text_gt, text_pred)


def get_text_distances(text, text_database, dist='hamming'):
    dists = np.array([np.min(
        [text_distance(text, database_text[0], dist),
         text_distance(text, database_text[1], dist)]) for database_text in text_database])
    return dists


def get_text_distances_multiple(text_query_set, text_database, dist='hamming', save_folder=None):
    all_dists = []
    for text in tqdm(text_query_set, desc=f'Getting distances from Text using {dist.upper()} distance'):
        dists = get_text_distances(text, text_database, dist)
        all_dists.append(dists)
    # all_dists = [ele.tolist() for ele in all_dists]     # Comment this to use np.array

    if save_folder is not None:
        save_file = os.path.join(save_folder, f'dists_text_retrieval.pkl')
        with open(save_file, "wb") as f:
            pickle.dump(all_dists, f)

    return all_dists


def get_top_k_text(text, text_database, k=3, dist='hamming'):
    dists = get_text_distances(text, text_database, dist)
    most_similar = np.argsort(dists)[:k]
    most_similar = [ele.tolist() for ele in most_similar]
    return most_similar


def get_distances_text_multiple(query_set, text_database, dist='hamming', save_folder=None, save_dists=None):
    text_distances = []
    for img in tqdm(query_set, desc=f'Retrieving distances from Text'):
        dists = get_text_distances(img, text_database, dist)
        dists = -(dists - np.amax(dists))
        text_distances.append(dists)
    return text_distances


def get_distances_text_2paintings(files_dir, text_database, dist='hamming', save_folder=None, save_dists=None):
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    text_distances = []
    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        text_distances_query = []
        for im, mask in zip(im_paths, mask_paths):
            im_query = imread(im)
            # dists = get_text_distances(im_query, text_database, dist)
            text = get_text_from_image(im_query, greys=True, binary=False, verbose=False)
            dists = get_text_distances(text, text_database, dist)
            dists = -(dists - np.amax(dists))
            text_distances_query.append(dists)
        text_distances.append(text_distances_query)
    return text_distances


def get_text_from_image(img: np.ndarray, greys=False, binary=False, verbose=False) -> str:
    bbox_pred = find_text_bounding_box(img)
    textbox_img = crop_bbox(img, bbox_pred)

    if greys:
        textbox_img = cv2.cvtColor(textbox_img, cv2.COLOR_RGB2GRAY)
    if binary:
        textbox_img = cv2.cvtColor(textbox_img, cv2.COLOR_RGB2GRAY)
        _, textbox_img = cv2.threshold(textbox_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        parsed_text = pytesseract.image_to_string(textbox_img, lang='cat')
        if parsed_text == '':
            parsed_text = pytesseract.image_to_string(textbox_img, lang='eng')
    except:
        raise Exception('You have a problem with PyTesserac')

    parsed_text = parsed_text.replace('\n', '')

    if verbose:
        start_point, end_point = bbox_pred[0:2], bbox_pred[2:4]
        img_bbox = cv2.rectangle(img.copy(), start_point, end_point, color=(255, 0, 0), thickness=2)

        f, axarr = plt.subplots(1, 2, figsize=(6, 10))
        axarr[0].imshow(img_bbox)
        if greys or binary:
            axarr[1].imshow(textbox_img, cmap='Greys')
        else:
            axarr[1].imshow(textbox_img)
        axarr[1].title.set_text(parsed_text)
        plt.show()

    return parsed_text


def get_top_k_text_multiple(text_query_set, text_database, k=3, dist='hamming', save_folder=None):
    top_k = []
    for text in tqdm(text_query_set, desc=f'Retrieving top {k} results from Text using {dist.upper()} distance'):
        top_k_text = get_top_k_text(text, text_database, k, dist)
        top_k.append(top_k_text)

    if save_folder is not None:
        save_file = os.path.join(save_folder, f'top_{k}_text_retrieval.pkl')
        with open(save_file, "wb") as f:
            pickle.dump(top_k, f)

    return top_k


def plot_image_and_similar_by_text(db, db_text, qs, qs_text, top_k_qs, examples=7) -> None:
    k = len(top_k_qs[0])
    for i in range(min(examples, len(qs))):
        f, axarr = plt.subplots(1, k+1, figsize=(15, 4*k))
        axarr[0].imshow(qs[i])
        axarr[0].title.set_text(qs_text[i])
        for j in range(k):
            axarr[j+1].imshow(db[top_k_qs[i][j]])
            axarr[j+1].title.set_text(db_text[top_k_qs[i][j]])
        plt.show()


def run_and_evaluate_text_retrieval(text_query_set, save_folder, text_database, gt_corresps_folder, output_file, verbose=False):
    eval_data = []
    for dist in ['hamming', 'levenshtein', 'damerau_levenshtein']:
        for k in (1, 3, 5):
            top_k = get_top_k_text_multiple(
                text_query_set = text_query_set,
                save_folder = save_folder,
                text_database = text_database,
                k = k,
                dist = dist
                )

            expected_results_query_set = pickle.load(open(os.path.join(gt_corresps_folder, 'gt_corresps.pkl'), "rb"))
            mapk_value = mapk(expected_results_query_set, top_k, k)
            correct, total = int(len(top_k)*mapk_value), len(top_k)
            eval_data.append([save_folder, gt_corresps_folder, dist, k, mapk_value, correct, total])

            if verbose:
                print("MAP @ {} Score: {:.4f}% ({}/{})\n".format(k, mapk_value*100, correct, total))

    dataframe = pd.DataFrame(eval_data, columns=['text_query_set', 'gt_corresps_folder', 'dist', 'k', 'mapk_value', 'correct', 'total'])
    dataframe.to_csv(output_file, index=False)


def evaluate_get_text(gt_text_folder, pred_text_folder, dist='hamming', verbose=False):
    dists = []
    for gt_textfile in tqdm(sorted(os.listdir(gt_text_folder))):
        if gt_textfile.endswith('.txt'):
            gt_text_path = os.path.join(gt_text_folder, gt_textfile)
            with open(gt_text_path, encoding="latin-1") as f:
                gt_text = f.read()

            pred_text_path = os.path.join(pred_text_folder, gt_textfile)
            with open(pred_text_path, encoding="latin-1") as f:
                pred_text = f.read()

            dists.append(text_distance(gt_text, pred_text, dist))

    dist_mean, dist_median = np.mean(dists), np.median(dists)
    if verbose:
        print(f'{dist.upper()} distances mean: {dist_mean}  median {dist_median}')
    return dist_mean, dist_median


def evaluate_all_get_text(gt_text_folder, pred_text_folders: List, output_file):
    eval_data = []
    for (greys, binary), pred_folder in tqdm(zip([(False, False), (True, False), (False, True)], pred_text_folders)):
        for dist in ['hamming', 'levenshtein', 'damerau_levenshtein']:
            dist_mean, dist_median = evaluate_get_text(gt_text_folder, pred_folder, dist)
            eval_data.append([pred_folder, greys, binary, dist, dist_mean, dist_median])

    dataframe = pd.DataFrame(eval_data, columns=['pred_folder', 'greys', 'binary', 'dist', 'dist_mean', 'dist_median'])
    dataframe.to_csv(output_file, index=False)


def generate_text_files_from_folder(images_folder, text_folder, greys=False, binary=False):
    for image_filename in tqdm(sorted(os.listdir(images_folder))):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(images_folder, image_filename)
            image = cv2.imread(image_path)

            pred_text = get_text_from_image(image, greys, binary)

            with open(os.path.join(text_folder, image_filename.replace('.jpg', '.txt')), 'w') as file:
                file.write(pred_text)


def generate_text_files(img_set, img_set_files, text_folder, greys=False, binary=False):
    for img, img_file in tqdm(zip(img_set, img_set_files)):
        pred_text = get_text_from_image(img, greys, binary)
        with open(os.path.join(text_folder, os.path.basename(img_file).replace('.jpg', '.txt')), 'w') as file:
            file.write(pred_text)



def generate_text_files_multiple(files_dir, text_folder, greys=False, binary=False):
    dir_names = [dir_name for dir_name in os.listdir(files_dir)]
    dir_names.sort()

    for dir_name in dir_names:
        im_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.jpg")]
        mask_paths = [im_path for im_path in glob.glob(f"{os.path.join(files_dir, dir_name)}/*.png")]
        im_paths.sort()
        mask_paths.sort()

        for i, (im, mask) in enumerate(zip(im_paths, mask_paths)):
            print(im)
            im_query = imread(im)
            pred_text = get_text_from_image(im_query, greys, binary)

            query_name = pathlib.PurePath(im)
            query_name = query_name.parent.name
            filename = os.path.join(text_folder, query_name) + '.txt'
            write_option = 'w' if i == 0 else 'a'
            with open(os.path.join(filename), write_option) as file:
                if i > 0:
                    file.write("\n")
                file.write(pred_text)