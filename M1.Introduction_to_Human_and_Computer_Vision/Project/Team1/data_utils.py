import os
import pickle
import re
from typing import Tuple, List

import numpy as np
from joblib import Parallel, delayed
from skimage.io import imread
from tqdm import tqdm

"""
DataHandler Class: Contains operations related to I/O.
"""


class DataHandler():
    def __init__(self, n_process: int = 2) -> None:
        self.n_process = n_process
        print("Initialized DataHandler with {} processes".format(self.n_process))

    def store_outputs_single(self, outputs: List[List[str]], path: str, save: bool) -> List[List[int]]:
        """
        Stores outputs for a single image query as a pickle object.
        """
        # Function vectorization, apply to each element of the list
        get_image_id_vect = np.vectorize(self.get_image_id)
        results = get_image_id_vect(outputs)

        if not os.path.exists(path):
            os.makedirs(path)

        if save:
            # pickle data
            pickle.dump(obj=results, file=open(path + "/result.pkl", "wb"))
            print("Results saved at {}".format(path + "/result.pkl"))

        return results

    def store_outputs_multiple(self, outputs: List[List[str]], save: bool, name_file=None, path: str='') -> List[List[int]]:
        """
        Stores outputs as a pickle object for a single image query that may be associated with up to 2 pictures.
        """
        get_image_id_vect = np.vectorize(self.get_image_id)

        results = []
        for query_outputs in outputs:
            try:
                results_query = get_image_id_vect(query_outputs).tolist()
                results.append(results_query)
            except:
                results.append([[-1]])

        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            # pickle data
            if name_file:
                pickle.dump(obj=results, file=open(path + "/result_{}.pkl".format(name_file), "wb"))
                print("Results saved at {}".format(path + "/result_{}.pkl".format(name_file)))
            else:
                pickle.dump(obj=results, file=open(path + "/result.pkl", "wb"))
                print("Results saved at {}".format(path + "/result.pkl"))

        return results

    def format_multi_image(self, outputs: List[int], outputs_files: List[str]) -> List[List[int]]:
        """
        Formats outputs for multi-image query.
        """
        final_outputs = []
        partial_outputs = []
        last_file = outputs_files[0]

        for i in range(len(outputs)):
            # Add Results from same image (multi-image case)
            if last_file == outputs_files[i]:
                partial_outputs.append(outputs[i][0])

            # image is different so we create another list and fix the current one
            else:
                final_outputs.append(partial_outputs)
                partial_outputs = []
                partial_outputs.append(outputs[i][0])

            last_file = outputs_files[i]

        final_outputs.append(partial_outputs)

        return final_outputs

    def load_image(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads image at path.
        Returns both the image and path.
        """
        return path, imread(path)

    def load_images(self, folder: str, extension: str, desc: str) -> Tuple[np.ndarray, List[str]]:
        """
        Loads all images at folder with a certain extension.
        Returns images and paths.
        """
        # List files and read data
        files = [
            folder + image for image in os.listdir(folder) if extension in image]
        data = Parallel(n_jobs=self.n_process)(
            delayed(self.load_image)(file) for file in tqdm(files, desc=desc))
        data = sorted(data)  # Sorted by path (0000.png, 0001.png, ...)
        print('{} read: {} images'.format(folder, len(data)))

        # Split Images' names and Paths
        images_names, images = list(zip(*data))

        return images, images_names
    
    def load_masks(self, folder: str, extension: str, desc: str) -> Tuple[np.ndarray, List[str]]:
        """
        Loads all masks at folder with a certain extension.
        Returns masks and paths.
        """
        # List files and read data
        files = [
            folder + image for image in os.listdir(folder) if extension in image]
        data = Parallel(n_jobs=self.n_process)(
            delayed(self.load_image)(file) for file in tqdm(files, desc=desc))
        data = sorted(data)
        print('{} read: {} images'.format(folder, len(data)))
        mask_names, masks = list(zip(*data))
        return masks, mask_names

    def read_text(self, text_path: str) -> str:
        """
        Reads text at path.
        Returns both the text and path.
        """
        with open(text_path, encoding="latin-1") as f:
            return text_path, f.read()

    def load_text(self, folder: str, extension: str, desc: str, tuples=True) -> Tuple[str, str]:
        """
        Reads all text files at the specified folder.
        """
        # List files and read data
        files = [folder + image for image in os.listdir(folder) if extension in image]
        data = Parallel(n_jobs=self.n_process)(
            delayed(self.read_text)(file) for file in tqdm(files, desc=desc))
        
        data = sorted(data)  # Sort by path (0000.png, 0001.png, ...)
        print('{} read: {} images'.format(folder, len(data)))

        # Split Images' names and Paths
        _, text = list(zip(*data))
        
        if tuples:
            try:
                text = [eval("('', '')") if t == '\n' else eval(t.replace('\n', '')) for t in text]
            except:
                print("Error while reading text files. Please check that the text files are in the correct format.")
                print("[t for t in text] = {}".format([t for t in text]))
        return text

    def load_text_multiple(self, folder: str, extension: str, desc: str) -> Tuple[str, str]:
        """
        Reads all text files at the specified folder.
        """
        # List files and read data
        files = [
            folder + image for image in os.listdir(folder) if extension in image]
        data = Parallel(n_jobs=self.n_process)(
            delayed(self.read_text)(file) for file in tqdm(files, desc=desc))
        data = sorted(data)

        _, text = list(zip(*data))

        res_text = []
        # Clean and eval str of list of tuples
        for tup in text:
            # split_tup = tup.split("\n")
            list_tup = tup.split("\n")
            res_tup = []
            for i in range(len(list_tup)-1):
                res_tup.append(eval(list_tup[i]))
            res_text.append(res_tup)

        return res_text

    def remove_special(self, sentence: str, index: int) -> str:
        """
        Removes special characters froma a string
        """
        title = sentence.split(',')[index].replace(
            ')', '').replace("'", "").replace('\n', '')
        return re.sub("[^A-Za-z0-9- ]", "", title)

    def extract_author_title(self, data: list, desc: str, index: int) -> list:
        """
        Extracts author and title from a list of strings, and removes special characters.
        """
        titles = []
        for sentence in tqdm(data, desc=desc):
            if sentence == '\n':
                titles.append('')
            else:
                title = self.remove_special(sentence, index)
                titles.append(title)
        return titles

    def get_image_id(self, image: str) -> str:
        """
        Extracts image ID from image relative path.
        """
        try:
            # Extract BBBD_XYZ.jpg from relative path
            file = os.path.basename(image)
            # Extract XYZ id from BBBD_XYZ.jpg
            id = file.replace(".jpg", "").split("_")[1]
            return int(id)
        except:
            return -1

    def save_object_pickle(self, obj: object, path: str) -> None:
        """
        Saves an object as a pickle file.
        """
        import copyreg
        import cv2
        def _pickle_keypoints(point):
            return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                                point.response, point.octave, point.class_id)

        copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

        pickle.dump(obj, open(path, "wb"))
        print("Object saved at {}".format(path))

    def load_object_pickle(self, path: str) -> object:
        """
        Loads an object from a pickle file.
        """
        return pickle.load(open(path, "rb"))
