import itertools
import pandas as pd
import os
import pickle
from collections import defaultdict

def load_from_txt(path):
    """
    :param path: path file

    :return: list = [[class,x1, y1, x2, y2, conf]]
    """
    detections = []
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        frame = int(ll[0]) 
        detections.append(
            [
                frame,
                float(ll[1]),
                float(ll[2]),
                float(ll[3]),
                float(ll[2]) + float(ll[4]),
                float(ll[3]) + float(ll[5]),
                float(ll[6])
            ]
        )

    """Group the detected boxes by frame_id as a dictionary"""
    detections.sort(key=lambda x: x[0])
    detections = itertools.groupby(detections, key=lambda x: x[0])
    final_dict = {}
    for k,v in detections:
        det = []
        for vv in v:
            det.append(list(vv)[1:])
        final_dict[k] = det

    return final_dict


def load_from_txt_video(path):
    """
    :param path: path file

    :return: dict[frame] = [[t_id,x1, y1, x2, y2]]
    """
    detections = defaultdict(list)
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(",")
        frame = int(ll[0]) 
        detections[frame].append(
            [
                float(ll[1]),
                float(ll[2]),
                float(ll[3]),
                float(ll[2]) + float(ll[4]),
                float(ll[3]) + float(ll[5]),
            ]
        )

    return detections


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_all_pkl_files(folder_path):
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    all_data = {}
    for file_name in pkl_files:
        file_path = os.path.join(folder_path, file_name)
        data = load_pkl_file(file_path)
        all_data[file_name] = data

    return all_data

def convert_pkl_to_txt(pkl_folder, txt_folder):
    all_data = read_all_pkl_files(pkl_folder)

    for file_name, data in all_data.items():
        fname = file_name.split('.')[0]
        f = open(pkl_folder+'/'+fname+'.txt','w')
        print(f"Writing content of {fname}: in txt format for TrackEval")
        for frame,dets in data.items():
            for det in dets:
                w = det[3] - det[1]
                h = det[4] - det[2]
                f.write(f'{frame},{det[-1]},{det[1]},{det[2]},{w},{h},{det[-2]},-1,-1,-1 \n')

        f.close()