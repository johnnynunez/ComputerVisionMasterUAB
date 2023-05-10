import os
import pickle
from util import write_csv


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


# Replace 'your_folder_path' with the path to the folder containing the .pkl files
folder_path = '../Results/tracking/'
all_data = read_all_pkl_files(folder_path)

for file_name, data in all_data.items():
    print(f"Content of {file_name}:")
    write_csv(data, f"./Results/Task3/S04/{file_name}.csv")