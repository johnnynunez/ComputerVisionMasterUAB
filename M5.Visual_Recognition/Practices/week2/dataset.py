import os

import PIL.Image as Image
import numpy as np
import pandas as pd

path = "/export/home/group03/mcv/datasets/KITTI-MOTS/instances/"

df = pd.DataFrame(
    columns=['video_id', 'frame_id', 'class_id', 'obj_instance_id', 'path_frame_annotation', 'path_frame_image']
)

for video_id in os.listdir(path):
    video_path = os.path.join(path, video_id)

    if os.path.isdir(video_path):
        for frame_id in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame_id)

            if frame_path[-4:] == '.png':
                img = np.array(Image.open(frame_path))
                obj_ids = np.unique(img)

                for obj_id in obj_ids:
                    class_id = obj_id // 1000
                    obj_instance_id = obj_id % 1000

                    frame_path_image = frame_path.replace('instances', 'training/image_02')

                    new_values = {
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'class_id': class_id,
                        'obj_instance_id': obj_instance_id,
                        'path_frame_annotation': frame_path,
                        'path_frame_image': frame_path_image,
                    }
                    new_row = pd.DataFrame.from_records(new_values, index=[0])
                    df = pd.concat([df, new_row])

df.to_csv('kitti-mots_annotations.csv')
