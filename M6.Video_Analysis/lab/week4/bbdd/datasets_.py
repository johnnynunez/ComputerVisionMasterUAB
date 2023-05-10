import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset


class AICityDataset(Dataset):
    def __init__(self, data_path, seqs, transforms=None, ground_truth_path=None):
        self.transforms = transforms
        self.data_path = data_path
        self.clips = []
        self.gt_data = []
        self.car_class_id = 3
        clip_lengths = [0]

        if isinstance(seqs, list):
            seqs_dict = {seq: sorted(os.listdir(os.path.join(data_path, seq))) for seq in seqs}
            seqs = seqs_dict

        for seq, cam_ids in seqs.items():
            for cam_id in cam_ids:
                cam_dir = os.path.join(data_path, seq, cam_id)
                if os.path.isdir(cam_dir):
                    if ground_truth_path:
                        gt = pd.read_csv(ground_truth_path,
                                         names=['frame', 'id', 'left', 'top', 'width', 'height',
                                                '1', '2', '3', '4'])
                    else:
                        gt = pd.read_csv(os.path.join(cam_dir, 'gt', 'gt.txt'),
                                         names=['frame', 'id', 'left', 'top', 'width', 'height',
                                                '1', '2', '3', '4'])
                    video = cv2.VideoCapture(os.path.join(cam_dir, 'vdo.avi'))
                    self.gt_data.append(gt)
                    self.clips.append(video)
                    clip_lengths.append(len(np.unique(gt["frame"].values)))
        self.clip_starts = np.array(clip_lengths).cumsum()
        self.total_length = sum(clip_lengths)
        self.clip_lengths = clip_lengths

    def __getitem__(self, index):
        clip_idx = (self.clip_starts[1:] >= index).argmax()
        frame_id = self.gt_data[clip_idx].iloc[index - self.clip_starts[clip_idx], 0]

        self.clips[clip_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self.clips[clip_idx].read()
        if self.transforms:
            img = self.transforms(img[:, :, ::-1].copy())

        bboxes = self.gt_data[clip_idx][self.gt_data[clip_idx]['frame'] == frame_id][
            ['left', 'top', 'width', 'height', 'id']].to_numpy()
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        labels = torch.ones((len(bboxes)), dtype=torch.int64) * self.car_class_id
        target = {'boxes': torch.tensor(bboxes[:, :4], dtype=torch.float32),
                  'track_id': torch.tensor(bboxes[:, 4], dtype=torch.float32), 'labels': labels,
                  'frame_id': torch.tensor([frame_id])}
        return img, target


class AICityDatasetDetector(Dataset):
    def __init__(self, data_path, seqs, transforms=None):
        self.transforms = transforms
        self.data_path = data_path
        self.clips = []
        self.gt_data = []
        self.car_class_id = 3
        clip_lengths = [0]

        if isinstance(seqs, list):
            seqs_dict = {seq: sorted(os.listdir(os.path.join(data_path, seq))) for seq in seqs}
            seqs = seqs_dict

        for seq, cam_ids in seqs.items():
            for cam_id in cam_ids:
                cam_dir = os.path.join(data_path, seq, cam_id)
                if os.path.isdir(cam_dir):
                    gt = pd.read_csv(os.path.join(cam_dir, 'gt', 'gt.txt'),
                                     names=['frame', 'id', 'left', 'top', 'width', 'height',
                                            '1', '2', '3', '4'])
                    video = cv2.VideoCapture(os.path.join(cam_dir, 'vdo.avi'))
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.gt_data.append(gt)
                    self.clips.append(video)
                    clip_lengths.append(frame_count)

        self.clip_starts = np.array(clip_lengths).cumsum()
        self.total_length = sum(clip_lengths)
        self.clip_lengths = clip_lengths

    def __getitem__(self, index):
        clip_idx = (self.clip_starts[1:] >= index).argmax()

        self.clips[clip_idx].set(cv2.CAP_PROP_POS_FRAMES, index - self.clip_starts[clip_idx])
        ret, img = self.clips[clip_idx].read()

        if ret:
            if self.transforms:
                img = self.transforms(img[:, :, ::-1].copy())

            frame_id = index - self.clip_starts[:clip_idx].sum()
            if frame_id not in self.gt_data[clip_idx]['frame'].unique():
                return img, {'image_id': torch.tensor([index])}
            else:
                bboxes = self.gt_data[clip_idx][self.gt_data[clip_idx]['frame'] == frame_id][
                    ['left', 'top', 'width', 'height', 'id']].to_numpy()
                bboxes[:, 2] += bboxes[:, 0]
                bboxes[:, 3] += bboxes[:, 1]
                labels = torch.ones((len(bboxes)), dtype=torch.int64) * self.car_class_id
                target = {'boxes': torch.tensor(bboxes[:, :4], dtype=torch.float32),
                          'track_id': torch.tensor(bboxes[:, 4], dtype=torch.float32),
                          'labels': labels, 'image_id': torch.tensor([index])}
                return img, target
        else:
            return torch.ones((24, 24, 3)), None
