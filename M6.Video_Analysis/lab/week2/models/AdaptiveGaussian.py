import concurrent
import os

import cv2
import numpy as np
from models.BaseModel import BaseModel


class AdaptiveGaussian(BaseModel):

    def __init__(self, video_path, num_frames, p, alpha, colorspace='gray', checkpoint=None):
        super().__init__(video_path, num_frames, colorspace, checkpoint)
        # 2 modes
        self.p = p
        self.mean = None
        self.std = None
        self.alpha = alpha

        self.base = os.path.join(os.getcwd(), "checkpoints", "AdaptativeGaussianModel")

    def compute_parameters(self):
        self.mean = self.images.mean(axis=-1)  # , dtype=np.float64)
        print("Mean computed successfully.")
        self.std = self.images.std(axis=-1)  # , dtype=np.float64)
        print("Standard deviation computed successfully.")

    def compute_next_foreground(self, frame_aux):
        def _set_fg_mask_uint8_row(i, row, output_row):
            output_row[row] = 255

        if not self.modeled:
            print("[ERROR] Background has not been modeled yet.")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_aux)  ##############################
        success, I = self.cap.read()
        if not success:
            return None

        I = cv2.cvtColor(I, self.colorspace_conversion)
        b_mask = (np.abs(I - self.mean) < self.alpha * (self.std + 2))
        self.mean[b_mask] = (self.p * I[b_mask] + (1 - self.p) * self.mean[b_mask])
        img_aux = (I - self.mean)
        self.std[b_mask] = np.sqrt(
            self.p * img_aux[b_mask] * img_aux[b_mask] + (1 - self.p) * (self.std[b_mask] * self.std[b_mask]))

        fg_mask = np.abs(I - self.mean) >= self.alpha * (self.std + 2)
        fg_mask_uint8 = np.zeros_like(fg_mask, dtype=np.uint8)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(fg_mask.shape[0]):
                futures.append(executor.submit(_set_fg_mask_uint8_row, i, fg_mask[i], fg_mask_uint8[i]))

            for future in concurrent.futures.as_completed(futures):
                future.result()

        return fg_mask_uint8, I

    def save_checkpoint(self):
        if not os.path.exists(f"{self.base}/{self.checkpoint}"):
            os.makedirs(f"{self.base}/{self.checkpoint}")

        np.save(f"{self.base}/{self.checkpoint}/mean.npy", self.mean)
        np.save(f"{self.base}/{self.checkpoint}/std.npy", self.std)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/mean.png", self.mean)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/std.png", self.std)

        assert (np.load(f"{self.base}/{self.checkpoint}/mean.npy") == self.mean).all()
        assert (np.load(f"{self.base}/{self.checkpoint}/std.npy") == self.std).all()

    def load_checkpoint(self):
        mean_path = f"{self.base}/{self.checkpoint}/mean.npy"
        std_path = f"{self.base}/{self.checkpoint}/std.npy"
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            return -1
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        print("Checkpoint loaded.")
