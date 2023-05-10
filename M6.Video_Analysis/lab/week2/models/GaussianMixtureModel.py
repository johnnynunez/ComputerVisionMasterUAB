import os

import cv2
import joblib
import numpy as np
from models.BaseModel import BaseModel


class GaussianMixtureModel(BaseModel):
    def __init__(self, video_path, num_frames, p, checkpoint=None, n_jobs=-1):
        super().__init__(video_path, num_frames, checkpoint)
        # 2 modes
        self.p = p
        self.mean = None
        self.std = None

        self.channels = 3
        self.base = os.path.join(os.getcwd(), "checkpoints", "AdaptativeGaussianModel")
        self.n_jobs = n_jobs

    def compute_parameters(self):
        self.mean = self.images.mean(axis=-1)  # , dtype=np.float64)
        print("Mean computed successfully.")
        self.std = self.images.std(axis=-1)  # , dtype=np.float64)
        print("Standard deviation computed successfully.")

    def compute_next_foreground(self):
        if not self.modeled:
            print("[ERROR] Background has not been modeled yet.")
            return None

        success, I = self.cap.read()
        if not success:
            return None
        I = cv2.cvtColor(I, self.color_transform)

        # ADAPTIVE STEP HERE
        bm = abs(I - self.mean) * (self.std + 2)  # background mask

        self.mean[bm] = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(lambda x: self.p * x[0] + (1 - self.p) * x[1])(I[bm][i], self.mean[bm][i])
            for i in range(bm.sum())
        )
        aux = I - self.mean  # no need of abs because it is squared
        self.std[bm] = np.sqrt(
            joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(lambda x: self.p * x[0] * x[0] + (1 - self.p) * x[1])(
                    aux[bm][i], self.std[bm][i] * self.std[bm][i]
                )
                for i in range(bm.sum())
            )
        )

        return (abs(I - self.mean) * (self.std + 2)).astype(np.uint8) * 255, I

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
