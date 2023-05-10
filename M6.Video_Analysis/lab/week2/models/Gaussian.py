import os

import cv2
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from models.BaseModel import BaseModel


class Gaussian(BaseModel):
    def __init__(self, video_path, num_frames, alpha, colorspace='gray', checkpoint=None):
        super().__init__(video_path, num_frames, colorspace, checkpoint)
        # 2 modes
        self.mean = None
        self.std = None
        self.alpha = alpha
        self.colorspace = colorspace

        self.base = os.path.join(os.getcwd(), "checkpoints", "GaussianModel")

    def compute_parameters(self):
        # Compute the mean and std of the images using self.ch_used channels [0,1,2] means that we keep the 3 channels, [0] means that we keep only the first channel
        if self.colorspace == 'gray':
            self.mean = np.mean(self.images, axis=-1, dtype=np.float32)
            print("Mean computed successfully.")
            self.std = np.std(self.images, axis=-1, dtype=np.float32)
            print("Standard deviation computed successfully.")
        else:
            self.mean = np.mean(self.images[:, :, self.ch_used, :], axis=-1, dtype=np.float32)
            print("Mean computed successfully.")
            self.std = np.std(self.images[:, :, self.ch_used, :], axis=-1, dtype=np.float32)
            print("Standard deviation computed successfully.")

        if self.colorspace == 'gray':
            # cv2.imwrite("./results/Gaussian/mean.png", self.mean)
            # cv2.imwrite("./results/Gaussian/std.png", self.std)
            # Plot the heatmap of the standard deviation without showing it
            plt.imshow(self.std, cmap='hot')
            plt.colorbar()
            plt.savefig("./results/task_1_Gaussian/std_heatmap.png")

    """ NOT TESTED YET
    @staticmethod
    @njit
    def compute_mean_std(images):
        mean = np.mean(images, axis=-1)
        std = np.std(images, axis=-1)
        return mean, std

    def compute_parameters(self):
        self.mean, self.std = self.compute_mean_std(self.images)
        print("Mean and standard deviation computed successfully.")
    """

    def compute_next_foreground(self, frame_aux):
        if not self.modeled:
            print("[ERROR] Background has not been modeled yet.")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_aux)  ##################################################3

        success, I = self.cap.read()
        if not success:
            return None

        I = cv2.cvtColor(I, self.colorspace_conversion)
        # Only keep the channels self.ch_used = [0,1,2] means that we keep the 3 channels
        if self.colorspace == 'gray':
            abs_diff = np.abs(I - self.mean)
        else:
            abs_diff = np.abs(I[:, :, self.ch_used] - self.mean)

        foreground = ne.evaluate("abs_diff >= alpha * (std + 2)",
                                 local_dict={"abs_diff": abs_diff, "std": self.std, "alpha": self.alpha})
        return foreground.astype(np.uint8) * 255, I

    def save_checkpoint(self):
        if not os.path.exists(f"{self.base}/{self.checkpoint}"):
            os.makedirs(f"{self.base}/{self.checkpoint}")

        np.save(f"{self.base}/{self.checkpoint}/mean.npy", self.mean)
        np.save(f"{self.base}/{self.checkpoint}/std.npy", self.std)
        # cv2.imwrite(f"{self.base}/{self.checkpoint}/mean.png", self.mean)
        # cv2.imwrite(f"{self.base}/{self.checkpoint}/std.png", self.std)

        assert (np.load(f"{self.base}/{self.checkpoint}/mean.npy") == self.mean).all()
        assert (np.load(f"{self.base}/{self.checkpoint}/std.npy") == self.std).all()

    def load_checkpoint(self):
        mean_path = f"{self.base}/{self.checkpoint}/mean.npy"
        std_path = f"{self.base}/{self.checkpoint}/std.npy"
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            print("[WARNING] Checkpoint does not exist.")
            return -1
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        print("Checkpoint loaded.")

    def compute_parameters_image(self, image):
        self.mean = np.mean(self.image, axis=-1, dtype=np.float32)
        print("Mean computed successfully.")
        self.std = np.std(self.image, axis=-1, dtype=np.float32)
