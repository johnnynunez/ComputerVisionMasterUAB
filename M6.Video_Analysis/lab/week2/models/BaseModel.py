import cv2
import numpy as np
from tqdm import tqdm


class BaseModel:  # Base class for the models
    def __init__(self, video_path, num_frames, colorspace='gray', checkpoint=None):
        self.cap = cv2.VideoCapture(video_path)
        if self.cap.isOpened():
            print("Video is open")
        else:
            assert "Video is not open"
        self.num_frames = num_frames
        self.modeled = False
        self.checkpoint = checkpoint
        self.channels = 3
        self.images = None

        self.used = {}
        if colorspace == 'gray':
            self.colorspace_conversion = cv2.COLOR_BGR2GRAY
            self.channels = 1
            self.ch_used = [0]

        elif colorspace == 'RGB':
            self.colorspace_conversion = cv2.COLOR_BGR2RGB
            self.channels = 3
            self.ch_used = [0, 1, 2]

        elif colorspace == 'YCRCB':
            self.colorspace_conversion = cv2.COLOR_BGR2YCrCb
            self.channels = 3
            self.ch_used = [1, 2]

        elif colorspace == 'HSV':
            self.colorspace_conversion = cv2.COLOR_BGR2HSV
            self.channels = 3
            self.ch_used = [1]

        elif colorspace == 'YUV':
            self.colorspace_conversion = cv2.COLOR_BGR2YUV
            self.channels = 3
            self.ch_used = [1, 2]


        else:
            raise Exception('Invalid Colorspace')

    def __add_image(self, frame, pos):
        """ if self.images is None:
            self.images = np.zeros((self.num_frames, self.channels, frame.shape[0] * frame.shape[1]))
        else:
            if len(frame.shape) == 2:
                self.images[pos, :, :] = frame.flatten()
            else:
                self.images[pos, :, :] = frame.transpose(2, 0, 1).reshape(self.channels, -1) """

        if self.images is None:
            if self.channels == 1:
                self.images = np.zeros((frame.shape[0], frame.shape[1], self.num_frames))
            else:
                self.images = np.zeros((frame.shape[0], frame.shape[1], self.channels, self.num_frames))
        else:
            if len(frame.shape) == 2:
                self.images[:, :, pos] = frame
            else:
                self.images[:, :, :, pos] = frame

    def save_images(self):
        if self.modeled:
            print("Background has already been modeled.")
            return

        if self.num_frames is None:
            assert "Number of frames is not defined."

        with tqdm(total=self.num_frames) as pbar:
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.num_frames - 1)
            for i in range(self.num_frames):
                success, frame = self.cap.read()
                frame = cv2.cvtColor(frame, self.colorspace_conversion)
                # Save frame to visualize
                # cv2.imwrite(f"./Results/frame_{i}.png", frame)
                if not success:
                    break
                self.__add_image(frame, i)
                pbar.update(1)

        self.modeled = True
        print("Background modeled!")

    def model_background(self):
        if self.checkpoint is not None and self.load_checkpoint() == 1:
            print("Background modeled!")
            # return self.num_frames

        self.save_images()
        self.compute_parameters()
        self.modeled = True
        print("Background modeled!")

        if self.checkpoint is not None:
            self.save_checkpoint()
            print("Checkpoint saved!")

    def save_checkpoint(self):
        raise NotImplementedError("Implemented by a subclass")

    def load_checkpoint(self):
        return False

    def compute_parameters(self):
        raise NotImplementedError("Implemented by a subclass")
