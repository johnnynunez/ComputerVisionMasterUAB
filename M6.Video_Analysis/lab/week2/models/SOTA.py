import cv2

from models.BaseModel import BaseModel


class SOTA(BaseModel):
    def __init__(self, video_path, num_frames, checkpoint=None, method='MOG', colorspace='gray'):
        super().__init__(video_path, num_frames, colorspace, checkpoint)

        if method == 'MOG':
            self.method = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2, backgroundRatio=0.7)
        elif method == 'MOG2':
            self.method = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=True)
        elif method == 'LSBP':
            self.method = cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif method == "knn":
            self.method = cv2.createBackgroundSubtractorKNN()
        elif method == "GMG":
            self.method = cv2.bgsegm.createBackgroundSubtractorGMG()
        else:
            raise Exception('Invalid method')

    def compute_next_foreground(self, frame_aux):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_aux)  ##################################################3

        success, I = self.cap.read()
        if not success:
            return None

        I = cv2.cvtColor(I, self.colorspace_conversion)

        foreground = self.method.apply(I)

        # for MOG2, remove shadows
        foreground[foreground == 127] = 0

        return foreground, I

    def model_background(self):
        frame = self.cap.read()
        counter = 1
        while frame is not None and counter < self.num_frames:
            frame = self.cap.read()
            counter += 1
        print("Background modeled!")
        return counter
