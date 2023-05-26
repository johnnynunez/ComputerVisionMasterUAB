"""Config defaults.

The configurations are mostly for MOT and MTMC, or an express run that first runs
MOT on all cameras, then MTMC.
Whenever a path is needed an absolute path or a relative path from the root repo is
to be provided.
"""

from pathlib import Path
from yacs.config import CfgNode as CN


C = CN()
C.SYSTEM = CN()

# path to the config directory
C.SYSTEM.CFG_DIR = str(Path(__file__).parent)

# path to the repo's root directory
C.SYSTEM.ROOT_DIR = str(Path(__file__).parent.parent)

# gpu ids to be used (currently 0 or 1 GPUs are supported)
C.SYSTEM.GPU_IDS = [0]

################################################################################
# Global config
################################################################################

# Path to the directory in which all outputs will be saved into
# it makes sense to provide a new directory under output, e.g output/run_1
# (to be overridden)
C.OUTPUT_DIR = None

# do only a debug run (only track the first few frames in MOT, and express runs)
C.DEBUG_RUN = False

# font for text subscriptions (id labels, etc.)
C.FONT = "assets/Hack-Regular.ttf"

# font size for text subscriptions
C.FONTSIZE = 11

################################################################################
# MOT (single camera tracking) config
################################################################################
C.MOT = CN()

# video input for tracking (to be overridden)
# absolute  or relative to the project ROOT_DIR
C.MOT.VIDEO = None

# reid model opts file for loading (to be overridden)
# absolute path or relative to the project ROOT_DIR
C.MOT.REID_MODEL_OPTS = None

# reid model checkpoint file (to be overridden)
# absolute path or relative to the project ROOT_DIR
C.MOT.REID_MODEL_CKPT = None

# use half precision (fp16) instead of fp32 in reid model
C.MOT.REID_FP16 = False

# batch size for running the reid model
C.MOT.REID_BATCHSIZE = 1

# object detector (yolov5s, yolov5m, yolov5l, other yolov5 versions)
C.MOT.DETECTOR = "yolov5l"

# classes that are kept from detection
# only bike, car, motorbike, bus, truck classes are default (yolov5)
C.MOT.TRACKED_CLASSES = [2, 7]

# tracker to use ('deepsort' 'bytetrack_iou')
C.MOT.TRACKER = "deepsort"

# path to the detection mask image showing the ROI (region of interest)
# in the image white pixels are included, while others (black ones) are excluded
C.MOT.DETECTION_MASK = None

# camera calibration file containing
# Homography matrix: a1 a2 a3; b1 b2 b3; c1 c2 c3
# in the first line
C.MOT.CALIBRATION = None

# show video output stream for tracking (cv2 library NEEDED)
C.MOT.SHOW = True

# save online tracked video stream
C.MOT.ONLINE_VIDEO_OUTPUT = False

# save final video output (with tracklet refinement)
C.MOT.VIDEO_OUTPUT = False

# minimum number of bounding boxes per track
C.MOT.MIN_FRAMES = 10

# list of dicts of feature_name:model_path format describing static feature extracting models
# A model can be:
# 1. convolutional neural net determining the feature from the image patch
# 2. fully connected neural net, that gets the reid feature as an input
# e.g: [{"color": "path_to_color_model.pth"}] (one dict per model)
C.MOT.STATIC_ATTRIBUTES = []

# list of dict of feature_name:model_path pairs describing dynamic feature extracting models
# constraints are the same as above at STATIC_ATTRIBUTES
C.MOT.DYNAMIC_ATTRIBUTES = []

# use fp16 in attribute inference
C.MOT.ATTRIBUTE_INFER_FP16 = False

# batch_size for static and dynamic attribute inference
C.MOT.ATTRIBUTE_INFER_BATCHSIZE = 1

# regular expressions describing valid paths of zones for tracks
# e.g: If only tracks that start and end in (zone 1 and 2) or (3 and 4) are good: ["1,.*,2", "3,.*,4"]
# zone strings consist of numbers separated by commas
# Zone 0 is reserved for detections that are not in any zone!
C.MOT.VALID_ZONEPATHS = []

# path to directory containing zone masks (if not provided, no zones will be used)
C.MOT.ZONE_MASK_DIR = None

# run tracklet refinement at the end (post processing), zones are needed for this
C.MOT.REFINE = False

################################################################################
# MTMC (Multi-target multi-camera tracking) config
################################################################################
C.MTMC = CN()

# path to the MTMC camera layout file, or None if it does not exist
C.MTMC.CAMERA_LAYOUT = None

# list of pickled tracklets for each camera (generated by the MOT phase)
C.MTMC.PICKLED_TRACKLETS = []

# linkage to use in MTMC matching phase, the meaning is similar to Agglomerative Clustering
# values: ('average', 'single', 'complete', 'mean_feature')
C.MTMC.LINKAGE = 'average'

# minimum cosine similarity for merging in MTMC phase (the algorithm terminates under this)
C.MTMC.MIN_SIM = 0.5


################################################################################
# Express config to run MOT on all cameras then MTMC automagically
################################################################################
C.EXPRESS = CN()

# camera video streams in order (same order as in camera layout)
# each stream is defined by a dict with the following keys:
#   - 'video': path to the video stream
#   - 'detection_mask': see MOT.DETECTION_MASK (optional)
#   - 'zone_mask_dir': see MOT.ZONE_MASK_DIR (optional)
#   - 'valid_zonepaths': see MOT.VALID_ZONEPATHS (optional)
#   - 'calibration': see MOT.CALIBRATION (optional)
C.EXPRESS.CAMERAS = []

# Save videos with final MTMC-matched ids for each camera
C.EXPRESS.FINAL_VIDEO_OUTPUT = False

################################################################################
# Evaluation config
################################################################################

C.EVAL = CN()

# Ground truth annotations for the videos in the MOTChallenge format
# In the case of MTMC this should contain a path per camera in order,
# in MOT, there is only one path (still in a list)
C.EVAL.GROUND_TRUTHS = []

# Prediction files for each video in the MOTChallenge format
# Only needed if evaluation is run separately, not as a part of MOT or MTMC
C.EVAL.PREDICTIONS = []

# minimal IOU similarity for matching predicted boxes with ground truth boxes
C.EVAL.MIN_IOU = 0.5

# Ignore false positive detections in evaluation. If the ground truth does not contain
# annotation for all frames and all relevant objects, this is needed.
C.EVAL.IGNORE_FP = False

# Drop tracks that only appear on a single camera (only in MTMC if there are more cameras)
C.EVAL.DROP_SINGLE_CAM = False


def get_cfg_defaults():
    """Get a yacs config object with default values."""
    return C.clone()