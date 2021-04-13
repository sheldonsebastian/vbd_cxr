# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
           "Workspace/vbd_cxr/6_environment_files/local.env "

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------imports
import pandas as pd
from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    binary_and_object_detection_processing
from common.kaggle_utils import submission_file_creator, up_scaler

# %% --------------------directories
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
KAGGLE_TEST_DIR = os.getenv("KAGGLE_TEST_DIR")
LOCAL_TEST_DIR = os.getenv("TEST_DIR")
output_directory = DETECTRON2_DIR + "/post_processing_local/submissions"

# %% --------------------
# probability threshold for 2 class classifier
upper_thr = 0.9  # more chance of having disease
lower_thr = 0.1  # less chance of having disease

obj_det_conf_thr = 0.05
iou_threshold = 0.3

# %% --------------------read the predictions
# read binary classifier outputs
binary_prediction = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/final_outputs/test_predictions/test_2_class_ensembled_resnet_152_vgg_19.csv")

# read object detection outputs
object_detection_prediction = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/final_outputs/test_predictions"
    "/test_ensemble_faster_rcnn_yolov5_311.csv")

# %% --------------------
original_dataset = pd.read_csv(KAGGLE_TEST_DIR + "/test_original_dimension.csv")

# get all image ids in original dataset
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------CONFIDENCE + NMS
# will also contain no findings class with 100% probability
nms_predictions = post_process_conf_filter_nms(object_detection_prediction,
                                               obj_det_conf_thr,
                                               iou_threshold)

# %% --------------------
# combine 2 class classifier and object detection predictions
binary_object_nms = binary_and_object_detection_processing(binary_prediction, nms_predictions,
                                                           lower_thr, upper_thr)

# %% --------------------submission prepper
# upscale
upscaled_nms = up_scaler(binary_object_nms, original_dataset)

# %% --------------------
# formatter
formatted_nms = submission_file_creator(upscaled_nms, "x_min", "y_min", "x_max", "y_max",
                                        "label",
                                        "confidence_score")

# %% --------------------
formatted_nms.to_csv(output_directory + "/resnet152_vgg19_fasterrcnn_yolov5_311_iou_03.csv", index=False)
