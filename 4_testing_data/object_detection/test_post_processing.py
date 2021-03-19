# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "d:/gwu/4 spring 2021/6501 capstone/vbd cxr/pycharm " \
           "workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
# env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add home dir to pythonpath
sys.path.append(os.getenv("home_dir"))

# %% --------------------imports
from common.kaggle_utils import up_scaler, submission_file_creator
import pandas as pd
from common.post_processing_utils import post_process_conf_filter_nms, post_process_conf_filter_wbf
import numpy as np

# %% --------------------directories
TEST_DIR = os.getenv("TEST_DIR")
KAGGLE_TEST_DIR = os.getenv("KAGGLE_TEST_DIR")

# %% --------------------
# read the predicted test csv
test_predictions = pd.read_csv(
    TEST_DIR + "/object_detection/predictions/archives/faster_rcnn_sgd_anchor_50"
               "/test_object_detection_prediction.csv")

# %% --------------------
# original dimensions
original_dimension = pd.read_csv(KAGGLE_TEST_DIR + "/test_original_dimension.csv")

# %% --------------------
confidence_threshold = 0.50
iou_threshold = 0.4

# %% --------------------CONFIDENCE INTERVAL + NMS
nms_filtered = post_process_conf_filter_nms(test_predictions, confidence_threshold, iou_threshold)

# %% --------------------CONFIDENCE INTERVAL + WBF
wbf_filtered = post_process_conf_filter_wbf(test_predictions, confidence_threshold, iou_threshold,
                                            original_dimension)

# %% --------------------
# adjust labels
nms_filtered["label"] -= 1
wbf_filtered["label"] -= 1

# %% --------------------
# check difference in original v/s predictions and add the missing images as no findings class
nms_missing_ids = np.setdiff1d(original_dimension["image_id"], nms_filtered["image_id"])

for missing_id in nms_missing_ids:
    # nms addon
    nms_filtered = nms_filtered.append(pd.DataFrame({
        "image_id": [missing_id],
        "x_min": [0],
        "y_min": [0],
        "x_max": [1],
        "y_max": [1],
        # class 14 is no findings class
        "label": [14],
        "confidence_score": [1]
    }), ignore_index=True)

# %% --------------------
# check difference in original v/s predictions and add the missing images as no findings class
wbf_missing_ids = np.setdiff1d(original_dimension["image_id"], wbf_filtered["image_id"])

for missing_id in wbf_missing_ids:
    # wbf addon
    wbf_filtered = wbf_filtered.append(pd.DataFrame({
        "image_id": [missing_id],
        "x_min": [0],
        "y_min": [0],
        "x_max": [1],
        "y_max": [1],
        # class 14 is no findings class
        "label": [14],
        "confidence_score": [1]
    }), ignore_index=True)

# %% --------------------
nms_filtered.to_csv(TEST_DIR + "/object_detection/object_detection_post_processed/nms_filtered_conf_50.csv",
                    index=False)
wbf_filtered.to_csv(TEST_DIR + "/object_detection/object_detection_post_processed/wbf_filtered_conf_50.csv",
                    index=False)

# %% --------------------submission prepper
# upscale
upscaled_nms = up_scaler(nms_filtered, original_dimension)
upscaled_wbf = up_scaler(wbf_filtered, original_dimension)

# %% --------------------
# formatter
formatted_nms = submission_file_creator(upscaled_nms, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")
formatted_wbf = submission_file_creator(upscaled_wbf, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")
# %% --------------------
formatted_nms.to_csv(TEST_DIR + "/submissions/nms_object_detection_conf_50.csv", index=False)
formatted_wbf.to_csv(TEST_DIR + "/submissions/wbf_object_detection_conf_50.csv", index=False)
