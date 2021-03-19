# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
# env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------imports
import pandas as pd
from common.post_processing_utils import post_process_conf_filter_nms, post_process_conf_filter_wbf
from common.kaggle_utils import up_scaler, submission_file_creator
import numpy as np

# %% --------------------directories
TEST_DIR = os.getenv("TEST_DIR")
KAGGLE_TEST_DIR = os.getenv("KAGGLE_TEST_DIR")

# %% --------------------
confidence_threshold = 0.50
iou_threshold = 0.4

# %% --------------------read the predictions
binary_prediction = pd.read_csv(
    TEST_DIR + "/2_class_classifier/predictions/archives/resnet50_augmentations"
               "/test_2_class_resnet50_vanilla.csv")

object_detection_prediction = pd.read_csv(
    TEST_DIR + "/object_detection/predictions/archives/faster_rcnn_sgd_anchor_50"
               "/test_object_detection_prediction.csv")

# %% --------------------
# get all image ids in original dataset
original_dataset = pd.read_csv(KAGGLE_TEST_DIR + "/test_original_dimension.csv")

# %% --------------------
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------
# normal ids from binary classifier
normal_ids = list(binary_prediction[binary_prediction["target"] == 0]["image_id"].unique())

# abnormal ids
abnormal_ids = list(binary_prediction[binary_prediction["target"] == 1]["image_id"].unique())

# %% --------------------
# subset the object detections based on binary classifier
object_detection_prediction_subset = object_detection_prediction[
    object_detection_prediction["image_id"].isin(abnormal_ids)]

# %% --------------------CONFIDENCE + NMS
nms_predictions = post_process_conf_filter_nms(object_detection_prediction_subset,
                                               confidence_threshold,
                                               iou_threshold, normal_ids)

# %% --------------------CONFIDENCE + WBF
wbf_predictions = post_process_conf_filter_wbf(object_detection_prediction_subset,
                                               confidence_threshold, iou_threshold,
                                               original_dataset, normal_ids)

# %% --------------------
# adjust predictions
nms_predictions["label"] -= 1
wbf_predictions["label"] -= 1

# %% --------------------
# check difference in original v/s predictions and add the missing images as no findings class
nms_missing_ids = np.setdiff1d(original_dataset["image_id"], nms_predictions["image_id"])

for missing_id in nms_missing_ids:
    # nms addon
    nms_predictions = nms_predictions.append(pd.DataFrame({
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
wbf_missing_ids = np.setdiff1d(original_dataset["image_id"], wbf_predictions["image_id"])

for missing_id in wbf_missing_ids:
    # wbf addon
    wbf_predictions = wbf_predictions.append(pd.DataFrame({
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
nms_predictions.to_csv(
    TEST_DIR + "/pipeline_predictions/nms_filtered_conf_50.csv",
    index=False)
wbf_predictions.to_csv(
    TEST_DIR + "/pipeline_predictions/wbf_filtered_conf_50.csv",
    index=False)

# %% --------------------submission prepper
# upscale
upscaled_nms = up_scaler(nms_predictions, original_dataset)
upscaled_wbf = up_scaler(wbf_predictions, original_dataset)

# %% --------------------
# formatter
formatted_nms = submission_file_creator(upscaled_nms, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")
formatted_wbf = submission_file_creator(upscaled_wbf, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")

# %% --------------------
formatted_nms.to_csv(TEST_DIR + "/submissions/nms_pipeline_conf_50.csv", index=False)
formatted_wbf.to_csv(TEST_DIR + "/submissions/wbf_pipeline_conf_50.csv", index=False)
