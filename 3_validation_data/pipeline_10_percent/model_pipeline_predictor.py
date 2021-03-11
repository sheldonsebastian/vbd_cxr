# %% --------------------
# pipeline for 2 class classifier and object detection for 10 % holdout data containing F + NF
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
import numpy as np
from common.utilities import merge_bb_nms

# %% --------------------directories
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
MERGED_DIR = os.getenv("MERGED_DIR")

# %% --------------------
confidence_threshold = 0.5
nms_threshold = 0.4

# %% --------------------read the predictions
binary_prediction = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/2_class_classifier/predictions/holdout.csv")

object_detection_prediction = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/object_detection/predictions/holdout_predictions.csv")

# %% --------------------
# adjust object detection classes
object_detection_prediction["label"] -= 1

# %% --------------------
# get all image ids in original dataset
original_dataset = pd.read_csv(MERGED_DIR + "/wbf_merged/holdout_df.csv")

# %% --------------------
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------
abnormal_ids = list(binary_prediction[binary_prediction["target"] == 1]["image_id"].unique())

# %% --------------------
# subset the object detections based on binary classifier
object_detection_prediction_subset = object_detection_prediction[
    object_detection_prediction["image_id"].isin(abnormal_ids)]

# %% --------------------
object_detection_prediction_filtered = object_detection_prediction_subset[
    object_detection_prediction_subset["confidence_score"] >= confidence_threshold]

# %% --------------------
# apply nms on confidence filtered
img_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for image_id in sorted(object_detection_prediction_filtered["image_id"].unique()):
    bb_df = object_detection_prediction_filtered[
        object_detection_prediction_filtered["image_id"] == image_id][
        ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]

    bb_df = bb_df.to_numpy()
    nms_bb = merge_bb_nms(bb_df, 0, 1, 2, 3, 4, iou_thr=nms_threshold, scores_col=5)

    for i in range(len(nms_bb)):
        img_id_arr.append(image_id)
        x_min_arr.append(nms_bb[i][0])
        y_min_arr.append(nms_bb[i][1])
        x_max_arr.append(nms_bb[i][2])
        y_max_arr.append(nms_bb[i][3])
        label_arr.append(nms_bb[i][4])
        score_arr.append(nms_bb[i][5])

# %% --------------------No Findings Class
# normal ids from binary classifier
normal_ids = list(binary_prediction[binary_prediction["target"] == 0]["image_id"].unique())

# ids which do not pass the confidence threshold
failed_confidence_threshold_ids = object_detection_prediction_subset[
    object_detection_prediction_subset["confidence_score"] < confidence_threshold][
    "image_id"].unique()

# ids which failed confidence and are not present in passed confidence threshold
# NOTE:: you can use len(pred_boxes) = 0, logic here too
normal_ids_object_detection = np.setdiff1d(failed_confidence_threshold_ids,
                                           object_detection_prediction_filtered[
                                               "image_id"].unique())

# %% --------------------
normal_ids_final = np.union1d(normal_ids, normal_ids_object_detection)

for normal_id in normal_ids_final:
    img_id_arr.append(normal_id)
    x_min_arr.append(0)
    y_min_arr.append(0)
    x_max_arr.append(1)
    y_max_arr.append(1)
    label_arr.append(14)
    score_arr.append(1)

# %% --------------------
# convert the predictions into pandas dataframe
final_predictions = pd.DataFrame({
    "image_id": img_id_arr,
    "x_min": x_min_arr,
    "y_min": y_min_arr,
    "x_max": x_max_arr,
    "y_max": y_max_arr,
    "label": label_arr,
    "confidence_score": score_arr
})

final_predictions.to_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/predictions/pipeline_prediction.csv",
    index=False)

# %% --------------------
print(len(original_image_ids) == len(final_predictions["image_id"].unique()))
