# pipeline for 2 class classifier and object detection for 10 % holdout data containing F + NF
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
from common.mAP_utils import normalize_bb, zfturbo_compute_mAP, get_id_to_label_mAP

# %% --------------------directories
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
MERGED_DIR = os.getenv("MERGED_DIR")

# %% --------------------
confidence_threshold = 0.50
iou_threshold = 0.4

# %% --------------------read the predictions
binary_prediction = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/2_class_classifier/predictions/archives/resnet_50_augmentations/holdout_resnet50_vanilla.csv")

object_detection_prediction = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/object_detection/predictions/faster_rcnn_sgd_anchor_50/holdout_predictions_anchor_sgd_50.csv")

# %% --------------------
# get all image ids in original dataset
original_dataset = pd.read_csv(MERGED_DIR + "/wbf_merged/holdout_df.csv")
original_dataset["class_id"] += 1

# %% --------------------
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------
normal_ids = list(binary_prediction[binary_prediction["target"] == 0]["image_id"].unique())

# %% --------------------
abnormal_ids = list(binary_prediction[binary_prediction["target"] == 1]["image_id"].unique())

# %% --------------------
# subset the object detections based on binary classifier
object_detection_prediction_subset = object_detection_prediction[
    object_detection_prediction["image_id"].isin(abnormal_ids)]

# %% --------------------
id_to_label = get_id_to_label_mAP()

# %% --------------------
# normalize gt
normalized_gt_df = normalize_bb(original_dataset, original_dataset, "transformed_height",
                                "transformed_width")

# %% --------------------CONF + NMS
# post process
nms_final_df = post_process_conf_filter_nms(object_detection_prediction_subset,
                                            confidence_threshold, iou_threshold, normal_ids)

# save as csv output
nms_final_df.to_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/predictions/nms_final_conf_50.csv",
    index=False)

# normalize
nms_normalized = normalize_bb(nms_final_df, original_dataset, "transformed_height",
                              "transformed_width")

# compute mAP
print(zfturbo_compute_mAP(normalized_gt_df, nms_normalized, id_to_label, gt_class_id="class_id",
                          pred_class_id="label"))

# %% --------------------CONF + WBF
# post process
wbf_final_df = post_process_conf_filter_wbf(object_detection_prediction_subset,
                                            confidence_threshold, iou_threshold, original_dataset,
                                            normal_ids)

# save as csv output
wbf_final_df.to_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/predictions/wbf_final_conf_50.csv",
    index=False)

# normalize
wbf_normalized = normalize_bb(wbf_final_df, original_dataset, "transformed_height",
                              "transformed_width")

# compute mAP
print(zfturbo_compute_mAP(normalized_gt_df, wbf_normalized, id_to_label, gt_class_id="class_id",
                          pred_class_id="label"))
