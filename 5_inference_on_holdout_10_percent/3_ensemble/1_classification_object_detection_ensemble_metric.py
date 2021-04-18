# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------
import numpy as np
import pandas as pd

from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    binary_and_object_detection_processing
from common.mAP_utils import zfturbo_compute_mAP, normalize_bb_512
from common.utilities_object_detection_ensembler import ensemble_object_detectors
from common.kaggle_utils import rescaler

# %% --------------------
# probability threshold for 2 class classifier
upper_thr = 0.95  # more chance of having disease
lower_thr = 0.05  # less chance of having disease

confidence_filter_thr = 0.05
iou_thr = 0.3

# %% --------------------
# 2 class filter prediction
binary_prediction = pd.read_csv(
    f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions/holdout_ensemble_classification.csv")

faster_rcnn = pd.read_csv(
    f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions/holdout_faster_rcnn.csv")

# read yolo v5 output in upscaled format
yolov5 = pd.read_csv(
    f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions/holdout_yolov5.csv")

# %% -------------------- GROUND TRUTH
holdout_gt_df = pd.read_csv(
    f"{BASE_DIR}/2_data_split/512/unmerged/10_percent_holdout/holdout_df.csv")
original_image_ids = holdout_gt_df["image_id"].unique()

# %% --------------------
# add 512x512 dimensions in GT
holdout_gt_df["transformed_height"] = 512
holdout_gt_df["transformed_width"] = 512

# %% --------------------Downscale YOLO
# downscale YOLOv5 predictions
yolov5 = rescaler(yolov5, holdout_gt_df, "height", "width", "transformed_height",
                  "transformed_width")

# %% --------------------Combine outputs of predictors
predictors = [faster_rcnn, yolov5]

# %% --------------------POST PROCESSING
post_processed_predictors = []
# apply post processing individually to each predictor
for holdout_predictions in predictors:
    validation_conf_nms = post_process_conf_filter_nms(holdout_predictions, confidence_filter_thr,
                                                       iou_thr)

    # ids which failed confidence
    normal_ids_nms = np.setdiff1d(original_image_ids,
                                  validation_conf_nms["image_id"].unique())
    print(f"NMS normal ids count: {len(normal_ids_nms)}")
    normal_pred_nms = []
    # add normal ids to dataframe
    for normal_id in set(normal_ids_nms):
        normal_pred_nms.append({
            "image_id": normal_id,
            "x_min": 0,
            "y_min": 0,
            "x_max": 1,
            "y_max": 1,
            "label": 14,
            "confidence_score": 1
        })

    normal_pred_df_nms = pd.DataFrame(normal_pred_nms,
                                      columns=["image_id", "x_min", "y_min", "x_max", "y_max",
                                               "label",
                                               "confidence_score"])
    validation_conf_nms = validation_conf_nms.append(normal_pred_df_nms)

    post_processed_predictors.append(validation_conf_nms)

# %% --------------------MERGE BB for post_processed_predictors
# ensembles the outputs and also adds missing image ids
ensembled_outputs = ensemble_object_detectors(post_processed_predictors, holdout_gt_df,
                                              "transformed_height", "transformed_width", iou_thr,
                                              [3, 9])

# %% --------------------CONF + NMS
validation_conf_nms = post_process_conf_filter_nms(ensembled_outputs, confidence_filter_thr,
                                                   iou_thr)

# ids which failed confidence
normal_ids_nms = np.setdiff1d(original_image_ids,
                              validation_conf_nms["image_id"].unique())
print(f"NMS normal ids count: {len(normal_ids_nms)}")
normal_pred_nms = []
# add normal ids to dataframe
for normal_id in set(normal_ids_nms):
    normal_pred_nms.append({
        "image_id": normal_id,
        "x_min": 0,
        "y_min": 0,
        "x_max": 1,
        "y_max": 1,
        "label": 14,
        "confidence_score": 1
    })

normal_pred_df_nms = pd.DataFrame(normal_pred_nms,
                                  columns=["image_id", "x_min", "y_min", "x_max", "y_max",
                                           "label",
                                           "confidence_score"])
validation_conf_nms = validation_conf_nms.append(normal_pred_df_nms)

# %% --------------------
# combine 2 class classifier and object detection predictions
binary_object_nms = binary_and_object_detection_processing(binary_prediction,
                                                           validation_conf_nms,
                                                           lower_thr, upper_thr)

# %% --------------------
# round to the next 3 digits to avoid normalization errors
binary_object_nms = binary_object_nms.round(3)

# %% --------------------NORMALIZE
normalized_gt = normalize_bb_512(holdout_gt_df)
normalized_preds_nms = normalize_bb_512(binary_object_nms)

# %% --------------------
binary_object_nms.to_csv(
    f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions/holdout_ensemble_classification_object_detection.csv",
    index=False)

# %% --------------------
id_to_label = {
    0: "aortic enlargement",
    1: "atelectasis",
    2: "calcification",
    3: "cardiomegaly",
    4: "consolidation",
    5: "ild",
    6: "infiltration",
    7: "lung opacity",
    8: "nodule/mass",
    9: "other lesion",
    10: "pleural effusion",
    11: "pleural thickening",
    12: "pneumothorax",
    13: "pulmonary fibrosis",
    14: "No Findings class"
}

print(zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label))
