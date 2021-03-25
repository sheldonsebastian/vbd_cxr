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

# %% --------------------start here
import pandas as pd
from common.mAP_utils import zfturbo_compute_mAP, normalize_bb
from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    post_process_conf_filter_wbf
import numpy as np

# %% --------------------DIRECTORIES
MERGE_DIR = os.getenv("MERGED_DIR")
DETECRON2_DIR = os.getenv("DETECTRON2_DIR")

# %% --------------------
# read ground truth csv
# TODO try with NMS unification logic also
holdout_gt_df = pd.read_csv(MERGE_DIR + "/512/nms_merged/10_percent_holdout/holdout_df_0_6.csv")
original_image_ids = holdout_gt_df["image_id"].unique()

# %% --------------------
# read the predicted validation csv
holdout_predictions = pd.read_csv(
    DETECRON2_DIR + "/faster_rcnn/holdout/unmerged_external/holdout.csv")

# %% --------------------
confidence_filter_thr = 0.10
iou_thr = 0.4
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

# %% --------------------NORMALIZE
normalized_gt = normalize_bb(holdout_gt_df, holdout_gt_df, "transformed_height",
                             "transformed_width")
normalized_preds = normalize_bb(holdout_predictions, holdout_gt_df, "transformed_height",
                                "transformed_width")

# %% --------------------NO POST PROCESSING
print(zfturbo_compute_mAP(normalized_gt, normalized_preds, id_to_label))

# %% --------------------CONF + NMS
validation_conf_nms = post_process_conf_filter_nms(holdout_predictions, confidence_filter_thr,
                                                   iou_thr)

# ids which failed confidence
normal_ids_nms = np.setdiff1d(original_image_ids,
                              validation_conf_nms["image_id"].unique())
# TODO verify this logic
normal_pred_nms = []
# add normal ids to dataframe
for normal_id in set(normal_ids_nms):
    normal_pred_nms.append({
        "image_id": normal_id,
        "x_min": 0,
        "y_min": 0,
        "x_max": 1,
        "y_max": 0,
        "label": 14,
        "confidence_score": 1
    })

normal_pred_df_nms = pd.DataFrame(normal_pred_nms,
                                  columns=["image_id", "x_min", " y_min", " x_max", " y_max",
                                           " label",
                                           " confidence_score"])
validation_conf_nms = validation_conf_nms.append(normal_pred_df_nms, axis=1)

# normalize
normalized_preds_nms = normalize_bb(validation_conf_nms, holdout_gt_df, "transformed_height",
                                    "transformed_width")

print(zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label))

# %% --------------------CONF + WBF
validation_conf_wbf = post_process_conf_filter_wbf(holdout_predictions, confidence_filter_thr,
                                                   iou_thr, holdout_gt_df)

# TODO verify this logic
# ids which failed confidence
normal_ids_wbf = np.setdiff1d(original_image_ids,
                              validation_conf_wbf["image_id"].unique())

normal_pred_wbf = []
# add normal ids to dataframe
for normal_id in set(normal_ids_wbf):
    normal_pred_wbf.append({
        "image_id": normal_id,
        "x_min": 0,
        "y_min": 0,
        "x_max": 1,
        "y_max": 0,
        "label": 14,
        "confidence_score": 1
    })

normal_pred_df_wbf = pd.DataFrame(normal_pred_wbf,
                                  columns=["image_id", "x_min", " y_min", " x_max", " y_max",
                                           " label",
                                           " confidence_score"])
validation_conf_wbf = validation_conf_wbf.append(normal_pred_df_wbf, axis=1)

# normalize
normalized_preds_wbf = normalize_bb(validation_conf_wbf, holdout_gt_df, "transformed_height",
                                    "transformed_width")

print(zfturbo_compute_mAP(normalized_gt, normalized_preds_wbf, id_to_label))
