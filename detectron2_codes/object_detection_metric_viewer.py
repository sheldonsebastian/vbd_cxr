# %% --------------------
import os
import sys

import warnings

warnings.filterwarnings("ignore")

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
from common.kaggle_utils import up_scaler, rescaler

# %% --------------------DIRECTORIES
MERGE_DIR = os.getenv("MERGED_DIR")
DETECRON2_DIR = os.getenv("DETECTRON2_DIR")

# %% --------------------
# read ground truth csv
holdout_gt_df = pd.read_csv(MERGE_DIR + "/512/unmerged/10_percent_holdout/holdout_df.csv")
original_image_ids = holdout_gt_df["image_id"].unique()
upscale_height = "transformed_height"
upscale_width = "transformed_width"

# %% --------------------
# read the predicted validation csv
# DYNAMIC
holdout_predictions = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/detectron2_codes/ensembles/holdout_ensemble_faster_rcnn_retinanet_yolov5.csv")

upscale_gt = False

if upscale_gt:
    # upscale GT
    # add confidence score to GT
    temp_conf = holdout_gt_df.copy(deep=True)
    temp_conf["confidence_score"] = np.ones(shape=(len(holdout_gt_df), 1))

    # upscale BB info for gt
    temp = rescaler(temp_conf, holdout_gt_df, "transformed_height",
                    "transformed_width",
                    "original_height", "original_width",
                    columns=["x_min", "y_min", "x_max", "y_max", "class_id", "confidence_score"])

    holdout_gt_df.loc[:, ["x_min", "x_max", "y_min", "y_max"]] = temp.loc[:,
                                                                 ["x_min", "x_max", "y_min",
                                                                  "y_max"]]
    upscale_height = "original_height"
    upscale_width = "original_width"


# %% --------------------
confidence_filter_thr = 0.05
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
normalized_gt = normalize_bb(holdout_gt_df, holdout_gt_df, upscale_height, upscale_width)
normalized_preds = normalize_bb(holdout_predictions, holdout_gt_df, upscale_height, upscale_width)

# %% --------------------NO POST PROCESSING
print(zfturbo_compute_mAP(normalized_gt, normalized_preds, id_to_label))

# %% --------------------CONF + NMS
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

# normalize
normalized_preds_nms = normalize_bb(validation_conf_nms, holdout_gt_df, upscale_height,
                                    upscale_width)

print(zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label))

# # %% --------------------CONF + WBF
# validation_conf_wbf = post_process_conf_filter_wbf(holdout_predictions, confidence_filter_thr,
#                                                    iou_thr, holdout_gt_df)
#
# # ids which failed confidence
# normal_ids_wbf = np.setdiff1d(original_image_ids,
#                               validation_conf_wbf["image_id"].unique())
# print(f"WBF normal ids count: {len(normal_ids_wbf)}")
# normal_pred_wbf = []
# # add normal ids to dataframe
# for normal_id in set(normal_ids_wbf):
#     normal_pred_wbf.append({
#         "image_id": normal_id,
#         "x_min": 0,
#         "y_min": 0,
#         "x_max": 1,
#         "y_max": 1,
#         "label": 14,
#         "confidence_score": 1
#     })
#
# normal_pred_df_wbf = pd.DataFrame(normal_pred_wbf,
#                                   columns=["image_id", "x_min", "y_min", "x_max", "y_max",
#                                            "label",
#                                            "confidence_score"])
# validation_conf_wbf = validation_conf_wbf.append(normal_pred_df_wbf)
#
# # normalize
# normalized_preds_wbf = normalize_bb(validation_conf_wbf, holdout_gt_df, upscale_height,
#                                     upscale_width)
#
# print(zfturbo_compute_mAP(normalized_gt, normalized_preds_wbf, id_to_label))
