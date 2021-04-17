# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
# BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------start here
import pandas as pd
from common.mAP_utils import zfturbo_compute_mAP, normalize_bb_512
from common.detectron2_post_processor_utils import post_process_conf_filter_nms
import numpy as np
from common.kaggle_utils import rescaler

# %% --------------------DIRECTORIES
SPLIT_DIR = f"{BASE_DIR}/2_data_split"

# %% --------------------
# read ground truth csv
holdout_gt_df = pd.read_csv(SPLIT_DIR + "/512/unmerged/10_percent_holdout/holdout_df.csv")
original_image_ids = holdout_gt_df["image_id"].unique()

# %% --------------------
# add 512x512 dimensions in GT
holdout_gt_df["transformed_height"] = 512
holdout_gt_df["transformed_width"] = 512

# %% --------------------
# read the predicted validation csv
holdout_predictions = pd.read_csv(
    f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions/holdout_yolov5.csv")

# %% --------------------Downscale YOLO
# downscale YOLOv5 predictions
holdout_predictions = rescaler(holdout_predictions, holdout_gt_df, "height", "width",
                               "transformed_height",
                               "transformed_width")

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

# %% --------------------CONF + NMS
validation_conf_nms = post_process_conf_filter_nms(holdout_predictions, confidence_filter_thr,
                                                   iou_thr)

# %% --------------------
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

# %% --------------------
validation_conf_nms = validation_conf_nms.append(normal_pred_df_nms)
validation_conf_nms = validation_conf_nms.round(3)

# %% --------------------
# normalize
normalized_preds_nms = normalize_bb_512(validation_conf_nms)
normalized_gt = normalize_bb_512(holdout_gt_df)

# %% --------------------
# compute mAP
print(zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label))
