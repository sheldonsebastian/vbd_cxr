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
from common.mAP_utils import zfturbo_compute_mAP, normalize_bb, get_id_to_label_mAP
from common.post_processing_utils import post_process_conf_filter_nms, post_process_conf_filter_wbf

# %% --------------------DIRECTORIES
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
MERGE_DIR = os.getenv("MERGED_DIR")

# %% --------------------
# read ground truth csv
val_gt_df = pd.read_csv(
    MERGE_DIR + "/wbf_merged/object_detection/val_df_20.csv")
val_gt_df["class_id"] += 1

# %% --------------------
# read the predicted validation csv
val_predictions = pd.read_csv(
    VALIDATION_PREDICTION_DIR + f"/object_detection/predictions/validation_predictions.csv")

# %% --------------------
# read ground truth csv
holdout_gt_df = pd.read_csv(
    MERGE_DIR + "/wbf_merged/object_detection/holdout_df.csv")
holdout_gt_df["class_id"] += 1

# %% --------------------
# read the predicted validation csv
holdout_predictions = pd.read_csv(
    VALIDATION_PREDICTION_DIR + f"/object_detection/predictions/holdout_predictions.csv")

# %% --------------------
confidence_filter_thr = 0.5
iou_thr = 0.4
id_to_label = get_id_to_label_mAP()

# %% --------------------
for gt_values, pred_values, title in zip([val_gt_df, holdout_gt_df],
                                         [val_predictions, holdout_predictions],
                                         ["Validation Data", "Holdout 5% Data"]):
    print("-" * 10 + title + "-" * 10)
    # %% --------------------
    # merge with validation predicted image ids
    validation_predictions_image_ids = pred_values["image_id"].unique()

    # https://stackoverflow.com/questions/19960077/how-to-filter-pandas-dataframe-using-in-and-not-in-like-in-sql
    gt_df = gt_values[gt_values["image_id"].isin(validation_predictions_image_ids)]
    gt_df = gt_df.reset_index(drop=True)

    # %% --------------------NORMALIZE
    normalized_gt = normalize_bb(gt_df, gt_df, "transformed_height", "transformed_width")
    normalized_preds = normalize_bb(pred_values, gt_df, "transformed_height",
                                    "transformed_width")

    # %% --------------------RAW
    print(zfturbo_compute_mAP(normalized_gt, normalized_preds, id_to_label))

    # %% --------------------CONF + NMS
    validation_conf_nms = post_process_conf_filter_nms(pred_values, confidence_filter_thr,
                                                       iou_thr)

    # normalize
    normalized_preds_nms = normalize_bb(validation_conf_nms, gt_df, "transformed_height",
                                        "transformed_width")

    print(zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label))

    # %% --------------------CONF + WBF
    validation_conf_wbf = post_process_conf_filter_wbf(pred_values, confidence_filter_thr,
                                                       iou_thr, gt_df)

    # normalize
    normalized_preds_wbf = normalize_bb(validation_conf_wbf, gt_df, "transformed_height",
                                        "transformed_width")

    print(zfturbo_compute_mAP(normalized_gt, normalized_preds_wbf, id_to_label))
