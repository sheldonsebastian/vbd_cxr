# Compute mAP for validation and 5% holdout set
# %% --------------------
import os
import sys

from dotenv import load_dotenv

# %% --------------------
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
import numpy as np
from mean_average_precision import MetricBuilder
from common.utilities import filter_df_based_on_confidence_threshold, merge_bb_nms, merge_bb_wbf

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

# %% --------------------
for gt_values, pred_values, title in zip([val_gt_df, holdout_gt_df],
                                         [val_predictions, holdout_predictions],
                                         ["Validation Data", "Holdout Data"]):
    print("-" * 10 + title + "-" * 10)
    # %% --------------------
    # merge with validation predicted image ids
    validation_predictions_image_ids = pred_values["image_id"].unique()

    # https://stackoverflow.com/questions/19960077/how-to-filter-pandas-dataframe-using-in-and-not-in-like-in-sql
    gt_df = gt_values[gt_values["image_id"].isin(validation_predictions_image_ids)]
    gt_df = gt_df.reset_index(drop=True)

    # %% --------------------
    # compute map based on validation target data
    pred_boxes = []
    for idx, row in pred_values.iterrows():
        # [xmin, ymin, xmax, ymax, class_id, confidence]
        temp = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]),
                int(row["label"]), float(row["confidence_score"]), ]
        pred_boxes.append(temp)
        del temp

    true_boxes = []
    for idx, row in gt_df.iterrows():
        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        temp = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]),
                int(row["class_id"]), 0, 0]
        true_boxes.append(temp)
        del temp

    # %% --------------------
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=14)
    metric_fn.add(np.asarray(pred_boxes), np.asarray(true_boxes))
    print(metric_fn.value(iou_thresholds=0.4)['mAP'])

    # %% --------------------
    filtered_validation_predictions = filter_df_based_on_confidence_threshold(pred_values,
                                                                              "confidence_score",
                                                                              confidence_filter_thr)

    # %% --------------------
    filtered_pred_boxes = []
    for idx, row in filtered_validation_predictions.iterrows():
        # [xmin, ymin, xmax, ymax, class_id, confidence]
        temp = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]),
                float(row["y_max"]), int(row["label"]), float(row["confidence_score"])]
        filtered_pred_boxes.append(temp)
        del temp

    # %% --------------------
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=14)
    metric_fn.add(np.asarray(filtered_pred_boxes), np.asarray(true_boxes))
    print(metric_fn.value(iou_thresholds=0.4)['mAP'])

    # %% --------------------
    # compute map based on NMS
    nms_filtered_pred_boxes = []

    for image_id in sorted(filtered_validation_predictions["image_id"].unique()):
        bb_df = \
            filtered_validation_predictions[
                filtered_validation_predictions["image_id"] == image_id][
                ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
        bb_df = bb_df.to_numpy()
        nms_bb = merge_bb_nms(bb_df, 0, 1, 2, 3, 4, iou_thr=iou_thr, scores_col=5)

        for i in range(len(nms_bb)):
            # [xmin, ymin, xmax, ymax, class_id, confidence]
            temp = [nms_bb[i][0], nms_bb[i][1], nms_bb[i][2], nms_bb[i][3], nms_bb[i][4],
                    nms_bb[i][5]]
            nms_filtered_pred_boxes.append(temp)

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=14)
    metric_fn.add(np.asarray(nms_filtered_pred_boxes), np.asarray(true_boxes))
    print(metric_fn.value(iou_thresholds=0.4)['mAP'])

    # %% --------------------
    # compute map based on WBF
    wbf_filtered_pred_boxes = []

    for image_id in sorted(filtered_validation_predictions["image_id"].unique()):
        bb_df = \
            filtered_validation_predictions[
                filtered_validation_predictions["image_id"] == image_id][
                ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
        bb_df = bb_df.to_numpy()
        t_width, t_height = \
            gt_df[gt_df["image_id"] == image_id][
                ["transformed_width", "transformed_height"]].values[0]

        wbf_bb = merge_bb_wbf(t_width, t_height, bb_df, 4, 0, 1, 2, 3, iou_thr=iou_thr,
                              scores_col=5)

        for i in range(len(wbf_bb)):
            # [xmin, ymin, xmax, ymax, class_id, confidence]
            temp = [wbf_bb[i][0], wbf_bb[i][1], wbf_bb[i][2], wbf_bb[i][3], wbf_bb[i][4],
                    wbf_bb[i][5]]
            wbf_filtered_pred_boxes.append(temp)

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=14)
    metric_fn.add(np.asarray(wbf_filtered_pred_boxes), np.asarray(true_boxes))
    print(metric_fn.value(iou_thresholds=0.4)['mAP'])
