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

# %% --------------------
# add home dir to pythonpath
sys.path.append(os.getenv("home_dir"))

# directories
image_dir = os.getenv("image_dir")
bb_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
          "Workspace/vbd_cxr/7_POC/fused_train_0_6_mAP_GT.csv "
validation_prediction_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
                            "Workspace/vbd_cxr/7_POC/validation_predictions_coco_api.csv "

# %% --------------------start here
import pandas as pd
from common.mAP_utils import mean_average_precision
from common.utilities import filter_df_based_on_confidence_threshold, merge_bb_nms, merge_bb_wbf

# %% --------------------
# read the predicted validation csv
validation_predictions = pd.read_csv(validation_prediction_dir)
validation_predictions["label"] -= 1

# %% --------------------
# read ground truth csv
gt_df = pd.read_csv(bb_file)

# %% --------------------
# merge with validation predicted image ids
validation_predictions_image_ids = validation_predictions["image_id"].unique()

# https://stackoverflow.com/questions/19960077/how-to-filter-pandas-dataframe-using-in-and-not-in-like-in-sql
gt_df = gt_df[gt_df["image_id"].isin(validation_predictions_image_ids)]
gt_df = gt_df.reset_index(drop=True)

# %% --------------------
# compute map based on validation target data
pred_boxes = []
for idx, row in validation_predictions.iterrows():
    # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    temp = [row["image_id"], int(row["label"]), float(row["confidence_score"]), float(row["x_min"]),
            float(row["y_min"]), float(row["x_max"]), float(row["y_max"])]
    pred_boxes.append(temp)
    del temp

true_boxes = []
for idx, row in gt_df.iterrows():
    # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    temp = [row["image_id"], int(row["class_id"]), 0.0, float(row["x_min"]), float(row["y_min"]),
            float(row["x_max"]), float(row["y_max"])]
    true_boxes.append(temp)
    del temp

# %% --------------------
print(mean_average_precision(pred_boxes, true_boxes, 0.4))

# %% --------------------
filtered_validation_predictions = filter_df_based_on_confidence_threshold(validation_predictions,
                                                                          "confidence_score", 0.10)

# %% --------------------
filtered_pred_boxes = []
for idx, row in filtered_validation_predictions.iterrows():
    # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    temp = [row["image_id"], int(row["label"]), float(row["confidence_score"]), float(row["x_min"]),
            float(row["y_min"]), float(row["x_max"]), float(row["y_max"])]
    filtered_pred_boxes.append(temp)
    del temp

# %% --------------------
print(mean_average_precision(filtered_pred_boxes, true_boxes, 0.4))

# %% --------------------
# compute map based on NMS
nms_filtered_pred_boxes = []

for image_id in sorted(filtered_validation_predictions["image_id"].unique()):
    bb_df = \
        filtered_validation_predictions[filtered_validation_predictions["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
    bb_df = bb_df.to_numpy()
    nms_bb = merge_bb_nms(bb_df, 0, 1, 2, 3, 4, iou_thr=0.10, scores_col=5)

    for i in range(len(nms_bb)):
        # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        temp = [image_id, nms_bb[i][4], nms_bb[i][5], nms_bb[i][0], nms_bb[i][1], nms_bb[i][2],
                nms_bb[i][3]]
        nms_filtered_pred_boxes.append(temp)

    # %% --------------------
print(mean_average_precision(nms_filtered_pred_boxes, true_boxes, 0.4))

# %% --------------------
# compute map based on WBF
wbf_filtered_pred_boxes = []

for image_id in sorted(filtered_validation_predictions["image_id"].unique()):
    bb_df = \
        filtered_validation_predictions[filtered_validation_predictions["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
    bb_df = bb_df.to_numpy()
    t_width, t_height = \
        gt_df[gt_df["image_id"] == image_id][["transformed_width", "transformed_height"]].values[0]

    wbf_bb = merge_bb_wbf(t_width, t_height, bb_df, 4, 0, 1, 2, 3, iou_thr=0.2, scores_col=5)

    for i in range(len(wbf_bb)):
        # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        temp = [image_id, wbf_bb[i][4], wbf_bb[i][5], wbf_bb[i][0], wbf_bb[i][1], wbf_bb[i][2],
                wbf_bb[i][3]]
        wbf_filtered_pred_boxes.append(temp)

# %% --------------------
print(mean_average_precision(wbf_filtered_pred_boxes, true_boxes, 0.4))
