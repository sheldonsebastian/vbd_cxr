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
saved_model_path = os.getenv("saved_model_dir") + "/saved_model_20210212.pt"
validation_indices = os.getenv("validation_indices")
image_dir = os.getenv("image_dir")
bb_file = os.getenv("bb_file")
validation_prediction_dir = os.getenv("validation_prediction_dir")

# %% --------------------start here
import pandas as pd
from common.map_utils import DataFrameToCOCODataset, print_map
from common.utilities import filter_df_based_on_confidence_threshold, merge_bb_nms, merge_bb_wbf

# %% --------------------
id_to_label_map = label2color = {
    1: "aortic enlargement",
    2: "atelectasis",
    3: "calcification",
    4: "cardiomegaly",
    5: "consolidation",
    6: "ild",
    7: "infiltration",
    8: "lung opacity",
    9: "nodule/mass",
    10: "other lesion",
    11: "pleural effusion",
    12: "pleural thickening",
    13: "pneumothorax",
    14: "pulmonary fibrosis"
}

# %% --------------------
# read the predicted validation csv
validation_predictions = pd.read_csv(validation_prediction_dir + "/validation_predictions.csv")

# %% --------------------
# read ground truth csv
gt_df = pd.read_csv(bb_file)

# %% --------------------
# merge with validation predicted image ids
validation_predictions_image_ids = validation_predictions["image_id"].unique()

# https://stackoverflow.com/questions/19960077/how-to-filter-pandas-dataframe-using-in-and-not-in-like-in-sql
gt_df = gt_df[gt_df["image_id"].isin(validation_predictions_image_ids)]

# %% --------------------
gt_df["class_id"] += 1

# %% --------------------
# compute map based on validation target data
pred_coco_ds = DataFrameToCOCODataset(validation_predictions, id_to_label_map, "image_id", "x_min",
                                      "y_min", "x_max", "y_max", "label",
                                      "confidence_score").get_coco_dataset()

gt_coco_ds = DataFrameToCOCODataset(gt_df, id_to_label_map, "image_id", "x_min", "y_min", "x_max",
                                    "y_max", "class_id").get_coco_dataset()

# %% --------------------
print_map(gt_coco_ds, pred_coco_ds, 0.4)

# %% --------------------
filtered_validation_predictions = filter_df_based_on_confidence_threshold(validation_predictions,
                                                                          "confidence_score", 0.10)

# %% --------------------
filtered_pred_coco_ds = DataFrameToCOCODataset(filtered_validation_predictions, id_to_label_map,
                                               "image_id", "x_min", "y_min", "x_max", "y_max",
                                               "label", "confidence_score").get_coco_dataset()

# %% --------------------
print_map(gt_coco_ds, filtered_pred_coco_ds, 0.4)

# %% --------------------
# compute map based on NMS
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for image_id in sorted(filtered_validation_predictions["image_id"].unique()):
    bb_df = \
        filtered_validation_predictions[filtered_validation_predictions["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
    bb_df = bb_df.to_numpy()
    nms_bb = merge_bb_nms(bb_df, 0, 1, 2, 3, iou_thr=0.10, scores_col=5)

    for i in range(len(nms_bb)):
        image_id_arr.append(image_id)
        x_min_arr.append(nms_bb[i][0])
        y_min_arr.append(nms_bb[i][1])
        x_max_arr.append(nms_bb[i][2])
        y_max_arr.append(nms_bb[i][3])
        label_arr.append(nms_bb[i][4])
        score_arr.append(nms_bb[i][5])

nms_filtered_df = pd.DataFrame(
    {"image_id": image_id_arr, "x_min": x_min_arr, "y_min": y_min_arr, "x_max": x_max_arr,
     "y_max": y_max_arr, "label": label_arr, "confidence_score": score_arr})

# %% --------------------
nms_filtered_pred_coco_ds = DataFrameToCOCODataset(nms_filtered_df, id_to_label_map,
                                                   "image_id", "x_min", "y_min", "x_max", "y_max",
                                                   "label", "confidence_score").get_coco_dataset()

# %% --------------------
print_map(gt_coco_ds, nms_filtered_pred_coco_ds, 0.4)

# %% --------------------
# compute map based on WBF
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for image_id in sorted(filtered_validation_predictions["image_id"].unique()):
    bb_df = \
        filtered_validation_predictions[filtered_validation_predictions["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
    bb_df = bb_df.to_numpy()
    t_width, t_height = \
        gt_df[gt_df["image_id"] == image_id][["transformed_width", "transformed_height"]].values[0]

    wbf_bb = merge_bb_wbf(t_width, t_height, bb_df, 4, 0, 1, 2, 3, iou_thr=0.2, scores_col=5)

    for i in range(len(wbf_bb)):
        image_id_arr.append(image_id)
        x_min_arr.append(wbf_bb[i][0])
        y_min_arr.append(wbf_bb[i][1])
        x_max_arr.append(wbf_bb[i][2])
        y_max_arr.append(wbf_bb[i][3])
        label_arr.append(wbf_bb[i][4])
        score_arr.append(wbf_bb[i][5])

wbf_filtered_df = pd.DataFrame(
    {"image_id": image_id_arr, "x_min": x_min_arr, "y_min": y_min_arr, "x_max": x_max_arr,
     "y_max": y_max_arr, "label": label_arr, "confidence_score": score_arr})

# %% --------------------
wbf_filtered_pred_coco_ds = DataFrameToCOCODataset(wbf_filtered_df, id_to_label_map,
                                                   "image_id", "x_min", "y_min", "x_max", "y_max",
                                                   "label", "confidence_score").get_coco_dataset()

# %% --------------------
print_map(gt_coco_ds, wbf_filtered_pred_coco_ds, 0.4)
