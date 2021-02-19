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
data_dir = os.getenv("DATA_DIR")

# %% --------------------start here
import pandas as pd
from common.utilities import bounding_box_plotter, get_image_as_array, get_bb_info, \
    bounding_box_plotter_side_to_side, merge_bb_nms, merge_bb_wbf

# %% --------------------
label2color = {0: ("Aortic enlargement", "#2a52be"),
               1: ("Atelectasis", "#ffa812"),
               2: ("Calcification", "#ff8243"),
               3: ("Cardiomegaly", "#4682b4"),
               4: ("Consolidation", "#ddadaf"),
               5: ("ILD", "#a3c1ad"),
               6: ("Infiltration", "#008000"),
               7: ("Lung Opacity", "#004953"),
               8: ("Nodule/Mass", "#e3a857"),
               9: ("Other lesion", "#dda0dd"),
               10: ("Pleural effusion", "#e6e8fa"),
               11: ("Pleural thickening", "#800020"),
               12: ("Pneumothorax", "#918151"),
               13: ("Pulmonary fibrosis", "#e75480"),
               14: ("No finding", "#ffffff")
               }

# %% --------------------
# predicted validation data
validation_prediction = pd.read_csv(validation_prediction_dir + "/validation_predictions.csv")

# convert classes by removing background class, since GT df does not have background class
validation_prediction["label"] -= 1

# %% --------------------
# get random image id
# img = validation_prediction["image_id"].unique()[0]
img = "81b2b950caf9b6c1f2ba9162f3fd259b"

# %% --------------------
img_as_arr = get_image_as_array(image_dir + f"/{img}.jpeg")

# %% --------------------
bounding_boxes_info_pred = get_bb_info(validation_prediction, img,
                                       ["x_min", "y_min", "x_max", "y_max", "label"])

# %% --------------------
# visualize predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"Predicted::{img}", bounding_boxes_info_pred, label2color)

# %% --------------------
# train data
abnormalities = pd.read_csv(bb_file)

# %% --------------------
# get train data which match with validation data
abnormalities_filtered = abnormalities[
    abnormalities["image_id"].isin(validation_prediction["image_id"].unique())]

# %% --------------------
bounding_boxes_info_actual = get_bb_info(abnormalities_filtered, img,
                                         ["x_min", "y_min", "x_max", "y_max", "class_id"])

# %% --------------------
# visualize original bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"Actual::{img}", bounding_boxes_info_actual, label2color)

# %% --------------------
# visualize the predicted bounding boxes vs actual bounding boxes
bounding_box_plotter_side_to_side(img_as_arr, img, bounding_boxes_info_actual,
                                  bounding_boxes_info_pred, "Actual", "Predicted", label2color)

# %% --------------------
# NMS
bounding_boxes_info_with_score = get_bb_info(validation_prediction, img,
                                             ["x_min", "y_min", "x_max", "y_max", "label",
                                              "confidence_score"])

# get_bb_info returns numpy array which has no names thus used index numbers
bounding_boxes_info_nms = merge_bb_nms(bounding_boxes_info_with_score, 0, 1, 2, 3, iou_thr=0.1,
                                       scores_col=5)

# visualize nms predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"NMS Predicted::{img}", bounding_boxes_info_nms, label2color)

# %% --------------------
# visualize the nms predicted bounding boxes vs actual bounding boxes
bounding_box_plotter_side_to_side(img_as_arr, img, bounding_boxes_info_actual,
                                  bounding_boxes_info_nms, "Actual", "NMS Predicted", label2color)

# %% --------------------
t_width, t_height = \
    abnormalities[abnormalities["image_id"] == img][[
        "transformed_width", "transformed_height"]].values[0]

# WBF
bounding_boxes_info_wbf = merge_bb_wbf(t_width, t_height, bounding_boxes_info_with_score, 4,
                                       0, 1, 2, 3, iou_thr=0.2, scores_col=5)

# visualize wbf predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"WBF Predicted::{img}", bounding_boxes_info_wbf, label2color)

# %% --------------------
# visualize the wbf predicted bounding boxes vs actual bounding boxes
bounding_box_plotter_side_to_side(img_as_arr, img, bounding_boxes_info_actual,
                                  bounding_boxes_info_wbf, "Actual", "WBF Predicted", label2color)
