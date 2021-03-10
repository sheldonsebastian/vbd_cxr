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
TEST_DIR = os.getenv("TEST_DIR")
TEST_PREDICTION_DIR = os.getenv("TEST_PREDICTION_DIR")

# %% --------------------start here
import pandas as pd
from common.utilities import bounding_box_plotter, get_image_as_array, get_bb_info, \
    merge_bb_nms, merge_bb_wbf

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
test_prediction = pd.read_csv(TEST_PREDICTION_DIR + "/test_predictions.csv")

# %% --------------------
# convert classes by removing background class
test_prediction["label"] -= 1

# %% --------------------
# get random image id
img = "0271d381c3e88527721efcfaf518be71"

# %% --------------------
img_as_arr = get_image_as_array(TEST_DIR + f"/{img}.jpeg")

# %% --------------------
bounding_boxes_info_pred = get_bb_info(test_prediction, img,
                                       ["x_min", "y_min", "x_max", "y_max", "label"])

# %% --------------------
# visualize predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"Predicted::{img}", bounding_boxes_info_pred, label2color)

# %% --------------------
# NMS
bounding_boxes_info_with_score = get_bb_info(test_prediction, img,
                                             ["x_min", "y_min", "x_max", "y_max", "label",
                                              "confidence_score"])

# get_bb_info returns numpy array which has no names thus used index numbers
bounding_boxes_info_nms = merge_bb_nms(bounding_boxes_info_with_score, 0, 1, 2, 3, iou_thr=0.1,
                                       scores_col=5)

# visualize nms predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"NMS Predicted::{img}", bounding_boxes_info_nms, label2color)

# %% --------------------
test_data = pd.read_csv(TEST_DIR + "/test_original_dimension.csv")

# %% --------------------
t_width, t_height = \
    test_data[test_data["image_id"] == img][[
        "transformed_width", "transformed_height"]].values[0]

# WBF
bounding_boxes_info_wbf = merge_bb_wbf(t_width, t_height, bounding_boxes_info_with_score, 4,
                                       0, 1, 2, 3, iou_thr=0.2, scores_col=5)

# visualize wbf predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, f"WBF Predicted::{img}", bounding_boxes_info_wbf, label2color)
