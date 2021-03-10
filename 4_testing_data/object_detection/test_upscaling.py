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

# %% --------------------START HERE
import pandas as pd
from common.utilities import convert_bb_smallest_max_scale, get_bb_info

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
# UPSCALE
nms_test = pd.read_csv(TEST_PREDICTION_DIR + "/nms test.csv")
wbf_test = pd.read_csv(TEST_PREDICTION_DIR + "/wbf test.csv")
no_findings_test = pd.read_csv(TEST_PREDICTION_DIR + "/no findings test.csv")

# %% --------------------
test_data = pd.read_csv(TEST_DIR + "/test_original_dimension.csv")

# %% --------------------
agg_test = test_data.groupby(["image_id"]).aggregate(
    {"original_width": "first", "original_height": "first", "transformed_width": "first",
     "transformed_height": "first"})

# %% --------------------
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for img in nms_test["image_id"].unique():
    o_width, o_height, t_width, t_height = agg_test.loc[
        img, ["original_width", "original_height", "transformed_width", "transformed_height"]]

    bounding_boxes_info_nms = get_bb_info(nms_test, img,
                                          ["x_min", "y_min", "x_max", "y_max", "label",
                                           "confidence_score"])

    # upscale the predicted bounding boxes based on original scale and visualize it
    bounding_boxes_info_nms[:, [0, 1, 2, 3]] = convert_bb_smallest_max_scale(
        bounding_boxes_info_nms[:, [0, 1, 2, 3]], t_width, t_height, o_width, o_height)

    for i in range(len(bounding_boxes_info_nms)):
        image_id_arr.append(img)
        x_min_arr.append(bounding_boxes_info_nms[i][0])
        y_min_arr.append(bounding_boxes_info_nms[i][1])
        x_max_arr.append(bounding_boxes_info_nms[i][2])
        y_max_arr.append(bounding_boxes_info_nms[i][3])
        label_arr.append(bounding_boxes_info_nms[i][4])
        score_arr.append(bounding_boxes_info_nms[i][5])

nms_test_upscaled = pd.DataFrame(
    {"image_id": image_id_arr, "x_min": x_min_arr, "y_min": y_min_arr, "x_max": x_max_arr,
     "y_max": y_max_arr, "label": label_arr, "confidence_score": score_arr})

nms_test_upscaled = nms_test_upscaled.append(no_findings_test)

# %% --------------------
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for img in wbf_test["image_id"].unique():
    o_width, o_height, t_width, t_height = agg_test.loc[
        img, ["original_width", "original_height", "transformed_width", "transformed_height"]]

    bounding_boxes_info_wbf = get_bb_info(wbf_test, img,
                                          ["x_min", "y_min", "x_max", "y_max", "label",
                                           "confidence_score"])

    # upscale the predicted bounding boxes based on original scale and visualize it
    bounding_boxes_info_wbf[:, [0, 1, 2, 3]] = convert_bb_smallest_max_scale(
        bounding_boxes_info_wbf[:, [0, 1, 2, 3]], t_width, t_height, o_width, o_height)

    for i in range(len(bounding_boxes_info_wbf)):
        image_id_arr.append(img)
        x_min_arr.append(bounding_boxes_info_wbf[i][0])
        y_min_arr.append(bounding_boxes_info_wbf[i][1])
        x_max_arr.append(bounding_boxes_info_wbf[i][2])
        y_max_arr.append(bounding_boxes_info_wbf[i][3])
        label_arr.append(bounding_boxes_info_wbf[i][4])
        score_arr.append(bounding_boxes_info_wbf[i][5])

wbf_test_upscaled = pd.DataFrame(
    {"image_id": image_id_arr, "x_min": x_min_arr, "y_min": y_min_arr, "x_max": x_max_arr,
     "y_max": y_max_arr, "label": label_arr, "confidence_score": score_arr})

wbf_test_upscaled = wbf_test_upscaled.append(no_findings_test)

# %% --------------------
nms_test_upscaled.to_csv(TEST_PREDICTION_DIR + "/nms_test_upscaled.csv")
wbf_test_upscaled.to_csv(TEST_PREDICTION_DIR + "/wbf_test_upscaled.csv")
