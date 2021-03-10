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

# %% --------------------START HERE
import pandas as pd
from common.utilities import bounding_box_plotter, convert_bb_smallest_max_scale, dicom2array, \
    get_bb_info, get_image_as_array

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
train_data = pd.read_csv(data_dir + "/transformed_train.csv")

# %% --------------------
agg_train = train_data.groupby(["image_id"]).aggregate(
    {"original_width": "first", "original_height": "first", "transformed_width": "first",
     "transformed_height": "first"})

# %% --------------------
img = "81b2b950caf9b6c1f2ba9162f3fd259b"

# %% --------------------
o_width, o_height, t_width, t_height = agg_train.loc[
    img, ["original_width", "original_height", "transformed_width", "transformed_height"]]

# %% --------------------
# predicted validation data
validation_prediction = pd.read_csv(validation_prediction_dir + "/validation_predictions.csv")

# convert classes by removing background class
validation_prediction["label"] -= 1

# %% --------------------
bounding_boxes_info_pred = get_bb_info(validation_prediction, img,
                                       ["x_min", "y_min", "x_max", "y_max", "label"])

# %% --------------------
img_as_arr = get_image_as_array(image_dir + f"/{img}.jpeg")

# %% --------------------
# visualize predicted bb on 1024 dimension
bounding_box_plotter(img_as_arr, img, bounding_boxes_info_pred, label2color)

# %% --------------------
# upscale the predicted bounding boxes based on original scale and visualize it
bounding_boxes_info_pred[:, [0, 1, 2, 3]] = convert_bb_smallest_max_scale(
    bounding_boxes_info_pred[:, [0, 1, 2, 3]], t_width, t_height, o_width, o_height)

# %% --------------------
og_img_arr = dicom2array(f"{validation_prediction_dir}/{img}.dicom", voi_lut=True,
                         fix_monochrome=True)

# %% --------------------
bounding_box_plotter(og_img_arr, img, bounding_boxes_info_pred, label2color)
