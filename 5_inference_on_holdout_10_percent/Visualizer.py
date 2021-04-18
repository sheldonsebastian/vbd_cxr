# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
# BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------
import pandas as pd

from common.utilities import bounding_box_plotter_side_to_side, \
    get_label_2_color_dict, get_image_as_array, get_bb_info

# %% --------------------
gt = pd.read_csv(f"{BASE_DIR}/2_data_split/512/unmerged/10_percent_holdout/holdout_df.csv")

predictions = pd.read_csv(
    f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions/holdout_ensemble_classification_object_detection.csv")

# %% --------------------
label2color = get_label_2_color_dict()

# %% --------------------
original_image_ids = gt["image_id"].unique()

# %% --------------------
for image_id in original_image_ids[0:20]:
    img_as_arr = get_image_as_array(f"{BASE_DIR}/input_data/512x512/train/{image_id}.png")

    # %% --------------------
    left = get_bb_info(gt, image_id, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])
    right = get_bb_info(predictions, image_id, ['x_min', 'y_min', 'x_max', 'y_max', "label"])

    # %% --------------------
    bounding_box_plotter_side_to_side(img_as_arr, image_id, left,
                                      right, "Ground Truth", "Predictions", label2color)
