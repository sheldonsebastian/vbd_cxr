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
import numpy as np
from common.utilities import filter_df_based_on_confidence_threshold, merge_bb_nms, merge_bb_wbf

# %% --------------------
# read the predicted test csv
test_predictions = pd.read_csv(TEST_PREDICTION_DIR + "/test_predictions.csv")

# %% --------------------
test_predictions["label"] -= 1

# %% --------------------
filtered_test_predictions = filter_df_based_on_confidence_threshold(test_predictions,
                                                                    "confidence_score", 0.10)

# %% --------------------
# merge bb based on NMS
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for image_id in sorted(filtered_test_predictions["image_id"].unique()):
    bb_df = \
        filtered_test_predictions[filtered_test_predictions["image_id"] == image_id][
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
test_data = pd.read_csv(TEST_DIR + "/test_original_dimension.csv")

# %% --------------------
# merge bb based on WBF
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
score_arr = []

for image_id in sorted(filtered_test_predictions["image_id"].unique()):
    bb_df = \
        filtered_test_predictions[filtered_test_predictions["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]
    bb_df = bb_df.to_numpy()
    t_width, t_height = \
        test_data[test_data["image_id"] == image_id][
            ["transformed_width", "transformed_height"]].values[0]

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

# %% --------------------NO FINDINGS LOGIC
print(len(test_predictions["image_id"].unique()))  # 2996

# %% --------------------
print(len(filtered_test_predictions["image_id"].unique()))  # 2980

# %% --------------------
image_ids_with_no_findings = np.setdiff1d(
    np.union1d(test_predictions["image_id"], test_data["image_id"]),
    filtered_test_predictions["image_id"]).tolist()

# %% --------------------
no_findings_df = pd.DataFrame(
    {"image_id": image_ids_with_no_findings, "x_min": [0] * len(image_ids_with_no_findings),
     "y_min": [0] * len(image_ids_with_no_findings), "x_max": [1] * len(image_ids_with_no_findings),
     "y_max": [1] * len(image_ids_with_no_findings),
     "label": [14] * len(image_ids_with_no_findings),
     "confidence_score": [1] * len(image_ids_with_no_findings)})

# %% --------------------
no_findings_df.to_csv(TEST_PREDICTION_DIR + "/no findings test.csv", index=False)
nms_filtered_df.to_csv(TEST_PREDICTION_DIR + "/nms test.csv", index=False)
wbf_filtered_df.to_csv(TEST_PREDICTION_DIR + "/wbf test.csv", index=False)
