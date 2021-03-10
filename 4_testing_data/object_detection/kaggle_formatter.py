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
TEST_PREDICTION_DIR = os.getenv("TEST_PREDICTION_DIR")

# %% --------------------START HERE
import pandas as pd


# %% --------------------
# https://www.kaggle.com/pestipeti/vinbigdata-fasterrcnn-pytorch-inference#kln-125
def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            int(j[0]), j[1], j[2][0], j[2][1], j[2][2], j[2][3]))

    return " ".join(pred_strings)


# %% --------------------
def submission_file_creator(df, x_min_col, y_min_col, x_max_col, y_max_col, label_col, score_col):
    # img_id, label confidence bb label confidence bb label confidence bb
    image_id_arr = []
    predictions_arr = []

    for image_id in df["image_id"].unique():
        image_id_arr.append(image_id)
        labels = df[df["image_id"] == image_id][label_col]
        scores = df[df["image_id"] == image_id][score_col]
        boxes = df[df["image_id"] == image_id][
            [x_min_col, y_min_col, x_max_col, y_max_col]].to_numpy()

        predictions_arr.append(format_prediction_string(labels, boxes, scores))

    return pd.DataFrame({"image_id": image_id_arr, "PredictionString": predictions_arr})


# %% --------------------
nms_upscaled = pd.read_csv(TEST_PREDICTION_DIR + "/nms_test_upscaled.csv")
wbf_upscaled = pd.read_csv(TEST_PREDICTION_DIR + "/wbf_test_upscaled.csv")

# %% --------------------
# NMS output
result_nms = submission_file_creator(nms_upscaled, "x_min", "y_min", "x_max", "y_max", "label",
                                     "confidence_score")
result_nms.to_csv(TEST_PREDICTION_DIR + "/nms_submission.csv", index=False)

# %% --------------------
# WBF output
result_wbf = submission_file_creator(wbf_upscaled, "x_min", "y_min", "x_max", "y_max", "label",
                                     "confidence_score")
result_wbf.to_csv(TEST_PREDICTION_DIR + "/wbf_submission.csv", index=False)
