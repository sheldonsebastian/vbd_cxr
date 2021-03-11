# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "d:/gwu/4 spring 2021/6501 capstone/vbd cxr/pycharm " \
           "workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
# env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add home dir to pythonpath
sys.path.append(os.getenv("home_dir"))

# %% --------------------imports
from common.kaggle_utils import up_scaler, submission_file_creator
import pandas as pd
from common.post_processing_utils import post_process_conf_filter_nms, post_process_conf_filter_wbf

# %% --------------------directories
TEST_DIR = os.getenv("TEST_DIR")
KAGGLE_TEST_DIR = os.getenv("KAGGLE_TEST_DIR")

# %% --------------------
# read the predicted test csv
test_predictions = pd.read_csv(
    TEST_DIR + "/object_detection/predictions/test_object_detection_prediction.csv")

# %% --------------------
# original dimensions
original_dimension = pd.read_csv(KAGGLE_TEST_DIR + "/test_original_dimension.csv")

# %% --------------------
# adjust labels
test_predictions["label"] -= 1

# %% --------------------
confidence_threshold = 0.5
iou_threshold = 0.4

# %% --------------------CONFIDENCE INTERVAL + NMS
nms_filtered = post_process_conf_filter_nms(test_predictions, confidence_threshold, iou_threshold)

# %% --------------------CONFIDENCE INTERVAL + WBF
wbf_filtered = post_process_conf_filter_wbf(test_predictions, confidence_threshold, iou_threshold,
                                            original_dimension)
# %% --------------------
nms_filtered.to_csv(TEST_DIR + "/object_detection/object_detection_post_processed/nms_filtered.csv",
                    index=False)
wbf_filtered.to_csv(TEST_DIR + "/object_detection/object_detection_post_processed/wbf_filtered.csv",
                    index=False)

# %% --------------------submission prepper
# upscale
upscaled_nms = up_scaler(nms_filtered, original_dimension)
upscaled_wbf = up_scaler(wbf_filtered, original_dimension)

# %% --------------------
# formatter
formatted_nms = submission_file_creator(upscaled_nms, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")
formatted_wbf = submission_file_creator(upscaled_wbf, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")
# %% --------------------
formatted_nms.to_csv(TEST_DIR + "/submissions/nms_object_detection.csv", index=False)
formatted_wbf.to_csv(TEST_DIR + "/submissions/wbf_object_detection.csv", index=False)
