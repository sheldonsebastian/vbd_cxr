# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
           "Workspace/vbd_cxr/6_environment_files/local.env "

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------imports
import pandas as pd
from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    post_process_conf_filter_wbf
from common.kaggle_utils import up_scaler, submission_file_creator

# %% --------------------directories
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
KAGGLE_TEST_DIR = os.getenv("KAGGLE_TEST_DIR")
LOCAL_TEST_DIR = os.getenv("TEST_DIR")
output_directory = DETECTRON2_DIR + "/post_processing_local/submissions"

# %% --------------------
binary_conf_thr = 0.6
obj_det_conf_thr = 0.10
iou_threshold = 0.4

# %% --------------------read the predictions
# read binary classifier outputs
binary_prediction = pd.read_csv(
    LOCAL_TEST_DIR + "/2_class_classifier/predictions/test_2_class_resnet152.csv")

# read object detection outputs
object_detection_prediction = pd.read_csv(DETECTRON2_DIR + "/faster_rcnn/test/unmerged_external/test.csv")

# %% --------------------
original_dataset = pd.read_csv(KAGGLE_TEST_DIR + "/test_original_dimension.csv")

# get all image ids in original dataset
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------
# UPDATE TARGET OF BINARY CLASSIFIER USING CONF THR
updated_targets = [1 if p >= binary_conf_thr else 0 for p in binary_prediction["probabilities"]]
binary_prediction["target"] = updated_targets

# %% --------------------
# get abnormal ids
abnormal_ids = list(binary_prediction[binary_prediction["target"] == 1]["image_id"].unique())

# subset the object detection df using binary classifier
object_detection_prediction_subset = object_detection_prediction[
    object_detection_prediction["image_id"].isin(abnormal_ids)]

# %% --------------------CONFIDENCE + NMS
nms_predictions = post_process_conf_filter_nms(object_detection_prediction_subset,
                                               obj_det_conf_thr,
                                               iou_threshold, original_image_ids)

# %% --------------------CONFIDENCE + WBF
wbf_predictions = post_process_conf_filter_wbf(object_detection_prediction_subset,
                                               obj_det_conf_thr, iou_threshold,
                                               original_dataset, original_image_ids)

# %% --------------------submission prepper
# upscale
upscaled_nms = up_scaler(nms_predictions, original_dataset)
upscaled_wbf = up_scaler(wbf_predictions, original_dataset)

# %% --------------------
# formatter
formatted_nms = submission_file_creator(upscaled_nms, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")
formatted_wbf = submission_file_creator(upscaled_wbf, "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score")

# %% --------------------write the submission file
# DYNAMIC
formatted_nms.to_csv(output_directory + "/pipeline_resnet152_faster_rcnn_unmerged_external_nms.csv",
                     index=False)
formatted_wbf.to_csv(output_directory + "/pipeline_resnet152_faster_rcnn_unmerged_external_wbf.csv",
                     index=False)
