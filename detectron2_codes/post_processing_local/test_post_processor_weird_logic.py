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
# probability threshold for 2 class classifier
upper_thr = 0.8  # more chance of having disease
lower_thr = 0.2  # less chance of having disease

obj_det_conf_thr = 0.10
iou_threshold = 0.4

# %% --------------------read the predictions
# read binary classifier outputs
binary_prediction = pd.read_csv(
    LOCAL_TEST_DIR + "/2_class_classifier/predictions/test_2_class_resnet152.csv")

# read object detection outputs
object_detection_prediction = pd.read_csv(
    DETECTRON2_DIR + "/faster_rcnn/test/unmerged_external/test.csv")

# %% --------------------
original_dataset = pd.read_csv(KAGGLE_TEST_DIR + "/test_original_dimension.csv")

# get all image ids in original dataset
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------CONFIDENCE + NMS
# will also contain no findings class with 100% probability
nms_predictions = post_process_conf_filter_nms(object_detection_prediction,
                                               obj_det_conf_thr,
                                               iou_threshold, original_image_ids)

# %% --------------------CONFIDENCE + WBF
wbf_predictions = post_process_conf_filter_wbf(object_detection_prediction,
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

# %% --------------------ONLY WORKING WITH NMS for now
# append probabilities
merged_preds = pd.merge(binary_prediction, formatted_nms, on='image_id', how='left')
merged_preds = merged_preds.drop("target", axis=1)


# %% --------------------
# TODO use updated functions instead of this
def filter_2cls(row, low_thr=lower_thr, high_thr=upper_thr):
    prob = row['probabilities']
    if prob < low_thr:
        ## Less chance of having any disease
        row['PredictionString'] = '14 1 0 0 1 1'
    elif low_thr <= prob < high_thr:
        ## More change of having any diesease
        row['PredictionString'] += f' 14 {prob} 0 0 1 1'
    elif high_thr <= prob:
        ## Good chance of having any disease so believe in object detection model
        row['PredictionString'] = row['PredictionString']
    else:
        raise ValueError('Prediction must be from [0-1]')
    return row


# %% --------------------
sub = merged_preds.apply(filter_2cls, axis=1)

# %% --------------------write the submission file
# DYNAMIC
sub[['image_id', 'PredictionString']].to_csv(
    output_directory + '/pipeline_resnet152_faster_rcnn_unmerged_external_nms_weird_logic.csv',
    index=False)
