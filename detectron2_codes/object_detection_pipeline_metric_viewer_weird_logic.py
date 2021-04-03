# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
# env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------imports
from common.kaggle_utils import up_scaler
import pandas as pd
from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    binary_and_object_detection_processing
from common.mAP_utils import normalize_bb, zfturbo_compute_mAP
import numpy as np

# %% --------------------directories
MERGED_DIR = os.getenv("MERGED_DIR")

# %% --------------------
# probability threshold for 2 class classifier
upper_thr = 0.9  # more chance of having disease
lower_thr = 0.1  # less chance of having disease

obj_det_conf_thr = 0.05
iou_threshold = 0.4

# %% --------------------read the predictions
# 2 class filter prediction
binary_prediction = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/final_outputs/holdout/holdout_binary_ensembled.csv")

# object detection predictions
object_detection_prediction = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/detectron2_codes/ensembles/holdout_ensemble_retinanet_yolov5_11.csv")
upscale_gt = False

# %% --------------------
upscale_height = "transformed_height"
upscale_width = "transformed_width"
# GROUND TRUTH
original_dataset = pd.read_csv(MERGED_DIR + "/512/unmerged/10_percent_holdout/holdout_df.csv")

if upscale_gt:
    # upscale GT
    # add confidence score to GT
    temp_conf = original_dataset.copy(deep=True)
    temp_conf["confidence_score"] = np.ones(shape=(len(original_dataset), 1))

    # upscale BB info for gt
    temp = up_scaler(temp_conf, original_dataset,
                     ["x_min", "y_min", "x_max", "y_max", "class_id", "confidence_score"])

    original_dataset.loc[:, ["x_min", "x_max", "y_min", "y_max"]] = temp.loc[:,
                                                                    ["x_min", "x_max", "y_min",
                                                                     "y_max"]]
    upscale_height = "original_height"
    upscale_width = "original_width"

# %% --------------------
# get all image ids in original dataset
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------
id_to_label = {
    0: "aortic enlargement",
    1: "atelectasis",
    2: "calcification",
    3: "cardiomegaly",
    4: "consolidation",
    5: "ild",
    6: "infiltration",
    7: "lung opacity",
    8: "nodule/mass",
    9: "other lesion",
    10: "pleural effusion",
    11: "pleural thickening",
    12: "pneumothorax",
    13: "pulmonary fibrosis",
    14: "No Findings class"
}

# %% --------------------
# normalize gt
normalized_gt_df = normalize_bb(original_dataset, original_dataset, upscale_height,
                                upscale_width)

# %% --------------------CONF + NMS
# post process
nms_final_df = post_process_conf_filter_nms(object_detection_prediction,
                                            obj_det_conf_thr, iou_threshold)

# %% --------------------
# combine 2 class classifier and object detection predictions
binary_object_nms = binary_and_object_detection_processing(binary_prediction, nms_final_df,
                                                           lower_thr, upper_thr)

# normalize
nms_normalized = normalize_bb(binary_object_nms, original_dataset, upscale_height,
                              upscale_width)

# compute mAP
total_map, all_class_map = zfturbo_compute_mAP(normalized_gt_df, nms_normalized,
                                               id_to_label,
                                               gt_class_id="class_id",
                                               pred_class_id="label")
print(total_map, all_class_map)
