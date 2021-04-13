from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import fmin

from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    binary_and_object_detection_processing
from common.kaggle_utils import up_scaler
from common.mAP_utils import normalize_bb, zfturbo_compute_mAP
from common.utilities_object_detection_ensembler import ensemble_object_detectors

# %% --------------------
# probability threshold for 2 class classifier
upper_thr = 0.9  # more chance of having disease
lower_thr = 0.1  # less chance of having disease

confidence_filter_thr = 0.05
iou_thr = 0.4

# %% --------------------
# 2 class filter prediction
binary_prediction = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/final_outputs/holdout/holdout_binary_resnet152_vgg19.csv")

faster_rcnn = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                          "Workspace/vbd_cxr/final_outputs/holdout/holdout_faster_rcnn.csv")

# read yolo v5 output in upscaled format
yolov5 = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                     "Workspace/vbd_cxr/final_outputs/holdout/holdout_yolov5.csv")

# %% -------------------- GROUND TRUTH
holdout_gt_df = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                            "Workspace/vbd_cxr/1_merger/512/unmerged/10_percent_holdout"
                            "/holdout_df.csv")
upscale_gt = False

original_image_ids = holdout_gt_df["image_id"].unique()
upscale_height = "transformed_height"
upscale_width = "transformed_width"

if upscale_gt:
    # upscale GT
    # add confidence score to GT
    temp_conf = holdout_gt_df.copy(deep=True)
    temp_conf["confidence_score"] = np.ones(shape=(len(holdout_gt_df), 1))

    # upscale BB info for gt
    temp = up_scaler(temp_conf, holdout_gt_df,
                     ["x_min", "y_min", "x_max", "y_max", "class_id", "confidence_score"])

    holdout_gt_df.loc[:, ["x_min", "x_max", "y_min", "y_max"]] = temp.loc[:,
                                                                 ["x_min", "x_max", "y_min",
                                                                  "y_max"]]
    upscale_height = "original_height"
    upscale_width = "original_width"

# # # %% --------------------Downscale YOLO
yolov5_downscaled = up_scaler(yolov5, holdout_gt_df,
                              columns=["x_min", "y_min", "x_max", "y_max", "label",
                                       "confidence_score"],
                              source_height_col="original_height",
                              source_width_col="original_width",
                              target_height_col="transformed_height",
                              target_width_col="transformed_width")

# %% --------------------Combine outputs of predictors
predictors = [faster_rcnn, yolov5_downscaled]

# %% --------------------POST PROCESSING
post_processed_predictors = []
# apply post processing individually to each predictor
for holdout_predictions in predictors:
    validation_conf_nms = post_process_conf_filter_nms(holdout_predictions, confidence_filter_thr,
                                                       iou_thr)

    # ids which failed confidence
    normal_ids_nms = np.setdiff1d(original_image_ids,
                                  validation_conf_nms["image_id"].unique())
    print(f"NMS normal ids count: {len(normal_ids_nms)}")
    normal_pred_nms = []
    # add normal ids to dataframe
    for normal_id in set(normal_ids_nms):
        normal_pred_nms.append({
            "image_id": normal_id,
            "x_min": 0,
            "y_min": 0,
            "x_max": 1,
            "y_max": 1,
            "label": 14,
            "confidence_score": 1
        })

    normal_pred_df_nms = pd.DataFrame(normal_pred_nms,
                                      columns=["image_id", "x_min", "y_min", "x_max", "y_max",
                                               "label",
                                               "confidence_score"])
    validation_conf_nms = validation_conf_nms.append(normal_pred_df_nms)

    post_processed_predictors.append(validation_conf_nms)


# %% --------------------MERGE BB for post_processed_predictors
def foo(weights):
    # ensembles the outputs and also adds missing image ids
    ensembled_outputs = ensemble_object_detectors(post_processed_predictors, holdout_gt_df,
                                                  upscale_height, upscale_width, iou_thr, weights)

    # %% --------------------NORMALIZE
    normalized_gt = normalize_bb(holdout_gt_df, holdout_gt_df, upscale_height, upscale_width)

    # %% --------------------CONF + NMS
    validation_conf_nms = post_process_conf_filter_nms(ensembled_outputs, confidence_filter_thr,
                                                       iou_thr)

    # ids which failed confidence
    normal_ids_nms = np.setdiff1d(original_image_ids,
                                  validation_conf_nms["image_id"].unique())
    print(f"NMS normal ids count: {len(normal_ids_nms)}")
    normal_pred_nms = []
    # add normal ids to dataframe
    for normal_id in set(normal_ids_nms):
        normal_pred_nms.append({
            "image_id": normal_id,
            "x_min": 0,
            "y_min": 0,
            "x_max": 1,
            "y_max": 1,
            "label": 14,
            "confidence_score": 1
        })

    normal_pred_df_nms = pd.DataFrame(normal_pred_nms,
                                      columns=["image_id", "x_min", "y_min", "x_max", "y_max",
                                               "label",
                                               "confidence_score"])
    validation_conf_nms = validation_conf_nms.append(normal_pred_df_nms)

    # %% --------------------
    # combine 2 class classifier and object detection predictions
    binary_object_nms = binary_and_object_detection_processing(binary_prediction,
                                                               validation_conf_nms,
                                                               lower_thr, upper_thr)

    # %% --------------------
    # normalize
    normalized_preds_nms = normalize_bb(binary_object_nms, holdout_gt_df, upscale_height,
                                        upscale_width)

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

    return zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label)[0]


# %% --------------------
partial_loss = partial(foo)
init_weights = np.random.dirichlet(np.ones(len(predictors)))
final_weights = fmin(partial_loss, init_weights, disp=True)

print(final_weights)