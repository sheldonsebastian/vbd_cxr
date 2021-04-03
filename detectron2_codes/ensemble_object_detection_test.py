import numpy as np
import pandas as pd

from common.detectron2_post_processor_utils import post_process_conf_filter_nms
from common.kaggle_utils import up_scaler
# %% --------------------PREDICTIONS
# read faster rcnn output
from common.utilities_object_detection_ensembler import ensemble_object_detectors

faster_rcnn = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                          "Workspace/vbd_cxr/final_outputs/test_predictions/test_faster_rcnn.csv")

# read retinanet output
retinanet = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                        "Workspace/vbd_cxr/final_outputs/test_predictions/test_retinanet.csv")

# read yolo v5 output in upscaled format
yolov5 = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                     "Workspace/vbd_cxr/final_outputs/test_predictions/test_yolov5.csv")

# %% -------------------- GROUND TRUTH
test_dim = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                       "Workspace/vbd_cxr/9_data/512/transformed_data/test/test_original_dimension.csv")

original_image_ids = test_dim["image_id"].unique()
upscale_height = "transformed_height"
upscale_width = "transformed_width"

# # # %% --------------------Downscale YOLO
yolov5_downscaled = up_scaler(yolov5, test_dim,
                              columns=["x_min", "y_min", "x_max", "y_max", "label",
                                       "confidence_score"],
                              source_height_col="original_height",
                              source_width_col="original_width",
                              target_height_col="transformed_height",
                              target_width_col="transformed_width")

# %% --------------------Combine outputs of predictors
predictors = [faster_rcnn, retinanet, yolov5_downscaled]

# %% --------------------POST PROCESSING
confidence_filter_thr = 0.05
iou_thr = 0.4
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
# ensembles the outputs and also adds missing image ids
ensembled_outputs = ensemble_object_detectors(post_processed_predictors, test_dim,
                                              upscale_height, upscale_width, iou_thr, [2, 1, 6])

# %% --------------------
ensembled_output_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
                       "Workspace/vbd_cxr/detectron2_codes/ensembles"
# save output as csv
ensembled_outputs.to_csv(
    ensembled_output_dir + "/test_ensemble_faster_rcnn_retinanet_yolov5_216.csv",
    index=False)
