# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------imports
import pandas as pd
import numpy as np
from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
    binary_and_object_detection_processing
from common.kaggle_utils import submission_file_creator, rescaler
from common.utilities_object_detection_ensembler import ensemble_object_detectors
import time

start = time.time()

# %% --------------------
# probability threshold for 2 class classifier
upper_thr = 0.95  # more chance of having disease
lower_thr = 0.05  # less chance of having disease

confidence_filter_thr = 0.05
iou_thr = 0.3

# %% --------------------read the predictions
resnet152 = pd.read_csv(
    f"{BASE_DIR}/6_inference_on_kaggle_test_files/files/test_resnet152.csv")
vgg19 = pd.read_csv(f"{BASE_DIR}/6_inference_on_kaggle_test_files/files/test_vgg19.csv")
faster_rcnn = pd.read_csv(f"{BASE_DIR}/6_inference_on_kaggle_test_files/files/test_faster_rcnn.csv")
yolov5 = pd.read_csv(f"{BASE_DIR}/6_inference_on_kaggle_test_files/files/test_yolov5.csv")

# %% --------------------ENSEMBLE CLASSIFICATION MODELS
ensembled_classification = vgg19.copy(deep=True)
ensembled_classification = ensembled_classification.drop(columns=["target"])

# average probability of two models
ensembled_classification["probabilities"] = (vgg19["probabilities"] + resnet152[
    "probabilities"]) / 2

print("Finished Classification Ensembling")

# %% --------------------
# read the test csv that contains original dimensions
original_dataset = pd.read_csv(f"{BASE_DIR}/input_data/512x512/test.csv")

# get all image ids in original dataset
original_image_ids = list(original_dataset["image_id"].unique())

# %% --------------------
# add 512x512 dimensions in GT
original_dataset["transformed_height"] = 512
original_dataset["transformed_width"] = 512

# %% --------------------Downscale YOLO
# downscale YOLOv5 predictions
yolov5 = rescaler(yolov5, original_dataset, "height", "width", "transformed_height",
                  "transformed_width")
print("Finished Downscaling YOLOv5")

# %% --------------------ENSEMBLE OBJECT DETECTION MODELS
predictors = [faster_rcnn, yolov5]

# %% --------------------POST PROCESSING
post_processed_predictors = []
# apply post processing individually to each predictor
for test_predictions in predictors:
    test_conf_nms = post_process_conf_filter_nms(test_predictions, confidence_filter_thr,
                                                 iou_thr)

    # ids which failed confidence
    normal_ids_nms = np.setdiff1d(original_image_ids,
                                  test_conf_nms["image_id"].unique())
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
    test_conf_nms = test_conf_nms.append(normal_pred_df_nms)

    post_processed_predictors.append(test_conf_nms)

# %% --------------------MERGE BB for post_processed_predictors
# ensembles the outputs and also adds missing image ids
ensembled_outputs = ensemble_object_detectors(post_processed_predictors, original_dataset,
                                              "transformed_height", "transformed_width", iou_thr,
                                              [3, 9])
print("Finished WBF process")

# %% --------------------CONF + NMS (POST PROCESSING)
test_conf_nms = post_process_conf_filter_nms(ensembled_outputs, confidence_filter_thr,
                                             iou_thr)

# ids which failed confidence
normal_ids_nms = np.setdiff1d(original_image_ids,
                              test_conf_nms["image_id"].unique())
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
test_conf_nms = test_conf_nms.append(normal_pred_df_nms)

# %% --------------------
# combine 2 class classifier and object detection predictions
binary_object_nms = binary_and_object_detection_processing(ensembled_classification,
                                                           test_conf_nms,
                                                           lower_thr, upper_thr)
print("Finished performing thresholding logic")

# %% --------------------
# round to the next 3 digits to avoid normalization errors
binary_object_nms = binary_object_nms.round(3)

# %% --------------------
upscaled_predictions = rescaler(binary_object_nms, original_dataset,
                                source_height_col="transformed_height",
                                source_width_col="transformed_width",
                                target_height_col="height", target_width_col="width")

print("Finished Upscaling the predictions")

# %% --------------------
# formatter
formatted_nms = submission_file_creator(upscaled_predictions, "x_min", "y_min", "x_max", "y_max",
                                        "label",
                                        "confidence_score")

# %% --------------------
formatted_nms.to_csv(f"{BASE_DIR}/6_inference_on_kaggle_test_files/files/best_submission.csv",
                     index=False)

# %% --------------------
end = time.time() - start

# %% --------------------
print(end)
