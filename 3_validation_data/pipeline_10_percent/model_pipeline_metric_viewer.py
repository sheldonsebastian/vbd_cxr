# Compute mAP for validation and 5% holdout set
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

# add home dir to pythonpath
sys.path.append(os.getenv("home_dir"))

# %% --------------------start here
import pandas as pd
import numpy as np
from mean_average_precision import MetricBuilder

# %% --------------------DIRECTORIES
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
MERGE_DIR = os.getenv("MERGED_DIR")

# %% --------------------
# read ground truth csv
holdout_gt_df = pd.read_csv(MERGE_DIR + "/wbf_merged/holdout_df.csv")

# %% --------------------
# read the predicted validation csv
# pipeline predictions are already adjusted for background class
# pipeline predictions are adjusted for confidence threshold and NMS
holdout_predictions = pd.read_csv(
    VALIDATION_PREDICTION_DIR + f"/pipeline_10_percent/predictions/pipeline_prediction.csv")

# %% --------------------
for gt_values, pred_values, title in zip([holdout_gt_df],
                                         [holdout_predictions],
                                         ["Holdout 10% Data"]):
    print("-" * 10 + title + "-" * 10)

    # compute map based on validation target data
    pred_boxes = []
    for idx, row in pred_values.iterrows():
        # [xmin, ymin, xmax, ymax, class_id, confidence]
        temp = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]),
                int(row["label"]), float(row["confidence_score"]), ]
        pred_boxes.append(temp)
        del temp

    true_boxes = []
    for idx, row in gt_values.iterrows():
        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        temp = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]),
                int(row["class_id"]), 0, 0]
        true_boxes.append(temp)
        del temp

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=14)
    metric_fn.add(np.asarray(pred_boxes), np.asarray(true_boxes))
    print(metric_fn.value(iou_thresholds=0.4)['mAP'])
