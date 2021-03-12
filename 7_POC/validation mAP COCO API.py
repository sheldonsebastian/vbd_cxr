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

# %% --------------------
# add home dir to pythonpath
sys.path.append(os.getenv("home_dir"))

# directories
image_dir = os.getenv("image_dir")
bb_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
          "Workspace/vbd_cxr/7_POC/fused_train_0_6_mAP_GT.csv "
validation_prediction_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
                            "Workspace/vbd_cxr/7_POC/validation_predictions_coco_api.csv "

# %% --------------------start here
import pandas as pd
from common.mAP_utils import DataFrameToCOCODataset, get_map, get_id_to_label_mAP
from common.post_processing_utils import post_process_conf_filter_nms, post_process_conf_filter_wbf

# %% --------------------
id_to_label_map = get_id_to_label_mAP()

# %% --------------------
# read the predicted validation csv
validation_predictions = pd.read_csv(validation_prediction_dir)

# %% --------------------
# read ground truth csv
gt_df = pd.read_csv(bb_file)

# %% --------------------
# merge with validation predicted image ids
validation_predictions_image_ids = validation_predictions["image_id"].unique()

# https://stackoverflow.com/questions/19960077/how-to-filter-pandas-dataframe-using-in-and-not-in-like-in-sql
gt_df = gt_df[gt_df["image_id"].isin(validation_predictions_image_ids)]

# %% --------------------
# validation_predictions["label"] -= 1
gt_df["class_id"] += 1

# %% --------------------
# compute map based on validation target data
pred_coco_ds = DataFrameToCOCODataset(validation_predictions, id_to_label_map, "image_id", "x_min",
                                      "y_min", "x_max", "y_max", "label",
                                      "confidence_score").get_coco_dataset()

gt_coco_ds = DataFrameToCOCODataset(gt_df, id_to_label_map, "image_id", "x_min", "y_min", "x_max",
                                    "y_max", "class_id").get_coco_dataset()

# %% --------------------
print(get_map(gt_coco_ds, pred_coco_ds, 0.4))

# %% --------------------CONF + NMS
validation_conf_nms = post_process_conf_filter_nms(validation_predictions, 0.10, 0.4)

# %% --------------------
filtered_pred_coco_ds_nms = DataFrameToCOCODataset(validation_conf_nms, id_to_label_map,
                                                   "image_id", "x_min", "y_min", "x_max", "y_max",
                                                   "label", "confidence_score").get_coco_dataset()

# %% --------------------
print(get_map(gt_coco_ds, filtered_pred_coco_ds_nms, 0.4))

# %% --------------------CONF + WBF
validation_conf_wbf = post_process_conf_filter_wbf(validation_predictions, 0.10, 0.4, gt_df)

# %% --------------------
filtered_pred_coco_ds_wbf = DataFrameToCOCODataset(validation_conf_wbf, id_to_label_map,
                                                   "image_id", "x_min", "y_min", "x_max", "y_max",
                                                   "label", "confidence_score").get_coco_dataset()

# %% --------------------
print(get_map(gt_coco_ds, filtered_pred_coco_ds_wbf, 0.4))
