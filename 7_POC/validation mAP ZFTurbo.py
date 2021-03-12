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

# directories
image_dir = os.getenv("image_dir")
bb_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
          "Workspace/vbd_cxr/7_POC/fused_train_0_6_mAP_GT.csv "
validation_prediction_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
                            "Workspace/vbd_cxr/7_POC/validation_predictions_coco_api.csv "

# %% --------------------start here
# https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes
import pandas as pd
from common.mAP_utils import normalize_bb, zfturbo_compute_mAP, get_id_to_label_mAP
from common.post_processing_utils import post_process_conf_filter_nms, post_process_conf_filter_wbf

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
id_to_label = get_id_to_label_mAP()

# %% --------------------NORMALIZE
normalized_gt = normalize_bb(gt_df, gt_df, "transformed_height", "transformed_width")
normalized_preds = normalize_bb(validation_predictions, gt_df, "transformed_height",
                                "transformed_width")

# %% --------------------RAW
print(zfturbo_compute_mAP(normalized_gt, normalized_preds, id_to_label))

# %% --------------------CONF + NMS
validation_conf_nms = post_process_conf_filter_nms(validation_predictions, 0.10, 0.4)

# normalize
normalized_preds_nms = normalize_bb(validation_conf_nms, gt_df, "transformed_height",
                                    "transformed_width")

print(zfturbo_compute_mAP(normalized_gt, normalized_preds_nms, id_to_label))

# %% --------------------CONF + WBF
validation_conf_wbf = post_process_conf_filter_wbf(validation_predictions, 0.10, 0.4, gt_df)

# normalize
normalized_preds_wbf = normalize_bb(validation_conf_wbf, gt_df, "transformed_height",
                                    "transformed_width")

print(zfturbo_compute_mAP(normalized_gt, normalized_preds_wbf, id_to_label))
