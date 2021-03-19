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
MERGED_DIR = os.getenv("MERGED_DIR")

# %% --------------------start here
import pandas as pd
from common.utilities import multi_label_split_based_on_percentage, display_fold_distribution

# %% --------------------
# merge mode
# mode = "wbf_merged"
mode = "nms_merged"

# %% --------------------splits for object detection
for iou_thr in ["0", "0_3", "0_6", "0_9"]:
    # read train.csv file
    merged_wbf = pd.read_csv(
        MERGED_DIR + f"/{mode}/100_percent_train/fused_train_{iou_thr}.csv")

    # %% --------------------
    # split into train-holdout sets 90%-10%
    train_df, holdout_df = multi_label_split_based_on_percentage(merged_wbf, 1, 0.1, "image_id",
                                                                 "class_id", seed=42)

    # %% --------------------
    # visualize the split
    display_fold_distribution(train_df, holdout_df, "class_id", color=list('rgbkymc'))

    # %% --------------------
    # drop the fold column
    train_df = train_df.drop(["fold"], axis=1)
    holdout_df = holdout_df.drop(["fold"], axis=1)

    # %% --------------------
    # save in csv
    train_df.to_csv(MERGED_DIR + f"/{mode}/90_percent_train/train_df_{iou_thr}.csv",
                    index=False)
    holdout_df.to_csv(MERGED_DIR + f"/{mode}/10_percent_holdout/holdout_df_{iou_thr}.csv",
                      index=False)
