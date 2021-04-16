# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
# BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------start here
import pandas as pd
from common.utilities import multi_label_split_based_on_percentage, display_fold_distribution
import os

# %% --------------------
# read train.csv file
merged_wbf = pd.read_csv(f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train/train_df.csv")

# %% --------------------
# filter and keep only rows which are abnormal
merged_wbf_abnormal = merged_wbf[merged_wbf["class_id"] != 14].copy(deep=True)

# %% --------------------
# perform 90-10 stratified split
train_df, holdout_df = multi_label_split_based_on_percentage(merged_wbf_abnormal, 1, 0.10,
                                                             "image_id",
                                                             "class_id", seed=42)

# %% --------------------
# visualize the split
display_fold_distribution(train_df, holdout_df, "class_id", color=list('rgbkymc'))

# %% --------------------
train_df = train_df.drop(["fold"], axis=1)
holdout_df = holdout_df.drop(["fold"], axis=1)

# %% --------------------
os.makedirs(f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train/object_detection/90_percent",
            exist_ok=True)
os.makedirs(f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train/object_detection/10_percent",
            exist_ok=True)

# %% --------------------
# save in csv
train_df.to_csv(
    f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train/object_detection/90_percent/train_df.csv",
    index=False)

holdout_df.to_csv(
    f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train/object_detection/10_percent/holdout_df.csv",
    index=False)
