# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
# BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------start here
import os
import pandas as pd
from common.utilities import multi_label_split_based_on_percentage, display_fold_distribution

# %% --------------------splits for object detection
# read train.csv file
data = pd.read_csv(f"{BASE_DIR}/2_data_split/512/unmerged/100_percent_train/unmerged.csv")

# %% --------------------
# split into train-holdout sets 90%-10%
train_df, holdout_df = multi_label_split_based_on_percentage(data, 1, 0.1, "image_id",
                                                             "class_id", seed=42)

# %% --------------------
# visualize the split
display_fold_distribution(train_df, holdout_df, "class_id", color=list('rgbkymc'))

# %% --------------------
# drop the fold column
train_df = train_df.drop(["fold"], axis=1)
holdout_df = holdout_df.drop(["fold"], axis=1)

# %% --------------------
os.makedirs(f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train", exist_ok=True)
os.makedirs(f"{BASE_DIR}/2_data_split/512/unmerged/10_percent_holdout", exist_ok=True)

# %% --------------------
# save in csv
train_df.to_csv(f"{BASE_DIR}/2_data_split/512/unmerged/90_percent_train/train_df.csv", index=False)
holdout_df.to_csv(f"{BASE_DIR}/2_data_split/512/unmerged/10_percent_holdout/holdout_df.csv",
                  index=False)
