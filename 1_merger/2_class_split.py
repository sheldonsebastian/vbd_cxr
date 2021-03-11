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
from common.utilities import multi_label_split_based_on_percentage, display_fold_distribution, \
    multi_label_split_based_on_fold

# %% --------------------splits for 2 class classifier
# read train_df.csv file
merged_wbf = pd.read_csv(MERGED_DIR + "/wbf_merged/train_df.csv")

# %% --------------------
# create binary class
# 1 = abnormal
# 0 = normal
merged_wbf["class_id"] = merged_wbf["class_id"].replace(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

# %% --------------------
# drop duplicates since we are not interested in bounding boxes
merged_wbf = merged_wbf[["image_id", "class_id"]].drop_duplicates()

# %% --------------------
print(merged_wbf["class_id"].value_counts())

# %% --------------------
# perform 95-5 stratified split
train_df, holdout_df = multi_label_split_based_on_percentage(merged_wbf, 1, 0.05, "image_id",
                                                             "class_id", seed=42)

# %% --------------------
# visualize the split
display_fold_distribution(train_df, holdout_df, "class_id", color=list('rgbkymc'))

# %% --------------------
train_df = train_df.drop(["fold"], axis=1)
holdout_df = holdout_df.drop(["fold"], axis=1)

# %% --------------------
# save in csv
train_df.to_csv(MERGED_DIR + "/wbf_merged/2_class_classifier/train_df.csv",
                index=False)
holdout_df.to_csv(MERGED_DIR + "/wbf_merged/2_class_classifier/holdout_df.csv",
                  index=False)

# %% --------------------
# save as 80-20% train-validation
train_df_80, val_df_20 = multi_label_split_based_on_percentage(train_df, 1, 0.20, "image_id",
                                                               "class_id", seed=42)

# %% --------------------
# visualize the split
display_fold_distribution(train_df_80, val_df_20, "class_id", color=list('rgbkymc'))

# %% --------------------
# save in csv
train_df_80.to_csv(MERGED_DIR + "/wbf_merged/2_class_classifier/train_df_80.csv",
                   index=False)
val_df_20.to_csv(
    MERGED_DIR + "/wbf_merged/2_class_classifier/validation_df_20.csv",
    index=False)