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
from common.utilities import display_fold_distribution, \
    multi_label_split_based_on_fold

# %% --------------------splits for 2 class classifier
# read train_df.csv file
# IOU Threshold does not matter for 2 class classifier
merged_wbf = pd.read_csv(MERGED_DIR + f"/512/unmerged/90_percent_train/train_df.csv")

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
# perform 90-10 stratified split
train_df, holdout_df = multi_label_split_based_on_fold(merged_wbf, 3, "image_id", "class_id",
                                                       seed=42)

# %% --------------------
# visualize the split
display_fold_distribution(train_df, holdout_df, "class_id", color=list('rgbkymc'))

# %% --------------------
os.makedirs(MERGED_DIR + f"/512/unmerged/90_percent_train/2_class_classifier/folds",
            exist_ok=True)

# %% --------------------
# save in csv
train_df.to_csv(
    MERGED_DIR + f"/512/unmerged/90_percent_train/2_class_classifier/folds/train_df_folds.csv",
    index=False)
holdout_df.to_csv(MERGED_DIR + f"/512/unmerged/90_percent_train/2_class_classifier/folds"
                               "/holdout_df_folds.csv", index=False)
