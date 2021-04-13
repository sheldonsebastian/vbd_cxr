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

# %% --------------------
# read train.csv file
merged_wbf = pd.read_csv(MERGED_DIR + f"/512/unmerged/90_percent_train/train_df.csv")

# %% --------------------
# filter and keep only rows which are abnormal
merged_wbf_abnormal = merged_wbf[merged_wbf["class_id"] != 14].copy(deep=True)

# %% --------------------
# perform 90-10 stratified split
train_df, holdout_df = multi_label_split_based_on_fold(merged_wbf_abnormal, 3, "image_id",
                                                       "class_id", seed=42)

# %% --------------------
# visualize the split
display_fold_distribution(train_df, holdout_df, "class_id", color=list('rgbkymc'))

# %% --------------------
os.makedirs(MERGED_DIR + f"/512/unmerged/90_percent_train/object_detection/folds/",
            exist_ok=True)

# %% --------------------
# save in csv
train_df.to_csv(
    MERGED_DIR + f"/512/unmerged/90_percent_train/object_detection/folds/train_df_fold.csv",
    index=False)

holdout_df.to_csv(
    MERGED_DIR + f"/512/unmerged/90_percent_train/object_detection/folds/holdout_df_fold.csv",
    index=False)
