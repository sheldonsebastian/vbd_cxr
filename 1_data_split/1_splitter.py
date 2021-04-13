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

# %% --------------------splits for object detection
# read train.csv file
data = pd.read_csv(MERGED_DIR + "/512/unmerged/100_percent_train/unmerged.csv")

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
os.makedirs(MERGED_DIR + f"/512/unmerged/90_percent_train", exist_ok=True)
os.makedirs(MERGED_DIR + f"/512/unmerged/10_percent_holdout", exist_ok=True)

# %% --------------------
# save in csv
train_df.to_csv(MERGED_DIR + f"/512/unmerged/90_percent_train/train_df.csv",
                index=False)
holdout_df.to_csv(MERGED_DIR + f"/512/unmerged/10_percent_holdout/holdout_df.csv",
                  index=False)
