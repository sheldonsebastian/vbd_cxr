# https://www.kaggle.com/backtracking/smart-data-split-train-eval-for-object-detection
# %%--------------------
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

# %% --------------------
# add home dir to pythonpath
sys.path.append(os.getenv("home_dir"))

# directories
train_dir = os.getenv("DATA_DIR")

# %% --------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings

warnings.filterwarnings('ignore')
# %% --------------------

df = pd.read_csv(train_dir + "/transformed_train.csv")
df = pd.DataFrame(df)

# %% --------------------
df = df[df['class_name'] != 'No finding']

# %% --------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_folds = df[['image_id']].copy()

# %% --------------------
df_folds.loc[:, 'bbox_count'] = 1

# %% --------------------
df_folds = df_folds.groupby('image_id').count()

# %% --------------------
df_folds.loc[:, 'object_count'] = df.groupby('image_id')['class_id'].nunique()

# %% --------------------
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['object_count'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)

# %% --------------------
df_folds.loc[:, 'fold'] = 0

# %% --------------------
for fold_number, (train_index, val_index) in enumerate(
        skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

# %% --------------------
# example with fold 0
df_folds.reset_index(inplace=True)

# %% --------------------
# fold 0 for validation
df_valid = pd.merge(df, df_folds[df_folds['fold'] == 1], on='image_id')

# fold 1,2,3,4 for training
df_train = pd.merge(df, df_folds[df_folds['fold'] != 1], on='image_id')

# %% --------------------
figure(num=None, figsize=(30, 8))
df_train['class_name'].hist()
df_valid['class_name'].hist()
plt.show()

# %% --------------------
print("Train")
print((df_train["class_name"].value_counts() / len(df_train)) * 100)

# %% --------------------
print("Validation")
print((df_valid["class_name"].value_counts() / len(df_valid)) * 100)
