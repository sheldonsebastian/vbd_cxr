# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
# BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------START HERE
import pandas as pd
import os

# %% --------------------
train_df = pd.read_csv(f"{BASE_DIR}/input_data/512/transformed_data/train/transformed_train.csv")

# %% --------------------
train_df.head()

# %% --------------------
train_df.loc[train_df.class_id == 14, ["x_max", "y_max"]] = 1

# %% --------------------
train_df.head()

# %% --------------------
label2color = {0: ("Aortic enlargement", "#2a52be"),
               1: ("Atelectasis", "#ffa812"),
               2: ("Calcification", "#ff8243"),
               3: ("Cardiomegaly", "#4682b4"),
               4: ("Consolidation", "#ddadaf"),
               5: ("ILD", "#a3c1ad"),
               6: ("Infiltration", "#008000"),
               7: ("Lung Opacity", "#004953"),
               8: ("Nodule/Mass", "#e3a857"),
               9: ("Other lesion", "#dda0dd"),
               10: ("Pleural effusion", "#e6e8fa"),
               11: ("Pleural thickening", "#800020"),
               12: ("Pneumothorax", "#918151"),
               13: ("Pulmonary fibrosis", "#e75480"),
               14: ("No finding", "#ffffff")
               }

# %% --------------------
# no finding class
no_finding_df = train_df[train_df["class_id"] == 14].copy()

# %% --------------------

no_finding_df.head()

# %% --------------------

len(no_finding_df["image_id"].unique())

# %% --------------------

# remove duplicate no finding class
no_finding_df = no_finding_df.drop_duplicates(subset=["image_id", "x_min", "y_min",
                                                      "x_max", "y_max", "class_id",
                                                      "class_name", "original_width",
                                                      "original_height", "transformed_width",
                                                      "transformed_height"],
                                              ignore_index=True)

# %% --------------------
no_finding_df.head()

# %% --------------------
len(no_finding_df)

# %% --------------------
no_finding_df.columns

# %% --------------------
# delete rad_id column
no_finding_df = no_finding_df.loc[:,
                ['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', "original_width",
                 "original_height", "transformed_width", "transformed_height"]]

# %% --------------------
no_finding_df.head()

# %% --------------------
# abnormality class present
finding_df = train_df[train_df["class_id"] != 14].reset_index().copy()

finding_df = finding_df.loc[:,
             ['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', "original_width",
              "original_height", "transformed_width", "transformed_height"]]

# %% --------------------
unmerged = no_finding_df.append(finding_df)

# %% --------------------
unmerged = unmerged.sort_values("image_id").reset_index(drop=True)

# %% --------------------
os.makedirs(f"{BASE_DIR}/1_data_split/512/unmerged/100_percent_train/", exist_ok=True)

# %% --------------------
unmerged.to_csv(f"{BASE_DIR}/1_data_split/512/unmerged/100_percent_train/unmerged.csv", index=False)
