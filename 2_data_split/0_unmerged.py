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
import albumentations as A
import numpy as np

# %% --------------------
train_df = pd.read_csv(f"{BASE_DIR}/input_data/512x512/train.csv")

# %% --------------------
train_df.head()

# %% --------------------
train_df.loc[train_df.class_id == 14, ["x_max", "y_max"]] = 1
train_df.loc[train_df.class_id == 14, ["x_min", "y_min"]] = 0

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
                                                      "class_name", "width",
                                                      "height"],
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
                ['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', "width", "height"]]

# %% --------------------
no_finding_df.head()

# %% --------------------
# abnormality class present
finding_df = train_df[train_df["class_id"] != 14].reset_index().copy()

finding_df = finding_df.loc[:,
             ['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', "width", "height"]]

# %% --------------------
# resize the annotations to be compatible for images of 512 x 512
for img_id in finding_df["image_id"].unique():
    width, height = finding_df.loc[finding_df["image_id"] == img_id, ['width', 'height']].values[0]

    # first height then width: checked for image: 004dc2a50591fb5f1aaf012bffa95fd9
    dummy_image = np.empty(shape=(height, width))

    # bbox information 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    bbox_info = finding_df.loc[
        finding_df["image_id"] == img_id, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"]].values

    resize_transform = A.Compose([A.Resize(height=512, width=512, always_apply=True)],
                                 bbox_params=A.BboxParams(format="pascal_voc"))

    # get the resized transformation
    transformed = resize_transform(image=dummy_image, bboxes=bbox_info)

    # update the bbox information
    finding_df.loc[
        finding_df["image_id"] == img_id, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"]] = \
        pd.DataFrame(transformed["bboxes"],
                     columns=['x_min', 'y_min', 'x_max', 'y_max', "class_id"]).values

# %% --------------------
unmerged = no_finding_df.append(finding_df)

# %% --------------------
unmerged = unmerged.sort_values("image_id").reset_index(drop=True)

# %% --------------------
os.makedirs(f"{BASE_DIR}/2_data_split/512/unmerged/100_percent_train/", exist_ok=True)

# %% --------------------
unmerged.to_csv(f"{BASE_DIR}/2_data_split/512/unmerged/100_percent_train/unmerged.csv", index=False)
