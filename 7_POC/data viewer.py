# %% --------------------
# handle imports
import random

import pandas as pd

from common.utilities import get_image_as_array, get_bb_info, bounding_box_plotter

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
# IMAGE DIR
img_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
          "Workspace/vbd_cxr/9_data/512/transformed_data/train"

# %% --------------------
# ANNOTATION DIR
train_data = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/1_merger/wbf_merged/90_percent_train/object_detection/95_percent"
    "/80_percent/train_df_0.csv")

# %% --------------------
# image_ids = train_data["image_id"].unique()
image_ids = ["e1a4353d3e747a7150cb06cac73f4d6f"]
# shuffle is inplace operation
random.shuffle(image_ids)

for img in image_ids[:10]:
    img_array = get_image_as_array(f"{img_dir}/{img}.jpeg")

    # get bounding box info
    img_bb_info = get_bb_info(train_data, img, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

    # plot image with bounding boxes
    bounding_box_plotter(img_array, img, img_bb_info, label2color, save_title_or_plot="plot")
