# %% --------------------
# handle imports
import pandas as pd

from common.utilities import plot_img, get_image_as_array, get_bb_info, bounding_box_plotter

# %% --------------------
train_dir_path = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/transformed_data/train"

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
train_data2 = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/transformed_data/transformed_train.csv")
train_data = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    "Workspace/vbd_cxr/1_merger/wbf_merged/fused_train_0_6.csv")

# %% --------------------
train_data.head()

# %% --------------------
train_data.columns

# %% --------------------
# for img in train_data2[train_data2["class_id"] != 14]["image_id"].unique()[:10]:
for img in ["f7fd31cb67b22bf95cc94d909b6dd2e3"]:
    img_array = get_image_as_array(f"{train_dir_path}/{img}.jpeg")

    # %% --------------------
    # plot original image
    plot_img(img_array, "Original")

    # %% --------------------
    # get bounding box info
    img_bb_info = get_bb_info(train_data, img, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

    # %% --------------------
    # plot image with bounding boxes
    bounding_box_plotter(img_array, img, img_bb_info, label2color)

    img_bb_info2 = get_bb_info(train_data2, img, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])
    bounding_box_plotter(img_array, img, img_bb_info2, label2color)
