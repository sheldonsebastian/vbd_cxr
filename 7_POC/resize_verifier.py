import os
import zipfile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from common.utilities import bounding_box_plotter, dicom2array, get_bb_info, \
    resize_image, get_label_2_color_dict


# %% --------------------
def extract_files(source, destination, delete_source=False):
    # extract file in destination
    with zipfile.ZipFile(source, 'r') as zipref:
        zipref.extractall(destination)

    if delete_source:
        # delete source file
        os.remove(source)


# %% --------------------
train_df = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                       "Workspace/vbd_cxr/7_POC/train_original_competition_dimension.csv")

# %% --------------------
api = KaggleApi()
api.authenticate()

# %% --------------------
image_id = "0b62bc6644be72ce4dfa5ea77a77f311"
# download DICOM image using image_id
api.competition_download_file('vinbigdata-chest-xray-abnormalities-detection',
                              f'train/{image_id}.dicom',
                              path='D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm '
                                   'Workspace/vbd_cxr/7_POC/DICOMs')
# %% --------------------
# extract zip
extract_files(
    f"D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace"
    f"/vbd_cxr/7_POC/DICOMs/{image_id}.dicom.zip",
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace"
    f"/vbd_cxr/7_POC/DICOMs/",
    True
)

# %% --------------------
raw_img_arr = dicom2array(f"D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/"
                          f"vbd_cxr/7_POC/DICOMs/{image_id}.dicom")
raw_img_bb_info = get_bb_info(train_df, image_id, ["x_min", "y_min", "x_max", "y_max", "class_id"])

# %% --------------------
# view image as DICOM w/ bounding boxes
bounding_box_plotter(raw_img_arr, image_id, raw_img_bb_info, get_label_2_color_dict(), False)

# %% --------------------
# transform image using albumentations
transformed = resize_image(train_df, raw_img_arr, image_id,
                           ["x_min", "y_min", "x_max", "y_max", "class_id"], smallest_max_size=1024)

# %% --------------------
# visualize the transformed image w/ bounding boxes
bounding_box_plotter(transformed["image"], image_id, transformed["bboxes"],
                     get_label_2_color_dict(), False)

# %% --------------------
# %% --------------------
# transform image using albumentations
transformed = resize_image(train_df, raw_img_arr, image_id,
                           ["x_min", "y_min", "x_max", "y_max", "class_id"], smallest_max_size=512)

# %% --------------------
# visualize the transformed image w/ bounding boxes
bounding_box_plotter(transformed["image"], image_id, transformed["bboxes"],
                     get_label_2_color_dict(), False)
