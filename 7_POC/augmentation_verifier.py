import os
import zipfile

import albumentations
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from common.utilities import bounding_box_plotter, dicom2array, get_bb_info, \
    get_label_2_color_dict


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
bounding_box_plotter(raw_img_arr, image_id, raw_img_bb_info, get_label_2_color_dict())

# %% --------------------
# transform image using albumentations
transform = albumentations.Compose(
    [
        albumentations.augmentations.transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                            saturation=0.5, hue=0.5,
                                                            always_apply=False,
                                                            p=0.5),
        # horizontal flipping
        albumentations.augmentations.transforms.HorizontalFlip(p=0.5),

        # resize the image
        albumentations.SmallestMaxSize(max_size=1024, always_apply=True)
    ],
    bbox_params=albumentations.BboxParams(format='pascal_voc')
)

# %% --------------------
# each row in bounding boxes will contain 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
bboxes = get_bb_info(train_df, image_id, ["x_min", "y_min", "x_max", "y_max", "class_id"])

# %% --------------------
transformed = transform(image=raw_img_arr, bboxes=bboxes)

# %% --------------------
# visualize the transformed image w/ bounding boxes
bounding_box_plotter(transformed["image"], image_id, transformed["bboxes"],
                     get_label_2_color_dict())
