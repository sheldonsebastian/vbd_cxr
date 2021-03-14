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
image_id = "b41de357cd8bbef33ae563b6299f802c"
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
augmentor = albumentations.Compose(
    [
        # augmentation operations
        albumentations.augmentations.transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                            saturation=0.3, hue=0.3,
                                                            always_apply=False,
                                                            p=0.4),
        albumentations.augmentations.transforms.GlassBlur(p=0.2),
        albumentations.augmentations.transforms.GaussNoise(p=0.2),
        albumentations.augmentations.transforms.RandomGamma(p=0.2),

        # horizontal flipping
        albumentations.augmentations.transforms.HorizontalFlip(p=0.4)
    ],
    bbox_params=albumentations.BboxParams(format='pascal_voc')
)

# %% --------------------
# each row in bounding boxes will contain 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
bboxes = get_bb_info(train_df, image_id, ["x_min", "y_min", "x_max", "y_max", "class_id"])

# %% --------------------
transformed = augmentor(image=raw_img_arr, bboxes=bboxes)

# %% --------------------
# visualize the transformed image w/ bounding boxes
bounding_box_plotter(transformed["image"], image_id, transformed["bboxes"],
                     get_label_2_color_dict())
