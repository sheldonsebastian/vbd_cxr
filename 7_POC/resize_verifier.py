import os
import zipfile

import pandas as pd
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

from common.utilities import bounding_box_plotter, dicom2array, get_bb_info, \
    resize_image, get_label_2_color_dict, resize_image_w_h


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
train_df.loc[train_df["class_id"] == 14, "x_min"] = 0
train_df.loc[train_df["class_id"] == 14, "y_min"] = 0
train_df.loc[train_df["class_id"] == 14, "x_max"] = 1
train_df.loc[train_df["class_id"] == 14, "y_max"] = 1

train_transformed_df = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                                   "Workspace/vbd_cxr/9_data/1024/transformed_data/train"
                                   "/transformed_train.csv")

# %% --------------------
api = KaggleApi()
api.authenticate()

# , "00bcb82818ea83d6a86df241762cd7d0",
#                  "013893a5fa90241c65c3efcdbdd2cec1", "01ee6e560f083255a630c41bba779405"

# %% --------------------
for image_id in sorted(train_transformed_df["image_id"].unique()[:3]):
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
    print("\nORIGINAL IMAGE")

    # %% --------------------
    raw_img_arr = dicom2array(f"D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/"
                              f"vbd_cxr/7_POC/DICOMs/{image_id}.dicom")
    print("Numpy shape")
    print(raw_img_arr.shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(raw_img_arr)
    print("PIL shape")
    print(im.size)

    # %% --------------------
    raw_img_bb_info = get_bb_info(train_df, image_id,
                                  ["x_min", "y_min", "x_max", "y_max", "class_id"])

    # %% --------------------
    # view image as DICOM w/ bounding boxes
    bounding_box_plotter(raw_img_arr, image_id, raw_img_bb_info, get_label_2_color_dict(), False)

    # %% --------------------
    print("\n512 IMAGE")

    # %% --------------------
    # transform image using albumentations
    transformed = resize_image(train_df, raw_img_arr, image_id,
                               ["x_min", "y_min", "x_max", "y_max", "class_id"],
                               smallest_max_size=1024)

    print("Numpy shape")
    print(transformed["image"].shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(transformed["image"])
    print("PIL shape")
    print(im.size)

    # %% --------------------
    # visualize the transformed image w/ bounding boxes
    bounding_box_plotter(transformed["image"], image_id, transformed["bboxes"],
                         get_label_2_color_dict(), False)

    # %% --------------------
    print("\nCORRECTED UPSCALED IMAGE")

    o_width, o_height = train_transformed_df[train_transformed_df["image_id"] == image_id][
        ["original_width", "original_height"]].values[0]
    print(f"\nStored in CSV as wxh:{o_width}x{o_height}")

    # %% --------------------
    # transform image using albumentations
    transformed2 = resize_image_w_h(train_transformed_df, transformed["image"], image_id,
                                    ["x_min", "y_min", "x_max", "y_max", "class_id"],
                                    width=o_width, height=o_height)

    print("Numpy shape")
    print(transformed2["image"].shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(transformed2["image"])
    print("PIL shape")
    print(im.size)

    # %% --------------------
    # visualize the transformed image w/ bounding boxes
    bounding_box_plotter(transformed2["image"], image_id, transformed2["bboxes"],
                         get_label_2_color_dict(), False)

    # %% --------------------
    print("\nBEFORE FIX UPSCALED IMAGE")

    o_width, o_height = train_transformed_df[train_transformed_df["image_id"] == image_id][
        ["original_width", "original_height"]].values[0]
    print(f"\nStored in CSV as wxh:{o_height}x{o_width}")

    # %% --------------------
    # transform image using albumentations
    transformed2 = resize_image_w_h(train_transformed_df, transformed["image"], image_id,
                                    ["x_min", "y_min", "x_max", "y_max", "class_id"],
                                    width=o_height, height=o_width)

    print("Numpy shape")
    print(transformed2["image"].shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(transformed2["image"])
    print("PIL shape")
    print(im.size)

    # %% --------------------
    # visualize the transformed image w/ bounding boxes
    bounding_box_plotter(transformed2["image"], image_id, transformed2["bboxes"],
                         get_label_2_color_dict(), False)

    print("-" * 25)
