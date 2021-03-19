import os
import zipfile

import albumentations
import pandas as pd
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

from common.utilities import dicom2array


# %% --------------------
def extract_files(source, destination, delete_source=False):
    # extract file in destination
    with zipfile.ZipFile(source, 'r') as zipref:
        zipref.extractall(destination)

    if delete_source:
        # delete source file
        os.remove(source)


# %% --------------------
test_original_dimension = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
                                      "Workspace/vbd_cxr/7_POC/test_original_dimension_1024_sample.csv")

# %% --------------------
api = KaggleApi()
api.authenticate()


# , "00bcb82818ea83d6a86df241762cd7d0",
#                  "013893a5fa90241c65c3efcdbdd2cec1", "01ee6e560f083255a630c41bba779405"

# %% --------------------
def resize_image_test(img_arr, smallest_max_size):
    # create resize transform pipeline
    transform = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=smallest_max_size, always_apply=True)
    ])

    return transform(image=img_arr)


# %% --------------------
def resize_image_w_h(img_arr, width, height):
    # create resize transform pipeline
    transform = albumentations.Compose([
        albumentations.Resize(width=width, height=height, always_apply=True)
    ])
    return transform(image=img_arr)


# %% --------------------
for image_id in sorted(test_original_dimension["image_id"].unique()[:3]):
    # download DICOM image using image_id
    api.competition_download_file('vinbigdata-chest-xray-abnormalities-detection',
                                  f'test/{image_id}.dicom',
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
    print("\n1024 IMAGE")

    # %% --------------------
    # transform image using albumentations
    transformed = resize_image_test(raw_img_arr, smallest_max_size=1024)

    print("Numpy shape")
    print(transformed["image"].shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(transformed["image"])
    print("PIL shape")
    print(im.size)

    # %% --------------------
    print("\nCORRECT UPSCALED IMAGE")

    o_width, o_height = test_original_dimension[test_original_dimension["image_id"] == image_id][
        ["original_width", "original_height"]].values[0]
    print(f"\nStored in CSV as wxh:{o_width}x{o_height}")

    # %% --------------------
    # transform image using albumentations
    transformed2 = resize_image_w_h(transformed["image"], width=o_width, height=o_height)

    print("Numpy shape")
    print(transformed2["image"].shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(transformed2["image"])
    print("PIL shape")
    print(im.size)

    # %% --------------------
    print("\nOLD TECHNIQUE UPSCALED IMAGE")

    o_width, o_height = test_original_dimension[test_original_dimension["image_id"] == image_id][
        ["original_width", "original_height"]].values[0]
    print(f"\nStored in CSV as wxh:{o_height}x{o_width}")

    # %% --------------------
    # transform image using albumentations
    transformed2 = resize_image_w_h(transformed["image"], width=o_height, height=o_width)

    print("Numpy shape")
    print(transformed2["image"].shape)

    print("Numpy shape")
    print(transformed2["image"].shape)

    # %% --------------------
    # convert Numpy to image
    im = Image.fromarray(transformed2["image"])
    print("PIL shape")
    print(im.size)

    print("-" * 25)
