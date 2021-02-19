# %% --------------------
import os
import warnings
from datetime import datetime

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from matplotlib import patches
from pydicom.pixel_data_handlers.util import apply_voi_lut

# %% --------------------
warnings.filterwarnings('ignore')

# directories
dataset_dir = '../input/vinbigdata-chest-xray-abnormalities-detection'
output_dir = './'


# %% --------------------
# https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # normalizing the data?
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


# %% --------------------

# https://www.kaggle.com/trungthanhnguyen0502/eda-vinbigdata-chest-x-ray-abnormalities#1.-Dicom-to-Numpy-array
# we are plotting the image by not resizing the image but resizing the plot
def plot_img(img, title, cmap='gray'):
    plt.figure(figsize=(7, 7))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


# %% --------------------

def view_DICOM_metadata(img_id, test_data=False):
    if test_data:
        dicom = pydicom.read_file(f'{dataset_dir}/test/{img_id}.dicom')
    else:
        dicom = pydicom.read_file(f'{dataset_dir}/train/{img_id}.dicom')
    print(dicom)


# %% --------------------

# read csv data
train_df = pd.read_csv(f'{dataset_dir}/train.csv')
train_df.head()

# %% --------------------

# select only those rows which have bounding boxes
finding_df = train_df[train_df['class_name'] != 'No finding']
finding_df.head()

# %% --------------------
img_ids = finding_df['image_id'].unique()

# %% --------------------
len(img_ids)

# %% --------------------

view_DICOM_metadata(img_ids[0])

# %% --------------------
shortlisted_img_ids = img_ids[:10]
og_imgs = [dicom2array(f'{dataset_dir}/train/{path}.dicom') for path in shortlisted_img_ids]

for img_as_arr, img_id in zip(og_imgs, shortlisted_img_ids):
    plot_img(img_as_arr, img_id)

# %% --------------------
shortlisted_img_ids

# %% --------------------
finding_df[finding_df["image_id"] == "9a5094b2563a1ef3ff50dc5c7ff71345"]


# %% --------------------
def get_bb_info(df, img_id):
    bounding_boxes_info = df.loc[
        df["image_id"] == img_id, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"]]

    bboxes = []
    for _, row in bounding_boxes_info.astype(np.int16).iterrows():
        bboxes.append(list(row))

    return bboxes


# %% --------------------

# class 14:"No finding"
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
               13: ("Pulmonary fibrosis", "#e75480")}


# %% --------------------

# code to plot image with bounding boxes
def bounding_box_plotter(img_as_arr, img_id, bounding_boxes_info):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1])

    # plot the image
    plt.imshow(img_as_arr, cmap="gray")
    plt.title(img_id)

    # add the bounding boxes
    for row in bounding_boxes_info:
        # each row contains 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
        xmin = row[0]
        xmax = row[2]
        ymin = row[1]
        ymax = row[3]

        width = xmax - xmin
        height = ymax - ymin

        # assign different color to different classes of objects
        edgecolor = label2color[row[4]][1]
        ax.annotate(label2color[row[4]][0], xy=(xmax - 40, ymin + 20))

        # add bounding boxes to the image
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)

    plt.show()


# %% --------------------
# plotting the original image with the original bounding boxes
for img_as_arr, img_id in zip(og_imgs, shortlisted_img_ids):
    bounding_boxes_info = get_bb_info(finding_df, img_id)
    bounding_box_plotter(img_as_arr, img_id, bounding_boxes_info)


# %% --------------------

# https://www.kaggle.com/bjoernholzhauer/eda-dicom-reading-vinbigdata-chest-x-ray#7.-Creating-fast-to-read-shelve-file
def resize_image(df, img_arr, image_id):
    # create resize transform pipeline
    transform = albumentations.Compose([
        # Faster RCNN needs minimum 800x800 dimension (https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/)
        albumentations.SmallestMaxSize(max_size=1024, always_apply=True)
    ], bbox_params=albumentations.BboxParams(format='pascal_voc'))

    # each row in bounding boxes will contain 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    bboxes = get_bb_info(df, image_id)

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed


# %% --------------------

for img_as_arr, img_id in zip(og_imgs, shortlisted_img_ids):
    transformed = resize_image(finding_df, img_as_arr, img_id)
    bounding_box_plotter(transformed["image"], img_id, transformed["bboxes"])
    print(
        f"Original Dimension: {img_as_arr.shape}\nTransformed Dimension: {transformed['image'].shape}")

# %% --------------------
train_df.head()

# %% --------------------

values = {'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1}
train_df = train_df.fillna(value=values)

# %% --------------------
train_df.head()

# %% --------------------

# create train directory
transformed_trained_dir = f"{output_dir}/transformed_data/train"
os.makedirs(transformed_trained_dir, exist_ok=True)

# create test directory
transformed_test_dir = f"{output_dir}/transformed_data/test"
os.makedirs(transformed_test_dir, exist_ok=True)

# %% --------------------
train_df.head()

# %% --------------------
train_df["image_id"].unique()


# %% --------------------


def get_info(df, image_id, columns_arr):
    info_row = df.loc[df["image_id"] == image_id, columns_arr]

    info = []
    for _, row in info_row.iterrows():
        info.append(list(row))

    return info


# %% --------------------

# https://www.kaggle.com/bjoernholzhauer/eda-dicom-reading-vinbigdata-chest-x-ray#7.-Creating-fast-to-read-shelve-file
def generic_resize_image(df, image_id, image_path, transform_bb=True):
    # convert dicom to array
    img_arr = dicom2array(image_path)

    # training data
    if transform_bb:
        # create resize transform pipeline
        transform = albumentations.Compose([
            # Faster RCNN needs minimum 800x800 dimension (https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/)
            albumentations.SmallestMaxSize(max_size=1024, always_apply=True)
        ], bbox_params=albumentations.BboxParams(format='pascal_voc'))

        columns = ["x_min", "y_min", "x_max", "y_max", "class_id", "class_name", "rad_id"]
        bboxes = get_info(df, image_id, columns)

        transformed = transform(image=img_arr, bboxes=bboxes)
    else:
        # create resize transform pipeline
        transform = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=1024, always_apply=True)
        ])

        transformed = transform(image=img_arr)

    return transformed, img_arr.shape[0], img_arr.shape[1]


# %% --------------------

# list for resized train data
image_id = []
x_min = []
y_min = []
x_max = []
y_max = []
class_id = []
class_name = []
rad_id = []
original_width = []
original_height = []
transformed_width = []
transformed_height = []

# start time
start = datetime.now()

# conversion
for img_id in train_df["image_id"].unique():
    transformed, width, height = generic_resize_image(train_df, img_id,
                                                      f"{dataset_dir}/train/{img_id}.dicom")

    # save image array as jpeg
    im = Image.fromarray(transformed["image"])
    im.save(transformed_trained_dir + f"/{img_id}.jpeg")

    for i in range(len(transformed["bboxes"])):
        image_id.append(img_id)
        # each row contains "x_min", "y_min", "x_max","y_max","class_id","class_name","rad_id"
        x_min.append(transformed["bboxes"][i][0])
        y_min.append(transformed["bboxes"][i][1])
        x_max.append(transformed["bboxes"][i][2])
        y_max.append(transformed["bboxes"][i][3])
        class_id.append(transformed["bboxes"][i][4])
        class_name.append(transformed["bboxes"][i][5])
        rad_id.append(transformed["bboxes"][i][6])
        original_width.append(width)
        original_height.append(height)
        # when using size we get width x height
        transformed_width.append(im.size[0])
        transformed_height.append(im.size[1])

updated_csv = pd.DataFrame({
    "image_id": image_id,
    "x_min": x_min,
    "y_min": y_min,
    "x_max": x_max,
    "y_max": y_max,
    "class_id": class_id,
    "class_name": class_name,
    "rad_id": rad_id,
    "original_width": original_width,
    "original_height": original_height,
    "transformed_width": transformed_width,
    "transformed_height": transformed_height
})
updated_csv.to_csv(f"{transformed_trained_dir}/transformed_train.csv", index=False)

# end time
print("End time:" + str(datetime.now() - start))

# %% --------------------

verifier_csv = pd.read_csv(f"{transformed_trained_dir}/transformed_train.csv")

# %% --------------------
shortlisted_img_ids

# %% --------------------
for img_id in shortlisted_img_ids:
    bounding_boxes_info = get_bb_info(verifier_csv, img_id)

    # read image as array
    im = Image.open(transformed_trained_dir + f"/{img_id}.jpeg")
    bounding_box_plotter(im, img_id, bounding_boxes_info)

# %% --------------------
test_ids = []
for f in sorted(os.listdir(f"{dataset_dir}/test/")):
    test_ids.append(f)

# %% --------------------

# start time
start = datetime.now()

original_width = []
original_height = []
transformed_width = []
transformed_height = []
image_id = []

# conversion
for file_name in test_ids:
    transformed, width, height = generic_resize_image(None, None, f"{dataset_dir}/test/{file_name}",
                                                      False)

    image_id.append(file_name[:-6])
    original_width.append(width)
    original_height.append(height)

    # save image array as jpeg
    im = Image.fromarray(transformed["image"])
    im.save(transformed_test_dir + f"/{file_name[:-6]}.jpeg")

    # when using size we get width x height
    transformed_width.append(im.size[0])
    transformed_height.append(im.size[1])

test_csv = pd.DataFrame({
    "image_id": image_id,
    "original_width": original_width,
    "original_height": original_height,
    "transformed_width": transformed_width,
    "transformed_height": transformed_height
})
test_csv.to_csv(f"{transformed_test_dir}/test_original_dimension.csv", index=False)

# end time
print("End time:" + str(datetime.now() - start))

# %% --------------------
view_DICOM_metadata("009bc039326338823ca3aa84381f17f1", test_data=True)
