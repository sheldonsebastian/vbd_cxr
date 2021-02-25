# %% --------------------
import ast

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
import torchvision
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
from matplotlib import patches
from pydicom.pixel_data_handlers.util import apply_voi_lut


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
def get_image_as_array(img_path):
    im_array = Image.open(img_path)
    return im_array


# %% --------------------
# https://www.kaggle.com/trungthanhnguyen0502/eda-vinbigdata-chest-x-ray-abnormalities#1.-Dicom-to-Numpy-array
# we are plotting the image by not resizing the image but resizing the plot
def plot_img(img_as_arr, title, cmap='gray'):
    plt.figure(figsize=(7, 7))
    plt.imshow(img_as_arr, cmap=cmap)
    plt.title(title)
    plt.show()


# %% --------------------
def view_DICOM_metadata(image_path, dicom_file_name):
    dicom = pydicom.read_file(f'{image_path}/{dicom_file_name}')
    return dicom


# %% --------------------
# code to plot image with bounding boxes
# bounding_boxes_info as [{'x_min':.., 'y_min':.., 'x_max':.., 'y_max':.., "class_id":..},{...}]
def bounding_box_plotter(img_as_arr, img_id, bounding_boxes_info, label2color):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()

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
def get_bb_info(df, img_id, columns):
    bounding_boxes_info = df.loc[df["image_id"] == img_id, columns]

    bboxes = []
    for _, row in bounding_boxes_info.iterrows():
        bboxes.append(list(row))

    # TODO convert the return type to be dataframe with column names
    return np.array(bboxes)


# %% --------------------
# https://www.kaggle.com/bjoernholzhauer/eda-dicom-reading-vinbigdata-chest-x-ray#7.-Creating-fast-to-read-shelve-file
def resize_image(df, img_arr, image_id, columns):
    # create resize transform pipeline
    transform = albumentations.Compose([
        # Faster RCNN needs minimum 800x800 dimension (
        # https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/)
        albumentations.SmallestMaxSize(max_size=1024, always_apply=True)
    ], bbox_params=albumentations.BboxParams(format='pascal_voc'))

    # each row in bounding boxes will contain 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    bboxes = get_bb_info(df, image_id, columns)

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed


# %% --------------------
def read_text_literal(file_path):
    file_content_string = open(file_path, "r").read()

    # convert file to python data structures
    python_data = ast.literal_eval(file_content_string)

    return python_data


# %% --------------------
# reverse operation for SmallestMaxSize Albumentation Transformation
def convert_bb_smallest_max_scale(bb_df, original_width, original_height, transformed_width,
                                  transformed_height):
    """bb_df contains x_min, y_min, x_max, y_max which is in original_height and original_width
    dimensions """
    original_min = min(original_height, original_width)
    transformed_min = min(transformed_height, transformed_width)

    scale_factor = original_min / transformed_min

    return bb_df / scale_factor


# %% --------------------
# reverse operation for Resize Albumentation Transformation
def convert_bb_resize_scale(bb_df, original_width, original_height, transformed_width,
                            transformed_height):
    """bb_df contains x_min, y_min, x_max, y_max which is in original_height and original_width
    dimensions """
    x_axis_scale_factor = original_height / transformed_height
    y_axis_scale_factor = original_width / transformed_width

    bb_df_res = bb_df.copy()

    bb_df_res[:, [0, 2]] /= x_axis_scale_factor
    bb_df_res[:, [1, 3]] /= y_axis_scale_factor

    return bb_df_res


# %% --------------------
def bounding_box_plotter_side_to_side(img_as_arr, img_id, bounding_boxes_left,
                                      bounding_boxes_right, left_title, right_title, label2color):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    for bb, img, ax, title in zip([bounding_boxes_left, bounding_boxes_right],
                                  [img_as_arr, img_as_arr], [ax1, ax2], [left_title, right_title]):

        # add the bounding boxes
        for row in bb:
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

            # add radiologist if Before
            label_bb = str(label2color[row[4]][0])

            # add bounding boxes to the image
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor,
                                     facecolor='none', label=label_bb)

            ax.add_patch(rect)
            ax.legend()

        # plot the image
        ax.imshow(img_as_arr, cmap="gray")
        ax.set_title(title + "::" + img_id)

    fig.set_size_inches(20, 10)
    plt.show()


# %% --------------------
def filter_df_based_on_confidence_threshold(df, confidence_col, confidence_threshold):
    # retain predictions which are greater than or equal to confidence threshold
    filtered_df = df[df[confidence_col] >= confidence_threshold]

    return filtered_df


# %% --------------------
# https://www.kaggle.com/quillio/vindr-bounding-box-fusion/notebook
# iou_thr=0 means, if we have iou>0 for same labels, then those BB will be fused together
def merge_bb_wbf(im_x_axis_size, im_y_axis_size, bb_df, label_column, x_min_col, y_min_col,
                 x_max_col, y_max_col, iou_thr=0, scores_col=None):
    """This function uses ZFTurbos implementation of Weighted Boxes Fusion"""
    dimensions = [im_x_axis_size, im_y_axis_size, im_x_axis_size, im_y_axis_size]

    # get bounding boxes for the image_id
    bboxes = bb_df[:, [x_min_col, y_min_col, x_max_col, y_max_col]]

    # normalize the bounding boxes so they are between 0 and 1
    normalized = [bboxes / dimensions]
    labels = [bb_df[:, label_column]]

    if scores_col is None:
        # each BB has equal confidence score
        scores = [[1] * bb_df.shape[0]]
    else:
        scores = [bb_df[:, scores_col]]

    # we are considering only 1 model with weight=1
    weights = [1]

    # skip bounding boxes which have confidence score < 0
    skip_box_thr = 0

    # ZFTurbo library
    boxes, scores, labels = weighted_boxes_fusion(normalized,
                                                  scores,
                                                  labels,
                                                  weights=weights,
                                                  iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)

    # convert the fused bounding box co-ordinates back to non-normalized values
    fused_boxes = boxes * dimensions

    return np.c_[fused_boxes, labels, scores]


# %% --------------------
# NMS DOES NOT CARE ABOUT LABELS, THUS LABELS FOR DIFFERENT CLASSES WILL ALSO BE CONSIDERED
# DURING BOX MERGING
def merge_bb_nms(bb_arr, x_min_col, y_min_col, x_max_col, y_max_col, iou_thr=0, scores_col=None):
    if scores_col is None:
        # each BB has equal confidence score
        scores = [[1] * bb_arr.shape[0]]
    else:
        scores = bb_arr[:, scores_col]

    boxes = bb_arr[:, [x_min_col, y_min_col, x_max_col, y_max_col]]

    # convert all to tensors
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)

    # indices to keep
    keep_indices = torchvision.ops.nms(boxes, scores, iou_thr)

    return bb_arr[keep_indices, :].reshape(len(keep_indices), bb_arr.shape[1])
