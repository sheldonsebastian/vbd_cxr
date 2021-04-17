# %% --------------------
import ast

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from matplotlib import patches
from pydicom.pixel_data_handlers.util import apply_voi_lut


# %% --------------------
# https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)

    # voi lut (if available by dicom device) is used to
    # transform raw dicom data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, x-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # scaling data to be between 0 to 255
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


# %% --------------------
def get_image_as_array(img_path):
    im_array = Image.open(img_path)
    return im_array


# %% --------------------
# https://www.kaggle.com/trungthanhnguyen0502/eda-vinbigdata-chest-x-ray-abnormalities#1.-dicom-to-numpy-array
# we are plotting the image by not resizing the image but resizing the plot
def plot_img(img_as_arr, title, cmap='gray'):
    if type(img_as_arr) is torch.Tensor:
        img_as_arr = img_as_arr.permute(1, 2, 0).numpy()

    plt.figure(figsize=(7, 7))
    plt.imshow(img_as_arr, cmap=cmap)
    plt.title(title)
    plt.show()


# %% --------------------
def view_dicom_metadata(image_path, dicom_file_name):
    dicom = pydicom.read_file(f'{image_path}/{dicom_file_name}')
    return dicom


# %% --------------------
# code to plot image with bounding boxes
# bounding_boxes_info as [{'x_min':.., 'y_min':.., 'x_max':.., 'y_max':.., "class_id":..},{...}]
def bounding_box_plotter(img_as_arr, img_id, bounding_boxes_info, label2color,
                         label_annotations=True, save_title_or_plot="plot"):
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

        if label_annotations:
            ax.annotate(label2color[row[4]][0], xy=(xmax - 40, ymin + 20))

        # add bounding boxes to the image
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)
    plt.tight_layout()
    if save_title_or_plot.lower() == "plot":
        plt.show()
        plt.close(fig)
    else:
        plt.savefig(save_title_or_plot)
        plt.close(fig)


# %% --------------------
def get_bb_info(df, img_id, columns):
    bounding_boxes_info = df.loc[df["image_id"] == img_id, columns]

    bboxes = []
    for _, row in bounding_boxes_info.iterrows():
        bboxes.append(list(row))

    return np.array(bboxes)


# %% --------------------
# https://www.kaggle.com/bjoernholzhauer/eda-dicom-reading-vinbigdata-chest-x-ray#7.-creating-fast-to-read-shelve-file
def resize_image(df, img_arr, image_id, columns, smallest_max_size=1024):
    # create resize transform pipeline
    transform = albumentations.Compose([
        # faster rcnn needs minimum 800x800 dimension (
        # https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/)
        albumentations.SmallestMaxSize(max_size=smallest_max_size, always_apply=True)
    ], bbox_params=albumentations.BboxParams(format='pascal_voc'))

    # each row in bounding boxes will contain 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    bboxes = get_bb_info(df, image_id, columns)

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed


# %% --------------------
def resize_image_w_h(df, img_arr, image_id, columns, width, height):
    # create resize transform pipeline
    transform = albumentations.Compose([
        albumentations.Resize(width=width, height=height, always_apply=True)
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
# reverse operation for smallestmaxsize albumentation transformation
def convert_bb_smallest_max_scale(bb_df, original_width, original_height, transformed_width,
                                  transformed_height):
    """bb_df contains x_min, y_min, x_max, y_max which is in original_height and original_width
    dimensions """
    original_min = min(original_height, original_width)
    transformed_min = min(transformed_height, transformed_width)

    scale_factor = original_min / transformed_min

    return bb_df / scale_factor


# %% --------------------
# reverse operation for resize albumentation transformation
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
                                      bounding_boxes_right, left_title, right_title, label2color,
                                      save_title_or_plot="plot"):
    fig, (ax1, ax2) = plt.subplots(1, 2)

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

            # add radiologist if before
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
    plt.tight_layout()

    if save_title_or_plot.lower() == "plot":
        plt.show()
        plt.close(fig)
    else:
        plt.savefig(save_title_or_plot)
        plt.close(fig)


# %% --------------------
def filter_df_based_on_confidence_threshold(df, confidence_col, confidence_threshold):
    # retain predictions which are greater than or equal to confidence threshold
    filtered_df = df[df[confidence_col] >= confidence_threshold]

    return filtered_df


# %% --------------------
# https://www.kaggle.com/quillio/vindr-bounding-box-fusion/notebook
# iou_thr=0 means, if we have iou>0 for same labels, then those bb will be fused together
def merge_bb_wbf(im_x_axis_size, im_y_axis_size, bb_df, label_column, x_min_col, y_min_col,
                 x_max_col, y_max_col, iou_thr, scores_col=None):
    """this function uses zfturbos implementation of weighted boxes fusion"""
    dimensions = [im_x_axis_size, im_y_axis_size, im_x_axis_size, im_y_axis_size]

    # get bounding boxes for the image_id
    bboxes = bb_df[:, [x_min_col, y_min_col, x_max_col, y_max_col]]

    # normalize the bounding boxes so they are between 0 and 1
    normalized = [np.divide(bboxes, dimensions)]
    labels = [bb_df[:, label_column]]

    if scores_col is None:
        # each bb has equal confidence score
        scores = [[1] * bb_df.shape[0]]
    else:
        scores = [bb_df[:, scores_col]]

    # we are considering only 1 model with weight=1
    weights = [1]

    # skip bounding boxes which have confidence score < 0
    skip_box_thr = 0

    # zfturbo library
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
def merge_bb_nms(bb_arr, x_min_col, y_min_col, x_max_col, y_max_col, class_col, iou_thr,
                 scores_col=None):
    result = []
    for class_label in list(set(bb_arr[:, [class_col]].flatten())):
        bb_arr_class = bb_arr[bb_arr[:, class_col] == class_label]

        # https://stackoverflow.com/questions/23911875/select-certain-rows-condition-met-but-only-some-columns-in-python-numpy
        boxes = bb_arr_class[:, [x_min_col, y_min_col, x_max_col, y_max_col]]

        if scores_col is None:
            # each bb has equal confidence score
            scores = np.ones(shape=(1, bb_arr_class.shape[0])).flatten()
        else:
            scores = bb_arr_class[:, scores_col]

        # convert all to tensors
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)

        keep_indices = torchvision.ops.nms(boxes, scores, iou_thr)

        # add the filtered boxes to results
        for idx in keep_indices:
            result.append(bb_arr_class[idx, :])

    return np.array(result)


# %% --------------------
def multi_label_split_based_on_percentage(df, n_splits, test_percentage, unique_id_column,
                                          target_column, seed):
    """
    :param df: The dataframe in which 1 row = 1 class for multi-label classification
    :param n_splits: How to split the dataframe
    :param test_percentage: how much should be the test percentage split?
    :param unique_id_column: the column which uniquely identifies the dataframe
    :param target_column: the classes column (multi labels). It has to be numeric
    :param seed: 42
    :return: train and validation dataframe same as df but with fold columns
    """

    # store unique ids
    unique_ids = df[unique_id_column].unique()

    # find unique classes
    unique_classes = df[target_column].unique()

    # convert the target column into multi label format
    one_hot_labels = []
    for uid in unique_ids:
        classes = df[df[unique_id_column] == uid][target_column].values
        x = np.eye(len(unique_classes))[classes.astype(int)].sum(0)
        one_hot_labels.append(x)

    # https://github.com/trent-b/iterative-stratification#multilabelstratifiedshufflesplit
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, train_size=1 - test_percentage,
                                            test_size=test_percentage, random_state=seed)

    # create train and validation splits
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # X is unique id
    for fold, (train_index, val_index) in enumerate(msss.split(unique_ids, one_hot_labels)):
        train_data = df[df[unique_id_column].isin(unique_ids[train_index])].copy(deep=True)
        val_data = df[df[unique_id_column].isin(unique_ids[val_index])].copy(deep=True)

        train_data["fold"] = fold
        val_data["fold"] = fold

        train_df = train_df.append(train_data, ignore_index=True)
        val_df = val_df.append(val_data, ignore_index=True)

    return train_df, val_df


# %% --------------------
def multi_label_split_based_on_fold(df, n_splits, unique_id_column, target_column, seed):
    """
    :param df: The dataframe in which 1 row = 1 class for multi-label classification
    :param n_splits: How to split the dataframe
    :param unique_id_column: the column which uniquely identifies the dataframe
    :param target_column: the classes column (multi labels). It has to be numeric
    :return: train and validation dataframe same as df but with fold columns
    """

    # store unique ids
    unique_ids = df[unique_id_column].unique()

    # find unique classes
    unique_classes = df[target_column].unique()

    # convert the target column into multi label format
    one_hot_labels = []
    for uid in unique_ids:
        classes = df[df[unique_id_column] == uid][target_column].values
        x = np.eye(len(unique_classes))[classes.astype(int)].sum(0)
        one_hot_labels.append(x)

    # https://github.com/trent-b/iterative-stratification#multilabelstratifiedkfold
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # create train and validation splits
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # X is unique id
    for fold, (train_index, val_index) in enumerate(mskf.split(unique_ids, one_hot_labels)):
        train_data = df[df[unique_id_column].isin(unique_ids[train_index])].copy(deep=True)
        val_data = df[df[unique_id_column].isin(unique_ids[val_index])].copy(deep=True)

        train_data["fold"] = fold
        val_data["fold"] = fold

        train_df = train_df.append(train_data, ignore_index=True)
        val_df = val_df.append(val_data, ignore_index=True)

    return train_df, val_df


# %% --------------------
def display_fold_distribution(train_df, val_df, target_column, color="blue"):
    """
    :train_df : contains the train data with fold column
    :val_df : contains the validation data with fold column
    :target_column : specifies the target column. It has to be numeric
    """

    n_splits = len(np.union1d(train_df["fold"].unique(), val_df["fold"].unique()))

    # Visualize the splits
    for i in range(n_splits):
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

        for dataframe, type_of, ax in zip([train_df, val_df], ["Train", "Validation"], [ax1, ax2]):
            # check distribution of data
            val_counts = dataframe[dataframe["fold"] == i][
                target_column].value_counts().reset_index()

            val_counts.sort_values(["index"])["class_id"].plot(kind='bar', color=color, ax=ax,
                                                               xticks=val_counts["index"])
            ax.set_title(f"{type_of} Fold {i}")

        plt.show()


# %% --------------------Non Normalizing the image
# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3?u=coolcucumber94
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensors): Tensors of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for tensor in tensors:
            for t, m, s in zip(tensor, self.mean, self.std):
                # inplace operation
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)

        return tensors


# %% --------------------
# https://vitalflux.com/python-draw-confusion-matrix-matplotlib/
def confusion_matrix_plotter(conf_matrix, title):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title, fontsize=18)
    plt.show()


# %% --------------------
# https://www.codespeedy.com/how-to-plot-roc-curve-using-sklearn-library-in-python/
def plot_roc_cur(fper, tper, title):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()


# %% --------------------
def get_label_2_color_dict():
    label2color = {
        0: ("Aortic enlargement", "#2a52be"),
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
        14: ("No findings", "#FFFFFF"),
    }

    return label2color


# %% --------------------
def extract_image_id_from_batch_using_dataset(targets, dataset):
    image_ids = []

    for t in targets:
        image_ids.append(dataset.get_image_id_using_index(t["image_id"].item()))

    return image_ids
