# %% --------------------
# pip install pycocotools

import numpy as np
import pandas as pd
import operator
from map_boxes import mean_average_precision_for_boxes
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.kaggle_utils import extract_dimension_df


# %% --------------------
# https://www.kaggle.com/pestipeti/competition-metric-map-0-4
class DataFrameToCOCODataset:
    def __init__(self, df, id_to_label_map, image_id_col, x_min_col, y_min_col, x_max_col,
                 y_max_col, label_col, confidence_col=None):
        """Confidence Col should be passed only for prediction dataframe"""
        self.df = df
        self.x_min_col = x_min_col
        self.y_min_col = y_min_col
        self.x_max_col = x_max_col
        self.y_max_col = y_max_col
        self.image_id_col = image_id_col
        self.label_col = label_col
        self.confidence_col = confidence_col
        self.id_to_label_map = id_to_label_map

    def get_coco_dataset(self):
        # check if label column contains 0 or not
        if 0 in self.df[self.label_col].unique():
            raise ValueError(
                'DataFrame Labels should not contain 0. 0 is reserved for background class')

        if 0 in set(self.id_to_label_map.values()):
            raise ValueError(
                'DataFrame Labels should not contain 0. 0 is reserved for background class')

        # prepare the dataframe in coco dataset format
        categories = []
        for label_id, label_name in self.id_to_label_map.items():
            categories.append({"id": label_id, "name": label_name, "supercategory": "none"})

        images = []
        annot = []

        k = 0
        for idx, image_id in enumerate(self.df[self.image_id_col].unique()):
            images.append({"id": idx, "file_name": image_id})

            for _, row in self.df[self.df[self.image_id_col] == image_id].iterrows():
                prepped_dict = {
                    "id": k,
                    "image_id": idx,
                    "category_id": int(row[self.label_col]),
                    "bbox": np.array([
                        int(row[self.x_min_col]),
                        int(row[self.y_min_col]),
                        int(row[self.x_max_col]),
                        int(row[self.y_max_col])]
                    ),
                    "segmentation": [],
                    "ignore": 0,
                    "area": (int(row[self.x_max_col]) - int(row[self.x_min_col])) * (
                            int(row[self.y_max_col]) - int(row[self.y_min_col])),
                    "iscrowd": 0,
                }

                # add confidence score only for predictions
                if self.confidence_col is not None:
                    prepped_dict["score"] = row[self.confidence_col]
                k += 1
                annot.append(prepped_dict)

        dataset = {
            "images": images,
            "categories": categories,
            "annotations": annot
        }

        coco_ds = COCO()
        coco_ds.dataset = dataset
        coco_ds.createIndex()

        return coco_ds


# %% --------------------
def get_map(gt_obj, pred_obj, iou_threshold):
    """gt_obj and pred_obj are objects of DataFrameToCOCODataset or ListToCOCODataset"""
    imgIds = sorted(gt_obj.getImgIds())

    cocoEval = COCOeval(gt_obj, pred_obj, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.useCats = True
    cocoEval.params.iouType = "bbox"
    cocoEval.params.iouThrs = np.array([iou_threshold])

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[0]


# %% --------------------
def get_id_to_label_mAP():
    id_to_label_map = {
        1: "aortic enlargement",
        2: "atelectasis",
        3: "calcification",
        4: "cardiomegaly",
        5: "consolidation",
        6: "ild",
        7: "infiltration",
        8: "lung opacity",
        9: "nodule/mass",
        10: "other lesion",
        11: "pleural effusion",
        12: "pleural thickening",
        13: "pneumothorax",
        14: "pulmonary fibrosis",
        15: "No Findings class"
    }

    return id_to_label_map


# %% --------------------
def normalize_bb(df_og, dimension_df, dimension_df_height_col, dimension_df_width_col):
    # make a copy
    df = df_og.copy(deep=True)

    # get height and width data
    image_dimensions = extract_dimension_df(dimension_df)

    for image_id in df["image_id"].unique():
        # normalize x_min
        df.loc[df["image_id"] == image_id, "x_min"] /= image_dimensions.loc[
            image_id, dimension_df_width_col]

        # normalize x_max
        df.loc[df["image_id"] == image_id, "x_max"] /= image_dimensions.loc[
            image_id, dimension_df_width_col]

        # normalize y_min
        df.loc[df["image_id"] == image_id, "y_min"] /= image_dimensions.loc[
            image_id, dimension_df_height_col]

        # normalize y_max
        df.loc[df["image_id"] == image_id, "y_max"] /= image_dimensions.loc[
            image_id, dimension_df_height_col]

    # normalization checker to check if values are b/w 0 and 1
    if not ((df["x_min"] > 1).any() or (df["x_min"] < 0).any() or
            (df["x_max"] > 1).any() or (df["x_max"] < 0).any() or
            (df["y_min"] > 1).any() or (df["y_min"] < 0).any() or
            (df["y_max"] > 1).any() or (df["y_max"] < 0).any()):
        return df
    else:
        raise ValueError("Normalization failed since normalized values were not in range of 0-1")


# %% --------------------
# https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes
# https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes/blob/4fea46a6153efa72632a968b4bc61292da1fd38f/map_boxes/__init__.py#L93
def zfturbo_compute_mAP(normalized_gt_df, normalized_pred_df, id_to_label, gt_class_id="class_id",
                        pred_class_id="label", verbose=False):
    """
    :gt: is ground truth dataframe. The co-ordinates are normalized based on image height and
    width using normalize_bb()
    :pred: is a prediction dataframe. The co-ordinates are normalized based on image height and
    width using normalize_bb()
    :id_to_label: maps the ids in gt and pred to string labels
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """
    # create copies
    normalized_gt_df_copy = normalized_gt_df.copy(deep=True)
    normalized_pred_df_copy = normalized_pred_df.copy(deep=True)

    # replace numeric ids to string labels
    # https://stackoverflow.com/questions/22100130/pandas-replace-multiple-values-one-column
    normalized_gt_df_copy[gt_class_id] = normalized_gt_df_copy[gt_class_id].astype(int).map(
        id_to_label)
    normalized_pred_df_copy[pred_class_id] = normalized_pred_df_copy[pred_class_id].astype(int).map(
        id_to_label)

    # convert dataframe to numpy array format as required by package
    normalized_gt_df_np = normalized_gt_df_copy[
        ["image_id", gt_class_id, "x_min", "x_max", "y_min", "y_max"]].values

    normalized_pred_df_np = normalized_pred_df_copy[
        ["image_id", pred_class_id, "confidence_score", "x_min", "x_max", "y_min", "y_max"]].values

    # compute mAP
    mean_ap, average_precisions = mean_average_precision_for_boxes(normalized_gt_df_np,
                                                                   normalized_pred_df_np,
                                                                   iou_threshold=0.4,
                                                                   exclude_not_in_annotations=True,
                                                                   verbose=verbose)

    return mean_ap, sorted(average_precisions.items(), key=operator.itemgetter(1), reverse=True)


# %% --------------------
class ZFTurbo_MAP_TRAINING:
    """Wrapper class for ZFTurbo mAP used during training"""

    def __init__(self, df_dimension, id_to_label_map):
        self.true_image_id_arr = []
        self.true_label_arr = []
        self.true_x_min_arr = []
        self.true_x_max_arr = []
        self.true_y_min_arr = []
        self.true_y_max_arr = []

        self.pred_image_id_arr = []
        self.pred_label_arr = []
        self.pred_confd_arr = []
        self.pred_x_min_arr = []
        self.pred_x_max_arr = []
        self.pred_y_min_arr = []
        self.pred_y_max_arr = []

        self.dimensions_df = extract_dimension_df(df_dimension)
        self.id_to_label = id_to_label_map

    def zfturbo_convert_targets_from_dataloader(self, targets, image_ids):
        """
        reads the targets from data loader, reads image_id and extends the arrays
        appropriately.
        """
        for img_id, t in zip(image_ids, targets):

            for box, label in zip(t["boxes"].numpy(), t["labels"].numpy()):
                self.true_image_id_arr.append(img_id)
                self.true_label_arr.append(label)
                self.true_x_min_arr.append(box[0])
                self.true_x_max_arr.append(box[2])
                self.true_y_min_arr.append(box[1])
                self.true_y_max_arr.append(box[3])

    def zfturbo_convert_outputs_from_model(self, outputs, image_ids):
        """
        reads the output from outputs, reads image_id and extends the arrays
        appropriately.
        """
        for img_id, output in zip(image_ids, outputs):

            for box, label, confd_score in zip(output["boxes"].cpu().numpy(),
                                               output["labels"].cpu().numpy(),
                                               output["scores"].cpu().numpy()):
                self.pred_image_id_arr.append(img_id)
                self.pred_label_arr.append(label)
                self.pred_confd_arr.append(confd_score)
                self.pred_x_min_arr.append(box[0])
                self.pred_x_max_arr.append(box[2])
                self.pred_y_min_arr.append(box[1])
                self.pred_y_max_arr.append(box[3])

    def zfturbo_compute_mAP(self):

        train_df = pd.DataFrame(
            {"image_id": self.true_image_id_arr, "class_id": self.true_label_arr,
             "x_min": self.true_x_min_arr, "x_max": self.true_x_max_arr,
             "y_min": self.true_y_min_arr, "y_max": self.true_y_max_arr})

        pred_df = pd.DataFrame(
            {"image_id": self.pred_image_id_arr, "label": self.pred_label_arr,
             "confidence_score": self.pred_confd_arr, "x_min": self.pred_x_min_arr,
             "x_max": self.pred_x_max_arr, "y_min": self.pred_y_min_arr,
             "y_max": self.pred_y_max_arr})

        # normalize
        train_df_normalized = normalize_bb(train_df, self.dimensions_df, "transformed_height",
                                           "transformed_width")
        pred_df_normalized = normalize_bb(pred_df, self.dimensions_df, "transformed_height",
                                          "transformed_width")

        # compute mAP
        mean_ap, average_precisions = zfturbo_compute_mAP(train_df_normalized, pred_df_normalized,
                                                          self.id_to_label,
                                                          gt_class_id="class_id",
                                                          pred_class_id="label", verbose=False)

        return mean_ap, average_precisions
