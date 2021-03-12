# %% --------------------
# pip install pycocotools

import numpy as np
from map_boxes import mean_average_precision_for_boxes
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.kaggle_utils import extract_dimension_df


# %% --------------------
class ListToCOCODataset:
    # to handle raw preds from model
    # to handle gt from target dictionary in data loader
    def __init__(self):
        pass


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
        14: "pulmonary fibrosis"
    }

    return id_to_label_map


# %% --------------------
def normalize_bb(pred_df, dimension_df, dimension_df_height_col, dimension_df_width_col):
    # get height and width data
    image_dimensions = extract_dimension_df(dimension_df)

    for image_id in pred_df["image_id"].unique():
        # normalize x_min
        pred_df.loc[pred_df["image_id"] == image_id, "x_min"] /= image_dimensions.loc[
            image_id, dimension_df_width_col]

        # normalize x_max
        pred_df.loc[pred_df["image_id"] == image_id, "x_max"] /= image_dimensions.loc[
            image_id, dimension_df_width_col]

        # normalize y_min
        pred_df.loc[pred_df["image_id"] == image_id, "y_min"] /= image_dimensions.loc[
            image_id, dimension_df_height_col]

        # normalize y_max
        pred_df.loc[pred_df["image_id"] == image_id, "y_max"] /= image_dimensions.loc[
            image_id, dimension_df_height_col]

    # normalization checker to check if values are b/w 0 and 1
    if not ((pred_df["x_min"] > 1).any() or (pred_df["x_min"] < 0).any() or
            (pred_df["x_max"] > 1).any() or (pred_df["x_max"] < 0).any() or
            (pred_df["y_min"] > 1).any() or (pred_df["y_min"] < 0).any() or
            (pred_df["y_max"] > 1).any() or (pred_df["y_max"] < 0).any()):
        return pred_df
    else:
        raise ValueError("Normalization failed since normalized values were not in range of 0-1")


# %% --------------------
# https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes
# https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes/blob/4fea46a6153efa72632a968b4bc61292da1fd38f/map_boxes/__init__.py#L93
def compute_mAP_zfturbo(normalized_gt_df, normalized_pred_df, id_to_label, gt_class_id="class_id",
                        pred_class_id="label"):
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
                                                                   verbose=True)
    return mean_ap, average_precisions
