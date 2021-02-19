# %% --------------------
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# %% --------------------
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
def print_map(gt_obj, pred_obj, iou_threshold):
    """gt_obj and pred_obj are objects of DataFrameToCOCODataset"""
    imgIds = sorted(gt_obj.getImgIds())

    cocoEval = COCOeval(gt_obj, pred_obj, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.useCats = True
    cocoEval.params.iouType = "bbox"
    cocoEval.params.iouThrs = np.array([iou_threshold])

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.summarize()
