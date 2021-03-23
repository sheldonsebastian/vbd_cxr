# https://detectron2.readthedocs.io/en/v0.2.1/tutorials/evaluation.html
import collections
import copy

import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from map_boxes import mean_average_precision_for_boxes  # ZFTurbo library


# this evaluator will be called by DefaultTrainer, which is single process after every epoch
class Detectron2_ZFTurbo_mAP(DatasetEvaluator):

    def __init__(self, conf_thr):
        self.cpu_device = torch.device("cpu")
        self.conf_thr = conf_thr

        # list to store values with column indices as ['ImageID', 'LabelName', 'XMin', 'XMax',
        # 'YMin', 'YMax']
        self.normalized_gt = []
        self.normalized_pred = []

        self.id2label = {0: "Aortic enlargement", 1: "Atelectasis", 2: "Calcification",
                         3: "Cardiomegaly", 4: "Consolidation", 5: "ILD", 6: "Infiltration",
                         7: "Lung Opacity", 8: "Nodule/Mass", 9: "Other lesion",
                         10: "Pleural effusion", 11: "Pleural thickening", 12: "Pneumothorax",
                         13: "Pulmonary fibrosis"}

    def reset(self):
        self.normalized_gt = []
        self.normalized_pred = []

    def process(self, inputs, outputs):
        """
        :inputs: these are input images based on batch size
        :outputs: these are predicted bb for images based on batch size
        """
        inputs_copy = copy.deepcopy(inputs)
        outputs_copy = copy.deepcopy(outputs)

        # iterate through images based on batch size
        for input_instance, output_pred in zip(inputs_copy, outputs_copy):
            # get dimensions of an image for normalizing co-ordinates
            y_axis_dim = input_instance["height"]
            x_axis_dim = input_instance["width"]
            image_id = input_instance["image_id"]

            instances = input_instance["instances"].to(self.cpu_device)
            boxes = instances.gt_boxes.tensor.numpy()
            boxes = boxes.tolist()
            classes = instances.gt_classes.tolist()

            for box, class_id in zip(boxes, classes):
                # normalize when accessing the bboxes
                normalized_box = box / np.array([x_axis_dim, y_axis_dim, x_axis_dim, y_axis_dim])
                class_label = self.id2label[int(class_id)]

                # ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin','YMax']
                self.normalized_gt.append(
                    [image_id, class_label, normalized_box[0], normalized_box[2],
                     normalized_box[1], normalized_box[3]])

            # iterate through predicted boxes and labels
            if "instances" in output_pred:
                instances = output_pred["instances"].to(self.cpu_device)
                boxes = instances.pred_boxes.tensor.numpy()
                boxes = boxes.tolist()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()

                for box, score, class_id in zip(boxes, scores, classes):
                    # append predictions only if greater than threshold
                    if score >= self.conf_thr:
                        # normalize when accessing the bboxes
                        normalized_box = box / np.array(
                            [x_axis_dim, y_axis_dim, x_axis_dim, y_axis_dim])
                        class_label = self.id2label[int(class_id)]

                        # ['ImageID', 'LabelName', "Confidence", 'XMin', 'XMax', 'YMin','YMax']
                        self.normalized_pred.append(
                            [image_id, class_label, score, normalized_box[0], normalized_box[2],
                             normalized_box[1], normalized_box[3]])

    def evaluate(self):
        # convert list to numpy array
        if len(self.normalized_gt) > 0:
            np_normalized_gt = np.array(self.normalized_gt).reshape(len(self.normalized_gt), 6)
        else:
            np_normalized_gt = np.empty(shape=(0, 6))

        if len(self.normalized_pred) > 0:
            np_normalized_pred = np.array(self.normalized_pred).reshape(
                len(self.normalized_pred), 7)
        else:
            np_normalized_pred = np.empty(shape=(0, 7))

        # compute mAP
        mean_ap, average_precisions = mean_average_precision_for_boxes(np_normalized_gt,
                                                                       np_normalized_pred,
                                                                       iou_threshold=0.4,
                                                                       exclude_not_in_annotations=True,
                                                                       verbose=True)
        results = collections.OrderedDict()
        results["Aggregate"] = {"Total mAP": mean_ap}
        for k, v in average_precisions.items():
            results[k] = {"mAP": v[0], "count": v[1]}

        return results
