import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion


# %% --------------------
def check_normalization(normalized_bb, technique, o_id):
    normalized_bb_temp = normalized_bb.values
    # normalization checker to check if values are b/w 0 and 1
    if not ((normalized_bb_temp[:, 0] > 1).any() or (normalized_bb_temp[:, 0] < 0).any() or
            (normalized_bb_temp[:, 1] > 1).any() or (normalized_bb_temp[:, 1] < 0).any() or
            (normalized_bb_temp[:, 2] > 1).any() or (normalized_bb_temp[:, 2] < 0).any() or
            (normalized_bb_temp[:, 3] > 1).any() or (normalized_bb_temp[:, 3] < 0).any()):
        return normalized_bb
    else:
        print(
            f"Normalization failed since normalized values were not in range of 0-1 for {technique} and ID: {o_id}")
        return None


# %% --------------------
def ensemble_object_detectors(list_object_detection_predictions, original_image_df, height_col,
                              width_col, iou_thr, weights_list):
    """this function uses zfturbos implementation of weighted boxes fusion"""
    ensembled_outputs = []

    # iterate through
    original_image_ids = original_image_df["image_id"].unique()

    # perform wbf for each image
    for o_id in original_image_ids:
        # get original image width and height
        width, height = original_image_df.loc[
            original_image_df["image_id"] == o_id, [width_col, height_col]].values[0]

        dimensions = [width, height, width, height]
        normalized_arr = []
        labels_arr = []
        scores_arr = []
        weights_arr = []

        # iterate through each prediction list and get the image id
        for prediction, technique, weights in zip(list_object_detection_predictions,
                                                  ["Faster RCNN", "YOLOv5"],
                                                  weights_list):
            image_data = prediction[prediction["image_id"] == o_id]

            # get bounding boxes for the image_id
            bboxes = image_data.loc[:, ["x_min", "y_min", "x_max", "y_max"]]

            # normalize the bounding boxes so they are between 0 and 1
            normalized = np.divide(bboxes, dimensions)

            normalized = check_normalization(normalized, technique, o_id)

            labels = image_data.loc[:, "label"]
            scores = image_data.loc[:, "confidence_score"]

            normalized_arr.append(normalized.values)
            labels_arr.append(labels.values)
            scores_arr.append(scores.values)
            weights_arr.append(weights)

        # zfturbo library
        boxes_merged, scores_merged, labels_merged = weighted_boxes_fusion(normalized_arr,
                                                                           scores_arr,
                                                                           labels_arr,
                                                                           weights=weights_arr,
                                                                           iou_thr=iou_thr,
                                                                           skip_box_thr=0)

        # convert the fused bounding box co-ordinates back to non-normalized values
        fused_boxes = boxes_merged * dimensions

        for merged_box, merged_score, merged_label in zip(fused_boxes, scores_merged,
                                                          labels_merged):
            ensembled_outputs.append({
                "image_id": o_id,
                "x_min": merged_box[0],
                "y_min": merged_box[1],
                "x_max": merged_box[2],
                "y_max": merged_box[3],
                "label": merged_label,
                "confidence_score": merged_score
            })

    # convert to Dataframe
    ensembled_outputs_df = pd.DataFrame(ensembled_outputs)

    # handle missing image ids
    missing_ids = np.setdiff1d(original_image_ids, ensembled_outputs_df["image_id"].unique())
    missing_data = []

    for missing_id in missing_ids:
        missing_data.append({
            "image_id": missing_id,
            "x_min": 0,
            "y_min": 0,
            "x_max": 1,
            "y_max": 1,
            "label": 14,
            "confidence_score": 1
        })

    missing_data_df = pd.DataFrame(missing_data)

    ensembled_outputs_df = ensembled_outputs_df.append(missing_data_df)

    return ensembled_outputs_df
