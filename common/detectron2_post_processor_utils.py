import numpy as np
import pandas as pd

from common.kaggle_utils import extract_dimension_df
from common.utilities import merge_bb_nms, merge_bb_wbf


# %% --------------------
def post_process_conf_filter_nms(df, confidence_threshold, nms_iou_threshold):
    # filter rows based on confidence threshold
    object_detection_prediction_filtered = df[df["confidence_score"] >= confidence_threshold].copy()

    # apply nms on confidence filtered
    img_id_arr = []
    x_min_arr = []
    y_min_arr = []
    x_max_arr = []
    y_max_arr = []
    label_arr = []
    score_arr = []

    for image_id in sorted(object_detection_prediction_filtered["image_id"].unique()):
        bb_df = object_detection_prediction_filtered[
            object_detection_prediction_filtered["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]

        bb_df = bb_df.to_numpy()
        nms_bb = merge_bb_nms(bb_df, 0, 1, 2, 3, 4, iou_thr=nms_iou_threshold, scores_col=5)

        for i in range(len(nms_bb)):
            img_id_arr.append(image_id)
            x_min_arr.append(nms_bb[i][0])
            y_min_arr.append(nms_bb[i][1])
            x_max_arr.append(nms_bb[i][2])
            y_max_arr.append(nms_bb[i][3])
            label_arr.append(nms_bb[i][4])
            score_arr.append(nms_bb[i][5])

    # convert the predictions into pandas dataframe
    final_predictions = pd.DataFrame({
        "image_id": img_id_arr,
        "x_min": x_min_arr,
        "y_min": y_min_arr,
        "x_max": x_max_arr,
        "y_max": y_max_arr,
        "label": label_arr,
        "confidence_score": score_arr
    })

    return final_predictions


# %% --------------------
def post_process_conf_filter_wbf(df, confidence_threshold, wbf_iou_threshold,
                                 original_dimensions_df):
    # filter rows based on confidence threshold
    object_detection_prediction_filtered = df[df["confidence_score"] >= confidence_threshold].copy()

    # get the dimensions from data frame containing repeated rows of image id
    original_dimensions_df_aggregated = extract_dimension_df(original_dimensions_df)

    # merge bb based on WBF
    img_id_arr = []
    x_min_arr = []
    y_min_arr = []
    x_max_arr = []
    y_max_arr = []
    label_arr = []
    score_arr = []

    for image_id in sorted(object_detection_prediction_filtered["image_id"].unique()):
        bb_df = object_detection_prediction_filtered[
            object_detection_prediction_filtered["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "label", "confidence_score"]]

        bb_df = bb_df.to_numpy()

        t_width, t_height = original_dimensions_df_aggregated.loc[
            image_id, ["transformed_width", "transformed_height"]].values

        wbf_bb = merge_bb_wbf(t_width, t_height, bb_df, 4, 0, 1, 2, 3, iou_thr=wbf_iou_threshold,
                              scores_col=5)

        for i in range(len(wbf_bb)):
            img_id_arr.append(image_id)
            x_min_arr.append(wbf_bb[i][0])
            y_min_arr.append(wbf_bb[i][1])
            x_max_arr.append(wbf_bb[i][2])
            y_max_arr.append(wbf_bb[i][3])
            label_arr.append(wbf_bb[i][4])
            score_arr.append(wbf_bb[i][5])

    # convert the predictions into pandas dataframe
    final_predictions = pd.DataFrame({
        "image_id": img_id_arr,
        "x_min": x_min_arr,
        "y_min": y_min_arr,
        "x_max": x_max_arr,
        "y_max": y_max_arr,
        "label": label_arr,
        "confidence_score": score_arr
    })

    return final_predictions


# %% --------------------
def binary_and_object_detection_processing(binary_pred_df, object_pred_df, low_thr, high_thr):
    """binary_pred_df should contain all original image_ids and 1 row = 1 image"""
    # get unique image id in binary detection dataframe
    binary_uids = binary_pred_df["image_id"].unique()

    # image_id, x_min, y_min, x_max, y_max, label, confidence_score
    sub = []

    for binary_uid in binary_uids:
        # get binary prediction probability
        prob = binary_pred_df[binary_pred_df["image_id"] == binary_uid]["probabilities"].item()

        if prob < low_thr:
            # Less chance of having any disease, so No findings class
            sub.append({"image_id": binary_uid, "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1,
                        "label": 14, "confidence_score": 1})

        elif low_thr <= prob < high_thr:
            data = object_pred_df[object_pred_df["image_id"] == binary_uid]

            if len(data) != 0:
                data = data.to_dict('records')
                # add original object detection output
                sub.extend(data)

            # More chance of having disease but also append no findings class
            sub.append({"image_id": binary_uid, "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1,
                        "label": 14, "confidence_score": prob})

        elif prob >= high_thr:
            # Good chance of having any disease so believe in object detection model outputs
            data = object_pred_df[object_pred_df["image_id"] == binary_uid]

            if len(data) != 0:
                data = data.to_dict('records')
                # add original object detection output
                sub.extend(data)

        else:
            raise ValueError('Prediction must be from [0-1]')

    # add logic here that any missing binary id becomes no findings class
    missing_id = np.setdiff1d(binary_uids, object_pred_df["image_id"].unique())

    for id in missing_id:
        sub.append({"image_id": id, "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1,
                    "label": 14, "confidence_score": 1})

    sub_df = pd.DataFrame(sub, columns=["image_id", "x_min", "y_min", "x_max", "y_max", "label",
                                        "confidence_score"])
    sub_df = sub_df.sort_values(["image_id"]).reset_index(drop=True)

    return sub_df
