# %% --------------------
import numpy as np
import pandas as pd

from common.kaggle_utils import extract_dimension_df
from common.utilities import merge_bb_nms, merge_bb_wbf


# %% --------------------
def post_process_conf_filter_nms(df, confidence_threshold, nms_iou_threshold,
                                 extra_normal_ids=None):
    # raise exception if predictions contain 0 i.e. background class
    if 0 in list(df["label"].unique()):
        raise ValueError("In predictions class 0 is reserved for background class")

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

    # ids which do not pass the confidence threshold
    failed_confidence_threshold_ids = df[df["confidence_score"] < confidence_threshold][
        "image_id"].unique()

    # ids which failed confidence and are not present in passed confidence threshold
    # NOTE:: you can use len(pred_boxes) = 0, logic here too
    normal_ids_object_detection = np.setdiff1d(failed_confidence_threshold_ids,
                                               object_detection_prediction_filtered[
                                                   "image_id"].unique())

    if extra_normal_ids is not None:
        normal_ids_object_detection = np.union1d(extra_normal_ids, normal_ids_object_detection)

    # add normal ids to dataframe
    for normal_id in normal_ids_object_detection:
        img_id_arr.append(normal_id)
        x_min_arr.append(0)
        y_min_arr.append(0)
        x_max_arr.append(1)
        y_max_arr.append(1)
        # NOTE: NO FINDINGS CLASS IS CLASS 15
        label_arr.append(15)
        score_arr.append(1)

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
                                 original_dimensions_df, extra_normal_ids=None):
    # raise exception if predictions contain 0 i.e. background class
    if 0 in list(df["label"].unique()):
        raise ValueError("In predictions class 0 is reserved for background class")

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

        # ids which do not pass the confidence threshold
        failed_confidence_threshold_ids = df[df["confidence_score"] < confidence_threshold][
            "image_id"].unique()

    # ids which failed confidence and are not present in passed confidence threshold
    # NOTE:: you can use len(pred_boxes) = 0, logic here too
    normal_ids_object_detection = np.setdiff1d(failed_confidence_threshold_ids,
                                               object_detection_prediction_filtered[
                                                   "image_id"].unique())

    if extra_normal_ids is not None:
        normal_ids_object_detection = np.union1d(extra_normal_ids, normal_ids_object_detection)

    # add normal ids to dataframe
    for normal_id in normal_ids_object_detection:
        img_id_arr.append(normal_id)
        x_min_arr.append(0)
        y_min_arr.append(0)
        x_max_arr.append(1)
        y_max_arr.append(1)
        # NOTE: NO FINDINGS CLASS IS CLASS 15
        label_arr.append(15)
        score_arr.append(1)

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
