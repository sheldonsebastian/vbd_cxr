# %% --------------------
import operator

from map_boxes import mean_average_precision_for_boxes

from common.kaggle_utils import extract_dimension_df


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
def normalize_bb_512(df_og):
    # make a copy
    df = df_og.copy(deep=True)

    for image_id in df["image_id"].unique():
        # normalize x_min
        df.loc[df["image_id"] == image_id, "x_min"] /= 512

        # normalize x_max
        df.loc[df["image_id"] == image_id, "x_max"] /= 512

        # normalize y_min
        df.loc[df["image_id"] == image_id, "y_min"] /= 512

        # normalize y_max
        df.loc[df["image_id"] == image_id, "y_max"] /= 512

    # normalization checker to check if values are b/w 0 and 1
    if not ((df["x_min"] > 1).any() or (df["x_min"] < 0).any() or
            (df["x_max"] > 1).any() or (df["x_max"] < 0).any() or
            (df["y_min"] > 1).any() or (df["y_min"] < 0).any() or
            (df["y_max"] > 1).any() or (df["y_max"] < 0).any()):
        return df
    else:
        raise ValueError("Normalization failed since normalized values were not in range of 0-1")


# %% --------------------
def normalize_bb_with_dimensions(df_og, dimension_df, dimension_df_height_col,
                                 dimension_df_width_col):
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
