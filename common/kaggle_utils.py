# %% --------------------
import pandas as pd

from common.utilities import get_bb_info, convert_bb_smallest_max_scale


# %% --------------------
def extract_dimension_df(df):
    agg_df = df.groupby(["image_id"]).aggregate(
        {"original_width": "first", "original_height": "first", "transformed_width": "first",
         "transformed_height": "first"}).copy()
    return agg_df


# %% --------------------
def up_scaler(predicted_df, df_with_original_dimension):
    # get the dimensions from data frame containing repeated rows of image id
    extracted_dimension_df = extract_dimension_df(df_with_original_dimension)

    image_id_arr = []
    x_min_arr = []
    y_min_arr = []
    x_max_arr = []
    y_max_arr = []
    label_arr = []
    score_arr = []

    for img in predicted_df["image_id"].unique():
        o_width, o_height, t_width, t_height = extracted_dimension_df.loc[
            img, ["original_width", "original_height", "transformed_width", "transformed_height"]]

        bounding_boxes_info = get_bb_info(predicted_df, img,
                                          ["x_min", "y_min", "x_max", "y_max", "label",
                                           "confidence_score"])

        # upscale the predicted bounding boxes based on original scale and visualize it
        bounding_boxes_info[:, [0, 1, 2, 3]] = convert_bb_smallest_max_scale(
            bounding_boxes_info[:, [0, 1, 2, 3]], t_width, t_height, o_width, o_height)

        for i in range(len(bounding_boxes_info)):
            # class 14 is no findings class and should be represented as 0,0,1,1 and score=1
            if bounding_boxes_info[i][4] == 14:
                image_id_arr.append(img)
                x_min_arr.append(0)
                y_min_arr.append(0)
                x_max_arr.append(1)
                y_max_arr.append(1)
                label_arr.append(bounding_boxes_info[i][4])
                score_arr.append(1)
            else:
                image_id_arr.append(img)
                x_min_arr.append(bounding_boxes_info[i][0])
                y_min_arr.append(bounding_boxes_info[i][1])
                x_max_arr.append(bounding_boxes_info[i][2])
                y_max_arr.append(bounding_boxes_info[i][3])
                label_arr.append(bounding_boxes_info[i][4])
                score_arr.append(bounding_boxes_info[i][5])

    upscaled_data = pd.DataFrame(
        {"image_id": image_id_arr, "x_min": x_min_arr, "y_min": y_min_arr, "x_max": x_max_arr,
         "y_max": y_max_arr, "label": label_arr, "confidence_score": score_arr})

    return upscaled_data


# %% --------------------
# https://www.kaggle.com/pestipeti/vinbigdata-fasterrcnn-pytorch-inference#kln-125
def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            int(j[0]), j[1], j[2][0], j[2][1], j[2][2], j[2][3]))

    return " ".join(pred_strings)


# %% --------------------
def submission_file_creator(df, x_min_col, y_min_col, x_max_col, y_max_col, label_col, score_col):
    # img_id, label confidence bb label confidence bb label confidence bb
    image_id_arr = []
    predictions_arr = []

    for image_id in df["image_id"].unique():
        image_id_arr.append(image_id)
        labels = df[df["image_id"] == image_id][label_col]
        scores = df[df["image_id"] == image_id][score_col]
        boxes = df[df["image_id"] == image_id][
            [x_min_col, y_min_col, x_max_col, y_max_col]].to_numpy()

        predictions_arr.append(format_prediction_string(labels, boxes, scores))

    return pd.DataFrame({"image_id": image_id_arr, "PredictionString": predictions_arr})
