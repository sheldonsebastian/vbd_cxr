# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
# env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# %% --------------------
# DIRECTORIES
IMAGE_DIR = os.getenv("IMAGE_DIR")
BASE_TRAIN_DIR = IMAGE_DIR

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------START HERE
import pandas as pd
from PIL import Image
from common.utilities import merge_bb_wbf, bounding_box_plotter_side_to_side

# %% --------------------
train_df = pd.read_csv(f"{BASE_TRAIN_DIR}/transformed_train.csv")

# %% --------------------
train_df.head()

# %% --------------------
train_df.loc[train_df.class_id == 14, ["x_max", "y_max"]] = 1

# %% --------------------
train_df.head()

# %% --------------------
label2color = {0: ("Aortic enlargement", "#2a52be"),
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
               14: ("No finding", "#ffffff")
               }

# %% --------------------
# no finding class
no_finding_df = train_df[train_df["class_id"] == 14].copy()

# %% --------------------
no_finding_df.head()

# %% --------------------
len(no_finding_df["image_id"].unique())

# %% --------------------
# remove duplicate no finding class
no_finding_df = no_finding_df.drop_duplicates(subset=["image_id", "x_min", "y_min",
                                                      "x_max", "y_max", "class_id",
                                                      "class_name", "original_width",
                                                      "original_height", "transformed_width",
                                                      "transformed_height"],
                                              ignore_index=True)

# %% --------------------
no_finding_df.head()

# %% --------------------
len(no_finding_df)

# %% --------------------
no_finding_df.columns

# %% --------------------
# delete rad_id column
no_finding_df = no_finding_df.loc[:,
                ['image_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', "original_width",
                 "original_height", "transformed_width", "transformed_height"]]

# %% --------------------
no_finding_df.head()

# %% --------------------
# abnormality class present
finding_df = train_df[train_df["class_id"] != 14].reset_index().copy()

# %% --------------------
len(finding_df["image_id"].unique())


# %% --------------------
def get_bb_info(df, img_id, columns):
    bounding_boxes_info = df.loc[df["image_id"] == img_id, columns]

    bboxes = []
    for _, row in bounding_boxes_info.iterrows():
        bboxes.append(list(row))

    return bboxes


# %% --------------------
# aggregate the findings dataframe to get the original and transformed dimensions
finding_df_agg = finding_df.groupby(["image_id"]).aggregate(
    {"original_width": "first", "original_height": "first", "transformed_width": "first",
     "transformed_height": "first"})

# %% --------------------
iou_thresholds = [0, 0.3, 0.6, 0.9]
multiple_merged_bb_iou = []

for iou_threshold in iou_thresholds:

    image_id_arr = []
    x_min_arr = []
    y_min_arr = []
    x_max_arr = []
    y_max_arr = []
    class_id_arr = []
    original_width_arr = []
    original_height_arr = []
    transformed_width_arr = []
    transformed_height_arr = []

    for image_id in finding_df["image_id"].unique():
        width, height = finding_df[finding_df["image_id"] == image_id][
            ["transformed_width", "transformed_height"]].values[0]

        bb_df = finding_df[finding_df["image_id"] == image_id][
            ["x_min", "y_min", "x_max", "y_max", "class_id"]].values

        # merged_info contains boxes[0-3], labels[4], and scores[5]
        merged_info = merge_bb_wbf(width, height, bb_df, 4, 0, 1, 2, 3, iou_thr=iou_threshold)

        for row, class_id in zip(merged_info[:, 0:4], merged_info[:, 4]):
            image_id_arr.append(image_id)

            x_min_arr.append(row[0])
            y_min_arr.append(row[1])
            x_max_arr.append(row[2])
            y_max_arr.append(row[3])

            class_id_arr.append(class_id)

            o_w, o_h, t_w, t_h = finding_df_agg.loc[image_id]
            original_width_arr.append(o_w)
            original_height_arr.append(o_h)
            transformed_width_arr.append(t_w)
            transformed_height_arr.append(t_h)

    fused_findings_df = pd.DataFrame({"image_id": image_id_arr,
                                      "x_min": x_min_arr,
                                      "y_min": y_min_arr,
                                      "x_max": x_max_arr,
                                      "y_max": y_max_arr,
                                      "class_id": class_id_arr,
                                      "original_width": original_width_arr,
                                      "original_height": original_height_arr,
                                      "transformed_width": transformed_width_arr,
                                      "transformed_height": transformed_height_arr})

    multiple_merged_bb_iou.append(fused_findings_df)

# %% --------------------
combined_df_iou_0 = no_finding_df.append(multiple_merged_bb_iou[0])
combined_df_iou_0_3 = no_finding_df.append(multiple_merged_bb_iou[1])
combined_df_iou_0_6 = no_finding_df.append(multiple_merged_bb_iou[2])
combined_df_iou_0_9 = no_finding_df.append(multiple_merged_bb_iou[3])

# %% --------------------
combined_df_iou_0 = combined_df_iou_0.sort_values("image_id").reset_index(drop=True)
combined_df_iou_0_3 = combined_df_iou_0_3.sort_values("image_id").reset_index(drop=True)
combined_df_iou_0_6 = combined_df_iou_0_6.sort_values("image_id").reset_index(drop=True)
combined_df_iou_0_9 = combined_df_iou_0_9.sort_values("image_id").reset_index(drop=True)

# %% --------------------
combined_df_iou_0.head()

# %% --------------------
combined_df_iou_0_3.head()

# %% --------------------
combined_df_iou_0_6.head()

# %% --------------------
combined_df_iou_0_9.head()

# %% --------------------
for img_id in sorted(combined_df_iou_0["image_id"].unique())[:20]:

    for df, iou_thr in zip(
            [combined_df_iou_0, combined_df_iou_0_3, combined_df_iou_0_6, combined_df_iou_0_9],
            ["0", "0_3", "0_6", "0_9"]):
        bounding_boxes_info_before = get_bb_info(train_df, img_id,
                                                 ['x_min', 'y_min', 'x_max', 'y_max', "class_id",
                                                  "rad_id"])
        bounding_boxes_info_after = get_bb_info(df, img_id,
                                                ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

        # read image as array
        im = Image.open(BASE_TRAIN_DIR + f"/{img_id}.jpeg")

        bounding_box_plotter_side_to_side(im, img_id, bounding_boxes_info_before,
                                          bounding_boxes_info_after, f"Before",
                                          f"After @ {iou_thr} IoU:", label2color,
                                          save_title_or_plot=f"D:/GWU/4 Spring 2021/6501 "
                                                             f"Capstone/VBD CXR/PyCharm "
                                                             f"Workspace/vbd_cxr"
                                                             f"/10_annotated_images/1024 merged "
                                                             f"wbf/{img_id}_{iou_thr}.jpeg")

# %% --------------------
os.makedirs("./wbf_merged/100_percent_train/", exist_ok=True)

# %% --------------------
combined_df_iou_0.to_csv("./wbf_merged/100_percent_train/fused_train_0.csv", index=False)
combined_df_iou_0_3.to_csv("./wbf_merged/100_percent_train/fused_train_0_3.csv", index=False)
combined_df_iou_0_6.to_csv("./wbf_merged/100_percent_train/fused_train_0_6.csv", index=False)
combined_df_iou_0_9.to_csv("./wbf_merged/100_percent_train/fused_train_0_9.csv", index=False)
