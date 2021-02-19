# %% --------------------
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
from matplotlib import patches

# %% --------------------
BASE_TRAIN_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
                 "Workspace/vbd_cxr/transformed_data/train"

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
# https://www.kaggle.com/quillio/vindr-bounding-box-fusion/notebook
# iou_thr=0 means, if we have iou>0 for same labels, then those BB will be fused together
def fuse_multiple_bounding_boxes(image_id, df, iou_thr=0):
    '''This function uses ZFTurbos implementation of Weighted Boxes Fusion'''
    # read image
    im = Image.open(f"{BASE_TRAIN_DIR}/{image_id}.jpeg")

    # get dimension of image
    # https://stackoverflow.com/a/34533139
    (width, height) = im.size
    dimensions = [width, height, width, height]

    # select data from csv
    record = df[df["image_id"] == image_id]

    # get bounding boxes for the image_id
    bboxes = record[["x_min", "y_min", "x_max", "y_max"]].values

    # normalize the bounding boxes so they are between 0 and 1
    normalized = [bboxes / dimensions]
    labels = [record["class_id"].values]

    # each BB has equal confidence score
    scores = [[1] * record.shape[0]]

    # we are considering only 1 model with weight=1
    weights = [1]

    # skip bounding boxes which have confidence score < 0
    skip_box_thr = 0

    # ZFTurbo library
    boxes, scores, labels = weighted_boxes_fusion(normalized,
                                                  scores,
                                                  labels,
                                                  weights=weights,
                                                  iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)

    # convert the fused bounding box co-ordinates back to non-normalized values
    fused_boxes = boxes * dimensions

    return fused_boxes, labels


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
        fused_boxes, class_ids = fuse_multiple_bounding_boxes(image_id, finding_df, iou_threshold)

        for row, class_id in zip(fused_boxes, class_ids):
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
# code to plot image with bounding boxes for fusion comparison
# https://www.kaggle.com/quillio/vindr-bounding-box-fusion/notebook
# bounding_boxes_before = 'x_min', 'y_min', 'x_max', 'y_max', "class_id", "rad_id"
# bounding_boxes_after = 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
def bounding_box_plotter_before_after(img_as_arr, img_id, bounding_boxes_before,
                                      bounding_boxes_after):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    for bb, img, ax, title in zip([bounding_boxes_before, bounding_boxes_after],
                                  [img_as_arr, img_as_arr], [ax1, ax2], ["Before", "After"]):

        # add the bounding boxes
        for row in bb:
            # each row contains 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
            xmin = row[0]
            xmax = row[2]
            ymin = row[1]
            ymax = row[3]

            width = xmax - xmin
            height = ymax - ymin

            # assign different color to different classes of objects
            edgecolor = label2color[row[4]][1]
            ax.annotate(label2color[row[4]][0], xy=(xmax - 40, ymin + 20))

            # add radiologist if Before
            label_bb = str(label2color[row[4]][0]) + "::" + str(
                row[5]) if title == "Before" else str(label2color[row[4]][0])

            # add bounding boxes to the image
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor,
                                     facecolor='none', label=label_bb)

            ax.add_patch(rect)
            ax.legend()

        # plot the image
        ax.imshow(img_as_arr, cmap="gray")
        ax.set_title(title + "::" + img_id)

    fig.set_size_inches(22, 16)
    plt.show()


# %% --------------------

for img_id in sorted(combined_df_iou_0["image_id"].unique())[:20]:
    bounding_boxes_info_before = get_bb_info(train_df, img_id,
                                             ['x_min', 'y_min', 'x_max', 'y_max', "class_id",
                                              "rad_id"])
    bounding_boxes_info_after = get_bb_info(combined_df_iou_0, img_id,
                                            ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

    # read image as array
    im = Image.open(BASE_TRAIN_DIR + f"/{img_id}.jpeg")
    bounding_box_plotter_before_after(im, img_id, bounding_boxes_info_before,
                                      bounding_boxes_info_after)

# %% --------------------
for img_id in sorted(combined_df_iou_0_3["image_id"].unique())[:20]:
    bounding_boxes_info_before = get_bb_info(train_df, img_id,
                                             ['x_min', 'y_min', 'x_max', 'y_max', "class_id",
                                              "rad_id"])
    bounding_boxes_info_after = get_bb_info(combined_df_iou_0_3, img_id,
                                            ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

    # read image as array
    im = Image.open(BASE_TRAIN_DIR + f"/{img_id}.jpeg")
    bounding_box_plotter_before_after(im, img_id, bounding_boxes_info_before,
                                      bounding_boxes_info_after)

# %% --------------------

for img_id in sorted(combined_df_iou_0_6["image_id"].unique())[:20]:
    bounding_boxes_info_before = get_bb_info(train_df, img_id,
                                             ['x_min', 'y_min', 'x_max', 'y_max', "class_id",
                                              "rad_id"])
    bounding_boxes_info_after = get_bb_info(combined_df_iou_0_6, img_id,
                                            ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

    # read image as array
    im = Image.open(BASE_TRAIN_DIR + f"/{img_id}.jpeg")
    bounding_box_plotter_before_after(im, img_id, bounding_boxes_info_before,
                                      bounding_boxes_info_after)

# %% --------------------
for img_id in sorted(combined_df_iou_0_9["image_id"].unique())[:20]:
    bounding_boxes_info_before = get_bb_info(train_df, img_id,
                                             ['x_min', 'y_min', 'x_max', 'y_max', "class_id",
                                              "rad_id"])
    bounding_boxes_info_after = get_bb_info(combined_df_iou_0_9, img_id,
                                            ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

    # read image as array
    im = Image.open(BASE_TRAIN_DIR + f"/{img_id}.jpeg")
    bounding_box_plotter_before_after(im, img_id, bounding_boxes_info_before,
                                      bounding_boxes_info_after)

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

combined_df_iou_0.to_csv("./fused_train_0.csv", index=False)
combined_df_iou_0_3.to_csv("./fused_train_0_3.csv", index=False)
combined_df_iou_0_6.to_csv("./fused_train_0_6.csv", index=False)
combined_df_iou_0_9.to_csv("./fused_train_0_9.csv", index=False)
