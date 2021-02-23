# %% --------------------
# handle imports
import pandas as pd

from common.utilities import plot_img, get_image_as_array, get_bb_info, bounding_box_plotter, \
    dicom2array, convert_bb_smallest_max_scale

# %% --------------------
train_dir_path = "/transformed_data/train"

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
train_data = pd.read_csv(
    "/transformed_data/transformed_train.csv")

# %% --------------------
train_data.head()

# %% --------------------
train_data.columns

# %% --------------------
# img = "9a5094b2563a1ef3ff50dc5c7ff71345"
img = "001d127bad87592efe45a5c7678f8b8d"

# %% --------------------
img_array = get_image_as_array(f"{train_dir_path}/{img}.jpeg")

# %% --------------------
# plot original image
plot_img(img_array, "Original")

# %% --------------------
# get bounding box info
img_bb_info = get_bb_info(train_data, img, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])

# %% --------------------
# plot image with bounding boxes
bounding_box_plotter(img_array, img, img_bb_info, label2color)

# %% --------------------
agg_train = train_data.groupby(["image_id"]).aggregate(
    {"original_width": "first", "original_height": "first", "transformed_width": "first",
     "transformed_height": "first"})

# %% --------------------
# original size (2336, 2080)
o_width, o_height = agg_train.loc[img, ["original_width", "original_height"]]

# %% --------------------
# transformed size (1024, 1150)
t_width, t_height = agg_train.loc[img, ["transformed_width", "transformed_height"]]

# %% --------------------
img_bb_info[:, [0, 1, 2, 3]] = convert_bb_smallest_max_scale(img_bb_info[:, [0, 1, 2, 3]], t_width,
                                                             t_height, o_width, o_height)

# %% --------------------
# verify up scaling
og_img_arr = dicom2array(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm "
    f"Workspace/VBD_CXR/0_preprocessor/{img}.dicom",
    voi_lut=True, fix_monochrome=True)

# %% --------------------
plot_img(og_img_arr, "Og Dicom")

# %% --------------------
bounding_box_plotter(og_img_arr, img, img_bb_info, label2color)

# %% --------------------
print(img_bb_info)
