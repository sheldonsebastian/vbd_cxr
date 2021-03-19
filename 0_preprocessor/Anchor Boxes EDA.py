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

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------
# DIRECTORIES
MERGED_DIR = os.getenv("MERGED_DIR")

# %% --------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common.utilities import bounding_box_plotter, get_label_2_color_dict, get_bb_info
from PIL import Image

# %% --------------------
# find out median box height, width and square for training data
train_df = pd.read_csv(
    "/1_merger/wbf_merged/archives/object_detection/train_df.csv")

# %% --------------------
train_df["height"] = train_df["y_max"] - train_df["y_min"]
train_df["width"] = train_df["x_max"] - train_df["x_min"]
train_df["area"] = train_df["height"] * train_df["width"]
train_df["square_root_area"] = np.sqrt(train_df["area"])

# %% --------------------
bin_size = 20

# # %% --------------------
# train_df["height"].plot.hist(by="height", bins=bin_size)
# plt.title("Histogram Height")
# plt.tight_layout()
# plt.xticks(np.arange(0, int(max(train_df["height"])) + bin_size, bin_size), rotation=90)
# plt.show()
#
# # %% --------------------
# train_df["width"].plot.hist(by="width", bins=bin_size)
# plt.title("Histogram Width")
# plt.tight_layout()
# plt.xticks(np.arange(0, int(max(train_df["width"])) + bin_size, bin_size), rotation=90)
# plt.show()
#
# # %% --------------------
# train_df["square_root_area"].plot.hist(by="square_root_area", bins=bin_size)
# plt.title("Histogram Square Root Area")
# plt.tight_layout()
# plt.xticks(np.arange(0, int(max(train_df["square_root_area"])) + bin_size, bin_size), rotation=90)
# plt.show()

# %% --------------------
# Individual Disease
disease_id = 13

# %% --------------------
# get histogram distribution of height
train_df[train_df["class_id"] == disease_id]["height"].plot.hist(by="height",
                                                                 bins=bin_size)
plt.title(f"Histogram Height for disease {disease_id}")
plt.tight_layout()
plt.xticks(np.arange(0, int(
    max(train_df[train_df["class_id"] == disease_id]["height"])) + bin_size, bin_size),
           rotation=90)
plt.show()

# get histogram distribution of width
train_df[train_df["class_id"] == disease_id]["width"].plot.hist(by="width",
                                                                bins=bin_size)
plt.title(f"Histogram Width of disease {disease_id}")
plt.tight_layout()
plt.xticks(np.arange(0, int(
    max(train_df[train_df["class_id"] == disease_id]["width"])) + bin_size, bin_size),
           rotation=90)
plt.show()

# get histogram distribution of area
train_df[train_df["class_id"] == disease_id]["square_root_area"].plot.hist(by="square_root_area",
                                                                           bins=bin_size)
plt.title(f"Histogram Square Root Area of disease {disease_id}")
plt.tight_layout()
plt.xticks(np.arange(0, int(
    max(train_df[train_df["class_id"] == disease_id]["square_root_area"])) + bin_size, bin_size),
           rotation=90)
plt.show()

# plot 3 sample bounding boxes for each disease
image_ids = sorted(train_df[train_df["class_id"] == disease_id]["image_id"].unique()[:10])
for id in image_ids:
    img = Image.open(
        f"D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/9_data/512/transformed_data/train/{id}.jpeg ")
    img_as_arr = np.asarray(img)
    bounding_boxes_info = get_bb_info(train_df, id,
                                      ['x_min', 'y_min', 'x_max', 'y_max', "class_id"])
    bounding_box_plotter(img_as_arr, id, bounding_boxes_info, get_label_2_color_dict(),
                         label_annotations=True)
