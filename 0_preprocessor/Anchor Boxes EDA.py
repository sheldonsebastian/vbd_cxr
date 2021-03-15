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

# %% --------------------
# find out median box height, width and square for training data
train_df = pd.read_csv(MERGED_DIR + "/wbf_merged/object_detection/train_df_80.csv")

# %% --------------------
train_df["height"] = train_df["y_max"] - train_df["y_min"]
train_df["width"] = train_df["x_max"] - train_df["x_min"]

# %% --------------------
bin_size = 20

# %% --------------------
train_df["height"].plot.hist(by="height", bins=bin_size)
plt.title("Histogram Height")
plt.tight_layout()
plt.xticks(np.arange(0, int(max(train_df["height"])) + bin_size, bin_size), rotation=90)
plt.show()

# %% --------------------
train_df["width"].plot.hist(by="width", bins=bin_size)
plt.title("Histogram Width")
plt.tight_layout()
plt.xticks(np.arange(0, int(max(train_df["width"])) + bin_size, bin_size), rotation=90)
plt.show()

# %% --------------------
