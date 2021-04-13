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

# %% --------------------IMPORTS
# https://www.kaggle.com/corochann/vinbigdata-detectron2-train

import random
import numpy as np
import torch
from common.detectron_config_manager import Flags

# %% --------------------set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and VARIABLES
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")

# %% --------------------
flag_path = DETECTRON2_DIR + "/faster_rcnn/configurations/train/v1.yaml"

# %% --------------------
flags_dict = {
    "debug": False,
    "outdir": "results/v9",
    "imgdir_name": "vinbigdata-chest-xray-resized-png-256x256",
    "split_mode": "valid20",
    "iter": 10000,
    "roi_batch_size_per_image": 512,
    "eval_period": 1000,
    "lr_scheduler_name": "WarmupCosineLR",
    "base_lr": 0.001,
    "num_workers": 4,
    "aug_kwargs": {
        "HorizontalFlip": {"p": 0.5},
        "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
        "RandomBrightnessContrast": {"p": 0.5}
    }
}

# %% --------------------
flags = Flags().update(flags_dict)
print(flags)

# %% --------------------
flags.save_yaml(flag_path)

# %% --------------------
flags = Flags()
# %% --------------------
print(flags.get_configs())

# %% --------------------
flags = flags.load_yaml(flag_path)

# %% --------------------
print(flags.get_configs())
