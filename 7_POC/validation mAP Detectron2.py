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
import itertools

import random
import numpy as np
import torch
from detectron2.utils.logger import setup_logger
from common.detectron2_utils import get_train_detectron_dataset, build_simple_dataloader
from detectron2.data import DatasetCatalog, MetadataCatalog
from common.detectron2_evaluator import Detectron2_ZFTurbo_mAP
from detectron2.structures import Boxes

# %% --------------------set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and VARIABLES
IMAGE_DIR = os.getenv("IMAGE_DIR")
# MERGED_DIR contains GT dataframes
MERGED_DIR = os.getenv("MERGED_DIR")
SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR")
TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR")
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
WORKERS = int(os.getenv("NUM_WORKERS"))

# %% --------------------
# DYNAMIC
train_gt_dataframe = MERGED_DIR + "/wbf_merged/90_percent_train/object_detection/90_percent" \
                                  "/train_df_0.csv"
val_gt_dataframe = MERGED_DIR + "/wbf_merged/90_percent_train/object_detection/10_percent" \
                                "/holdout_df_0.csv"

# %% -------------------- SETUP LOGGER
setup_logger(output=DETECTRON2_DIR + "faster_rcnn/outputs/current/")

# %% --------------------DATASET
# lambda is anonymous function
# train dataset
DatasetCatalog.register("train", lambda: get_train_detectron_dataset(IMAGE_DIR, train_gt_dataframe))
MetadataCatalog.get("train").set(
    thing_classes=["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                   "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                   "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                   "Pulmonary fibrosis"])

# validation dataset
DatasetCatalog.register("validation",
                        lambda: get_train_detectron_dataset(IMAGE_DIR, val_gt_dataframe))
MetadataCatalog.get("validation").set(
    thing_classes=["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                   "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                   "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                   "Pulmonary fibrosis"])


# %% --------------------ABC: visualize raw dataset
# dataset_sample_viewer("validation", 3)

# %% --------------------AUGMENTATIONS
# view_sample_augmentations("train", {
#     "HorizontalFlip": {"p": 0.5},
#     "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
#     "RandomBrightnessContrast": {"p": 0.5}
# }
#                           , 2, 4)


# %% --------------------
# testing evaluator
def get_all_inputs_outputs():
    for batch in build_simple_dataloader(["validation"], 3):
        gt_data = batch
        pred_data = gt_data.copy()
        # MOCK predictions
        for data in pred_data:
            instances = data["instances"]
            boxes = instances.gt_boxes.tensor.numpy()
            boxes = boxes.tolist()
            classes = instances.gt_classes
            scores = torch.Tensor([1] * len(classes))

            data["instances"].set("pred_boxes", Boxes(torch.Tensor(boxes)))
            data["instances"].set("pred_classes", classes)
            data["instances"].set("scores", scores)

        yield gt_data, pred_data


# create evaluation instance
evaluator = Detectron2_ZFTurbo_mAP(0.15, 0.40)

evaluator.reset()

for inputs, outputs in itertools.islice(get_all_inputs_outputs(), 2):
    evaluator.process(inputs, outputs)

eval_results = evaluator.evaluate()
print(eval_results)
