# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
# env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
#           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------
# https://www.kaggle.com/pestipeti/vinbigdata-fasterrcnn-pytorch-inference?scriptVersionId=50935253
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from common.object_detection_models import get_faster_rcnn_model_instance
from common.CustomDatasets import VBD_CXR_FASTER_RCNN_Train
from pathlib import Path

# %% --------------------set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and variables
IMAGE_DIR = os.getenv("IMAGE_DIR")
MERGED_DIR = os.getenv("MERGED_DIR")
SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR")
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")

# %% --------------------DATASET
# NOTE THE DATASET IS GRAY SCALE AND HAS MIN SIDE 512 AND IS NORMALIZED BY FASTER RCNN
# DYNAMIC
validation_data_set = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                                MERGED_DIR + "/wbf_merged"
                                                             "/object_detection"
                                                             "/val_df_20.csv",
                                                albumentation_transformations=None,
                                                clahe_normalization=False,
                                                histogram_normalization=False)
# use 5% holdout set
# holdout set does not have folds
# DYNAMIC
holdout_data_set = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                             MERGED_DIR + "/wbf_merged"
                                                          "/object_detection"
                                                          "/holdout_df.csv",
                                             albumentation_transformations=None,
                                             clahe_normalization=False,
                                             histogram_normalization=False)


# %% --------------------COLLATE FUNCTION required since the image are not of same size
# https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
# https://github.com/pytorch/vision/blob/master/references/detection/utils.py
def collate_fn(batch):
    # https://www.geeksforgeeks.org/zip-in-python/
    # zip(*x) is used to unzip x, where x is iterator
    # thus in the end we will have [(img_id, img_id, ...), (img, img, ...), (target, target, ...)]
    return tuple(zip(*batch))


# %% --------------------
BATCH_SIZE = 8
workers = int(os.getenv("NUM_WORKERS"))

validation_data_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=BATCH_SIZE,
                                                     shuffle=False, num_workers=workers,
                                                     collate_fn=collate_fn)

holdout_data_loader = torch.utils.data.DataLoader(holdout_data_set, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=workers,
                                                  collate_fn=collate_fn)

# %% --------------------
# define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# %% --------------------MODEL INSTANCE
# 15 = 14 classes (abnormalities) + 1 background class (class=0)
# NOTE:: no findings class is ignored
num_classes = 15

# initializing a pretrained model of Faster RCNN with ResNet50-FPN as Backbone
# NOTE:: FASTER RCNN PyTorch implementation performs normalization based on ImageNet
model = get_faster_rcnn_model_instance(num_classes, min_size=512)

# %% --------------------
# DYNAMIC
saved_model_full_path = "/object_detection/faster_rcnn_anchor_sgd.pt"

# load saved model state to appropriate device
saved_model_path = SAVED_MODEL_DIR + saved_model_full_path
model.load_state_dict(
    torch.load(saved_model_path, map_location=torch.device(device))["model_state_dict"])

# %% --------------------
# set model to eval mode, to not disturb the weights
model.eval()

# %% --------------------
# send model to device
model = model.to(device)

# %% --------------------VALIDATION DATA
# make predictions for validation data
print("Validation predictions started")
# start time
start = datetime.now()

# arrays
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
confidence_score_arr = []

with torch.no_grad():
    for images, targets in validation_data_loader:
        # iterate through images and send to device
        images_device = list(image.to(device) for image in images)

        image_ids = []
        for t in targets:
            index = t["image_id"].item()
            image_id = validation_data_loader.dataset.get_image_id_using_index(index)
            # need image_ids to compare with target data
            image_ids.append(image_id)

        # output is list of dictionary [{boxes:tensor([[xmin, ymin, xmax, ymax], [...]],
        # device=cuda), labels:tensor([15, 11, ...], device=cuda), scores:tensor([0.81, 0.92,
        # ...], device=cuda)},{...}]
        outputs = model(images_device)

        for img_id, output in zip(image_ids, outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                image_id_arr.append(img_id)
                x_min_arr.append(box[0])
                y_min_arr.append(box[1])
                x_max_arr.append(box[2])
                y_max_arr.append(box[3])
                label_arr.append(label)
                confidence_score_arr.append(score)

print("Predictions Complete")
print("End time:" + str(datetime.now() - start))

val_predictions = pd.DataFrame({"image_id": image_id_arr,
                                "x_min": x_min_arr,
                                "y_min": y_min_arr,
                                "x_max": x_max_arr,
                                "y_max": y_max_arr,
                                "label": label_arr,
                                "confidence_score": confidence_score_arr})

# %% --------------------
# validation path
validation_path = f"{VALIDATION_PREDICTION_DIR}/object_detection/predictions"

if not Path(validation_path).exists():
    os.makedirs(validation_path)

# write csv file
# DYNAMIC
val_predictions.to_csv(validation_path + f"/validation_predictions_anchor_sgd.csv", index=False)

# %% --------------------HOLDOUT DATA
# make predictions for holdout data
print("Holdout predictions started")
# start time
start = datetime.now()

# arrays
image_id_arr = []
x_min_arr = []
y_min_arr = []
x_max_arr = []
y_max_arr = []
label_arr = []
confidence_score_arr = []

with torch.no_grad():
    for images, targets in holdout_data_loader:
        # iterate through images and send to device
        images_device = list(image.to(device) for image in images)

        image_ids = []
        for t in targets:
            index = t["image_id"].item()
            image_id = holdout_data_loader.dataset.get_image_id_using_index(index)
            # need image_ids to compare with target data
            image_ids.append(image_id)

        # output is list of dictionary [{boxes:tensor([[xmin, ymin, xmax, ymax], [...]],
        # device=cuda), labels:tensor([15, 11, ...], device=cuda), scores:tensor([0.81, 0.92,
        # ...], device=cuda)},{...}]
        outputs = model(images_device)

        for img_id, output in zip(image_ids, outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                image_id_arr.append(img_id)
                x_min_arr.append(box[0])
                y_min_arr.append(box[1])
                x_max_arr.append(box[2])
                y_max_arr.append(box[3])
                label_arr.append(label)
                confidence_score_arr.append(score)

print("Predictions Complete")
print("End time:" + str(datetime.now() - start))

holdout_predictions = pd.DataFrame({"image_id": image_id_arr,
                                    "x_min": x_min_arr,
                                    "y_min": y_min_arr,
                                    "x_max": x_max_arr,
                                    "y_max": y_max_arr,
                                    "label": label_arr,
                                    "confidence_score": confidence_score_arr})

# %% --------------------
# holdout path
holdout_path = f"{VALIDATION_PREDICTION_DIR}/object_detection/predictions"

if not Path(holdout_path).exists():
    os.makedirs(holdout_path)

# write csv file
# DYNAMIC
holdout_predictions.to_csv(holdout_path + f"/holdout_predictions_anchor_sgd.csv", index=False)
